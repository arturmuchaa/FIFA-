import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)


# ── modal text parser ─────────────────────────────────────────────────────────

def _parse_modal(texts: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "h2h":   {},
        "form":  {"player1": [], "player2": []},
        "stats": {"player1": {}, "player2": {}},
    }
    upper = [t.upper() for t in texts]

    # Head to head
    try:
        idx = next((i for i, t in enumerate(upper) if "HEAD TO HEAD" in t), None)
        if idx is not None:
            chunk = " ".join(texts[idx : idx + 12])
            nums = re.findall(r"\b(\d+)\b", chunk)
            if len(nums) >= 2:
                data["h2h"]["wins_player1"] = int(nums[0])
                data["h2h"]["wins_player2"] = int(nums[1])
            floats = re.findall(r"\b(\d+\.\d+)\b", chunk)
            if floats:
                data["h2h"]["avg_goals_per_match"] = float(floats[0])
    except Exception as exc:
        logger.debug(f"H2H: {exc}")

    # Form
    try:
        idx = next((i for i, t in enumerate(upper) if t == "FORM"), None)
        if idx is not None:
            chunk = texts[idx + 1 : idx + 14]
            wld = [t for t in chunk if re.search(r"[WLD]", t.upper())]
            if len(wld) >= 2:
                data["form"]["player1"] = re.findall(r"[WLD]", wld[0].upper())
                data["form"]["player2"] = re.findall(r"[WLD]", wld[1].upper())
            elif len(wld) == 1:
                data["form"]["player1"] = re.findall(r"[WLD]", wld[0].upper())
    except Exception as exc:
        logger.debug(f"Form: {exc}")

    # Stats
    STAT_MAP = {
        "WINS %": "wins_pct", "WIN %": "wins_pct",
        "GOALS FOR": "goals_for", "GOALS AGAINST": "goals_against",
    }
    try:
        for i, t in enumerate(upper):
            key = STAT_MAP.get(t)
            if key is None:
                continue
            chunk = " ".join(texts[i : i + 4])
            nums = re.findall(r"\b[\d.]+\b", chunk)
            nums = [n for n in nums if "." in n or int(float(n)) <= 200]
            if len(nums) >= 2:
                data["stats"]["player1"][key] = float(nums[0])
                data["stats"]["player2"][key] = float(nums[1])
            elif len(nums) == 1:
                data["stats"]["player1"][key] = float(nums[0])
    except Exception as exc:
        logger.debug(f"Stats: {exc}")

    return data


# ── main ──────────────────────────────────────────────────────────────────────

async def scrape_details(url: str = UPCOMING_URL) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info(f"Details: navigating to {url}")
            await page.goto(url, timeout=30_000)
            await page.wait_for_load_state("networkidle")

            # ── wait for real content ────────────────────────────────────────
            try:
                await page.wait_for_selector("div", timeout=15_000)
            except Exception:
                logger.warning("Details: timeout waiting for div")

            # ── scroll to trigger lazy load ──────────────────────────────────
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2_000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1_000)

            # ── collect clickable match cards ────────────────────────────────
            all_divs = await page.query_selector_all("div")
            logger.info(f"Details: {len(all_divs)} divs total")

            match_cards = []
            for div in all_divs:
                try:
                    text = await div.inner_text()
                    if "VS" not in text.upper():
                        continue
                    if text.upper().count("VS") > 4:
                        continue
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    vs_pos = [i for i, l in enumerate(lines) if l.upper() == "VS"]
                    if not vs_pos:
                        continue
                    vi = vs_pos[0]
                    if vi < 1 or vi >= len(lines) - 1:
                        continue
                    player1 = lines[vi - 1]
                    player2 = lines[vi + 1]
                    if len(player1) < 2 or len(player2) < 2:
                        continue
                    match_cards.append((div, player1, player2))
                except Exception:
                    continue

            # ── debug sample ─────────────────────────────────────────────────
            if match_cards:
                logger.info(f"Details: {len(match_cards)} clickable match cards found")
                try:
                    sample_html = await match_cards[0][0].inner_html()
                    print("CARD SAMPLE:", sample_html[:500])
                except Exception:
                    pass
            else:
                logger.warning("Details: 0 match cards found — check debug_upcoming.html")

            seen_ids: set[str] = set()

            for card_div, player1, player2 in match_cards:
                match_id = hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]
                if match_id in seen_ids:
                    continue

                try:
                    # ── click the card ───────────────────────────────────────
                    await card_div.scroll_into_view_if_needed()
                    await card_div.click(timeout=5_000)

                    # ── wait for modal ───────────────────────────────────────
                    try:
                        await page.wait_for_selector(
                            'div[role="dialog"]',
                            timeout=7_000,
                        )
                        modal_el = await page.query_selector('div[role="dialog"]')
                    except Exception:
                        # fallback: wait for any known modal content
                        try:
                            await page.wait_for_selector(
                                "text=Head to head",
                                timeout=5_000,
                            )
                            modal_el = None
                        except Exception:
                            logger.debug(f"  Modal not found for {player1} vs {player2}, skip")
                            await page.keyboard.press("Escape")
                            await page.wait_for_timeout(600)
                            continue

                    # ── extract modal texts ──────────────────────────────────
                    if modal_el:
                        raw_text = await modal_el.inner_text()
                    else:
                        # modal might be rendered inline — grab whole page text
                        # but only the portion after the click changed
                        raw_text = await page.inner_text("body")

                    modal_texts = [l.strip() for l in raw_text.splitlines() if l.strip()]
                    logger.debug(f"  Modal texts ({len(modal_texts)}): {modal_texts[:6]}")

                    if len(modal_texts) < 4:
                        logger.debug(f"  Modal too short for {player1}, skip")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(600)
                        continue

                    modal_data = _parse_modal(modal_texts)
                    seen_ids.add(match_id)

                    results.append({
                        "match_id": match_id,
                        "player1":  player1,
                        "player2":  player2,
                        **modal_data,
                    })
                    logger.info(f"  ✓ {player1} vs {player2} | h2h={modal_data['h2h']}")

                except Exception as exc:
                    logger.debug(f"  Error for {player1}: {exc}")

                finally:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(1_000)

        except Exception as exc:
            logger.error(f"Details fatal: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details done: {len(results)} enriched")
    return results

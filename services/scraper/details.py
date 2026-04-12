import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

# Confirmed from logs: modal has class "fixed inset-0 z-[60] p-4 overflow-y-auto"
# Wait for CONTENT inside modal (more reliable than waiting for container)
MODAL_CONTENT_SELECTOR = "text=Head to head"
# Container selector — used to extract text after content appeared
MODAL_CONTAINER_SELECTORS = [
    '[class*="inset-0"]',
    '[class*="modal" i]',
    '[class*="Modal"]',
    '[role="dialog"]',
]


# ── Modal text parser ─────────────────────────────────────────────────────────

def _parse_modal(texts: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "h2h":   {},
        "form":  {"player1": [], "player2": []},
        "stats": {"player1": {}, "player2": {}},
    }
    upper = [t.upper() for t in texts]

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
    except Exception as e:
        logger.debug(f"H2H: {e}")

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
    except Exception as e:
        logger.debug(f"Form: {e}")

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
    except Exception as e:
        logger.debug(f"Stats: {e}")

    return data


# ── Main ──────────────────────────────────────────────────────────────────────

async def scrape_details(url: str = UPCOMING_URL) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info(f"Details: navigating to {url}")
            await page.goto(url, timeout=30_000)
            await page.wait_for_load_state("networkidle")

            try:
                await page.wait_for_selector("div", timeout=15_000)
            except Exception:
                pass

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2_000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1_000)

            # ── collect match cards (dedup by player pair) ───────────────────
            all_divs = await page.query_selector_all("div")
            logger.info(f"Details: {len(all_divs)} divs total")

            pair_to_card: dict[tuple, Any] = {}
            for div in all_divs:
                try:
                    text = await div.inner_text()
                    if "VS" not in text.upper():
                        continue
                    if text.upper().count("VS") > 4:
                        continue
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    vs_list = [i for i, l in enumerate(lines) if l.upper() == "VS"]
                    if not vs_list:
                        continue
                    vi = vs_list[0]
                    if vi < 1 or vi >= len(lines) - 1:
                        continue
                    p1, p2 = lines[vi - 1], lines[vi + 1]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    key = (p1, p2)
                    if key not in pair_to_card or len(lines) < pair_to_card[key][1]:
                        pair_to_card[key] = (div, len(lines))
                except Exception:
                    continue

            match_cards = [(div, p1, p2) for (p1, p2), (div, _) in pair_to_card.items()]
            logger.info(f"Details: {len(match_cards)} unique match cards")

            if match_cards:
                try:
                    sample = await match_cards[0][0].inner_html()
                    print("CARD SAMPLE:", sample[:300])
                except Exception:
                    pass

            seen_ids: set[str] = set()

            for idx, (card_div, player1, player2) in enumerate(match_cards):
                match_id = hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]
                if match_id in seen_ids:
                    continue

                modal_closed = True  # assume clean state

                try:
                    # ── FIX 1: ensure no modal is open before clicking ───────
                    existing_h2h = await page.query_selector(MODAL_CONTENT_SELECTOR)
                    if existing_h2h:
                        logger.debug(f"  [{idx}] closing leftover modal before click")
                        await page.keyboard.press("Escape")
                        try:
                            await page.wait_for_selector(
                                MODAL_CONTENT_SELECTOR, state="hidden", timeout=3_000
                            )
                        except Exception:
                            await page.wait_for_timeout(1_500)

                    await card_div.scroll_into_view_if_needed()
                    await page.wait_for_timeout(300)

                    # ── FIX 2: normal click — force=True breaks React events ─
                    await card_div.click(timeout=5_000)
                    logger.debug(f"  [{idx}] clicked {player1} vs {player2}")

                    # ── FIX 3: wait for modal CONTENT, not container ─────────
                    try:
                        await page.wait_for_selector(
                            MODAL_CONTENT_SELECTOR, timeout=8_000
                        )
                    except Exception:
                        logger.info(f"  [{idx}] no modal content for {player1}, skip")
                        modal_closed = False
                        continue

                    # ── FIX 4: extra 600ms for React to finish rendering ─────
                    await page.wait_for_timeout(600)

                    # ── extract from modal container ─────────────────────────
                    modal_el = None
                    for sel in MODAL_CONTAINER_SELECTORS:
                        modal_el = await page.query_selector(sel)
                        if modal_el:
                            logger.debug(f"  [{idx}] modal container: {sel}")
                            break

                    if modal_el:
                        raw_text = await modal_el.inner_text()
                    else:
                        raw_text = await page.inner_text("body")

                    modal_texts = [l.strip() for l in raw_text.splitlines() if l.strip()]
                    logger.info(
                        f"  [{idx}] modal: {len(modal_texts)} lines — "
                        f"{modal_texts[:4]}"
                    )

                    if len(modal_texts) < 4:
                        logger.debug(f"  [{idx}] modal too short, skip")
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
                    logger.warning(f"  [{idx}] error {player1}: {exc}")

                finally:
                    # ── FIX 5: close modal, wait for CONTENT to disappear ───
                    await page.keyboard.press("Escape")
                    try:
                        await page.wait_for_selector(
                            MODAL_CONTENT_SELECTOR, state="hidden", timeout=3_000
                        )
                        logger.debug(f"  [{idx}] modal closed")
                    except Exception:
                        await page.wait_for_timeout(1_500)

        except Exception as exc:
            logger.error(f"Details fatal: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details done: {len(results)} enriched")
    return results

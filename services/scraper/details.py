import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

# ── Modal selectors — tried in order ─────────────────────────────────────────
_MODAL_SELECTORS = [
    'div[role="dialog"]',
    'div.fixed.inset-0',
    '[class*="modal" i]',
    '[class*="Modal"]',
    '[class*="overlay" i]',
    '[class*="drawer" i]',
    '[class*="panel" i]',
    '[class*="popup" i]',
    '[class*="sheet" i]',
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
        logger.debug(f"H2H parse: {e}")

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
        logger.debug(f"Form parse: {e}")

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
        logger.debug(f"Stats parse: {e}")

    return data


# ── Modal finder ──────────────────────────────────────────────────────────────

async def _find_modal(page):
    """
    Try every known modal selector.
    Final fallback: JS finds the highest z-index fixed/absolute element
    that appeared after the click and has enough text content.
    """
    for sel in _MODAL_SELECTORS:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                logger.debug(f"  Modal found via selector: {sel}")
                return el
        except Exception:
            continue

    # JS fallback — find topmost visible overlay by z-index
    el_handle = await page.evaluate_handle("""
    () => {
        let best = null;
        let bestZ = -1;
        for (const el of document.querySelectorAll('*')) {
            const style = window.getComputedStyle(el);
            const pos = style.position;
            if (pos !== 'fixed' && pos !== 'absolute') continue;
            const z = parseInt(style.zIndex) || 0;
            if (z <= bestZ) continue;
            const rect = el.getBoundingClientRect();
            if (rect.width < 100 || rect.height < 100) continue;
            // must have enough text
            const txt = (el.innerText || '').trim();
            if (txt.length < 30) continue;
            best = el;
            bestZ = z;
        }
        return best;
    }
    """)

    try:
        # evaluate_handle returns JSHandle; check it's an element
        tag = await el_handle.evaluate("el => el ? el.tagName : null")
        if tag:
            logger.debug("  Modal found via JS z-index fallback")
            return el_handle
    except Exception:
        pass

    return None


# ── Clickable child finder ────────────────────────────────────────────────────

async def _best_click_target(page, card_div):
    """
    Within card_div prefer: <a>, <button>, or deepest child with cursor:pointer.
    Falls back to card_div itself.
    """
    try:
        handle = await page.evaluate_handle("""
        (card) => {
            // prefer anchor or button
            const link = card.querySelector('a, button');
            if (link) return link;
            // pointer-cursor children
            const all = card.querySelectorAll('*');
            for (const el of all) {
                if (window.getComputedStyle(el).cursor === 'pointer') return el;
            }
            // card itself if it has pointer cursor
            if (window.getComputedStyle(card).cursor === 'pointer') return card;
            return card;
        }
        """, card_div)
        return handle
    except Exception:
        return card_div


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
                logger.warning("Details: wait_for_selector('div') timed out")

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2_000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1_000)

            # ── collect match cards ──────────────────────────────────────────
            all_divs = await page.query_selector_all("div")
            logger.info(f"Details: {len(all_divs)} divs total")

            # dedup by (player1, player2) — keep smallest card (most specific)
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
                    # keep whichever card has fewer lines (more specific)
                    if key not in pair_to_card or len(lines) < pair_to_card[key][1]:
                        pair_to_card[key] = (div, len(lines))
                except Exception:
                    continue

            match_cards = [(div, p1, p2) for (p1, p2), (div, _) in pair_to_card.items()]
            logger.info(f"Details: {len(match_cards)} unique match cards")

            # ── debug: print first card HTML ─────────────────────────────────
            if match_cards:
                try:
                    sample = await match_cards[0][0].inner_html()
                    print("CARD SAMPLE:", sample[:500])
                except Exception:
                    pass

            seen_ids: set[str] = set()

            for idx, (card_div, player1, player2) in enumerate(match_cards):
                match_id = hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]
                if match_id in seen_ids:
                    continue

                try:
                    await card_div.scroll_into_view_if_needed()
                    await page.wait_for_timeout(400)

                    # ── find best click target inside card ───────────────────
                    click_target = await _best_click_target(page, card_div)
                    await click_target.click(timeout=5_000)

                    # ── wait 2s then screenshot ──────────────────────────────
                    await page.wait_for_timeout(2_000)
                    screenshot_path = f"debug_click_{idx}.png"
                    await page.screenshot(path=screenshot_path)
                    logger.info(f"  [{idx}] click done — screenshot: {screenshot_path}")

                    # ── wait for modal via selector ──────────────────────────
                    modal_appeared = False
                    for sel in _MODAL_SELECTORS:
                        try:
                            await page.wait_for_selector(sel, timeout=3_000)
                            modal_appeared = True
                            logger.debug(f"  [{idx}] wait_for_selector matched: {sel}")
                            break
                        except Exception:
                            continue

                    # also accept "Head to head" text appearing
                    if not modal_appeared:
                        try:
                            await page.wait_for_selector("text=Head to head", timeout=3_000)
                            modal_appeared = True
                            logger.debug(f"  [{idx}] wait_for_selector matched: text=Head to head")
                        except Exception:
                            pass

                    if not modal_appeared:
                        logger.info(f"  [{idx}] no modal detected for {player1} — skip")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(600)
                        continue

                    # ── find modal element and extract text ──────────────────
                    modal_el = await _find_modal(page)

                    if modal_el:
                        raw_text = await modal_el.inner_text()
                    else:
                        # last resort: diff the body — grab everything after VS sections
                        raw_text = await page.inner_text("body")

                    modal_texts = [l.strip() for l in raw_text.splitlines() if l.strip()]
                    logger.info(f"  [{idx}] modal texts ({len(modal_texts)}): {modal_texts[:6]}")

                    if len(modal_texts) < 4:
                        logger.debug(f"  [{idx}] modal too short, skip")
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
                    logger.warning(f"  [{idx}] error {player1}: {exc}")

                finally:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(1_000)

        except Exception as exc:
            logger.error(f"Details fatal: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details done: {len(results)} enriched")
    return results

import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\d{{1,2}}\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}"
    rf"|\d{{2}}:\d{{2}}",
    re.IGNORECASE,
)
_JUNK = {"vs", "results", "upcoming matches", "upcoming", "contact",
         "valhalla cup", "head to head", "form", "stats", "home"}

# Modal container confirmed: "fixed inset-0 z-[60] p-4 overflow-y-auto"
# Modal content anchor
MODAL_CONTENT_SEL = "text=Head to head"


# ── Close modal via JS (iframe intercepts keyboard/mouse events) ─────────────

_CLOSE_MODAL_JS = """
() => {
    // 1. Try clicking a visible close/X button inside the modal
    const selectors = [
        'button[aria-label*="close" i]',
        'button[aria-label*="dismiss" i]',
        '[class*="close" i] button',
        '[class*="close" i]',
    ];
    for (const sel of selectors) {
        const btn = document.querySelector(sel);
        if (btn instanceof HTMLElement) { btn.click(); return 'clicked: ' + sel; }
    }

    // 2. Try button containing SVG (X icon) — must click the button, not the SVG
    const svgEl = document.querySelector('button svg');
    if (svgEl) {
        const btn = svgEl.closest('button');
        if (btn instanceof HTMLElement) { btn.click(); return 'clicked svg button'; }
    }

    // 3. Dispatch Escape on document (bypasses iframe focus)
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Escape', code: 'Escape', keyCode: 27,
        bubbles: true, cancelable: true
    }));
    document.dispatchEvent(new KeyboardEvent('keyup', {
        key: 'Escape', code: 'Escape', keyCode: 27,
        bubbles: true, cancelable: true
    }));
    return 'dispatched Escape on document';
}
"""

_FORCE_REMOVE_MODAL_JS = """
() => {
    // Remove ALL fixed overlay divs + disable pointer events
    let count = 0;
    document.querySelectorAll('.fixed.inset-0').forEach(el => {
        el.remove();
        count++;
    });
    // Belt+suspenders: also hide any remaining z-[60] overlays
    document.querySelectorAll('[class*="z-\\\\[60\\\\]"]').forEach(el => {
        el.style.display = 'none';
        el.style.pointerEvents = 'none';
    });
    return count > 0 ? 'removed ' + count : 'not found';
}
"""


async def _close_modal(page) -> None:
    """
    Close modal even when an iframe inside intercepts events.
    Always force-removes the overlay — never relies on selector state.
    Never raises.
    """
    try:
        result = await page.evaluate(_CLOSE_MODAL_JS)
        logger.debug(f"  close_modal JS: {result}")
    except Exception as e:
        logger.debug(f"  close_modal JS error (ignored): {e}")

    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

    # Always force-remove — do NOT return early based on selector.
    # Bug: if modal never opened, selector would appear "hidden" and
    # we'd skip force-remove, leaving a stale overlay for the next card.
    try:
        removed = await page.evaluate(_FORCE_REMOVE_MODAL_JS)
        logger.debug(f"  force remove: {removed}")
    except Exception as e:
        logger.debug(f"  force remove error (ignored): {e}")

    await page.wait_for_timeout(300)


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

            # ── dismiss any widget/overlay already on page ───────────────────
            if await page.query_selector(MODAL_CONTENT_SEL):
                logger.info("Details: pre-existing modal found, closing…")
                await _close_modal(page)

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
                    # Filter candidates: remove junk, dates, match-id lines
                    before = [l for l in lines[:vi]
                              if len(l) >= 2
                              and l.lower() not in _JUNK
                              and not _DATE_RE.search(l)
                              and not l.lower().startswith("match ")]
                    after  = [l for l in lines[vi + 1:]
                              if len(l) >= 2
                              and l.lower() not in _JUNK
                              and not _DATE_RE.search(l)
                              and not l.lower().startswith("match ")]
                    if not before or not after:
                        continue
                    p1 = before[-1]
                    p2 = after[0]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    key = (p1, p2)
                    if key not in pair_to_card or len(lines) < pair_to_card[key][1]:
                        pair_to_card[key] = (div, len(lines))
                except Exception:
                    continue

            match_cards = [(div, p1, p2) for (p1, p2), (div, _) in pair_to_card.items()]
            logger.info(f"Details: {len(match_cards)} unique match cards")

            seen_ids: set[str] = set()

            for idx, (card_div, player1, player2) in enumerate(match_cards):
                match_id = hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]
                if match_id in seen_ids:
                    continue

                try:
                    # ── always clear any stale overlay before click ──────────
                    await _close_modal(page)

                    await card_div.scroll_into_view_if_needed()
                    await page.wait_for_timeout(300)

                    # ── JS click: bypasses Playwright's iframe interception ───
                    # card_div.click() uses coordinates → blocked by iframe overlay
                    # page.evaluate("el => el.click()") dispatches directly on element
                    await page.evaluate("el => el.click()", card_div)
                    logger.debug(f"  [{idx}] clicked {player1} vs {player2}")

                    # ── wait for modal content ───────────────────────────────
                    try:
                        await page.wait_for_selector(MODAL_CONTENT_SEL, timeout=8_000)
                    except Exception:
                        logger.info(f"  [{idx}] no modal for {player1}, skip")
                        continue

                    # ── wait for full render ─────────────────────────────────
                    await page.wait_for_timeout(600)

                    # ── extract modal text ───────────────────────────────────
                    modal_el = (
                        await page.query_selector('[class*="inset-0"]')
                        or await page.query_selector('[class*="modal" i]')
                        or await page.query_selector('[role="dialog"]')
                    )

                    if modal_el:
                        raw_text = await modal_el.inner_text()
                    else:
                        raw_text = await page.inner_text("body")

                    modal_texts = [l.strip() for l in raw_text.splitlines() if l.strip()]
                    logger.info(f"  [{idx}] modal: {len(modal_texts)} lines — {modal_texts[:4]}")

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
                    # ── close via JS (iframe-safe) ───────────────────────────
                    await _close_modal(page)

        except Exception as exc:
            logger.error(f"Details fatal: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details done: {len(results)} enriched")
    return results

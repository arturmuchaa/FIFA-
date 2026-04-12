"""
Details scraper — clicks each match card on the upcoming page,
waits for the modal, extracts H2H / Form / Stats.

Click strategy:
  1. JS finds VS-containing containers → returns bounding boxes + player names
  2. page.mouse.click(cx, cy) for each box  (no CSS selector needed)
  3. wait_for_selector('[role="dialog"], [class*="modal"], [class*="Modal"]')
  4. JS extracts modal leaf texts
  5. Python parses H2H / Form / Stats from text array
  6. Escape → 1s delay → next match
"""

import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

# ── JavaScript: find clickable match containers ──────────────────────────────
_FIND_CONTAINERS_JS = """
() => {
    function leafTexts(el) {
        const out = [];
        const tw = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
        let n;
        while ((n = tw.nextNode())) {
            const t = n.textContent.trim();
            if (t.length > 0) out.push(t);
        }
        return out;
    }

    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    const vsNodes = [];
    let n;
    while ((n = tw.nextNode())) {
        if (n.textContent.trim() === 'VS') vsNodes.push(n);
    }

    const containers = [];
    const seenKeys = new Set();

    for (const vsNode of vsNodes) {
        let el = vsNode.parentElement;

        while (el && el !== document.body) {
            const texts = leafTexts(el);

            if (texts.length >= 4 && texts.length <= 28) {
                const rect = el.getBoundingClientRect();
                if (rect.width > 10 && rect.height > 10) {
                    const key = `${Math.round(rect.x)},${Math.round(rect.y)}`;
                    if (!seenKeys.has(key)) {
                        seenKeys.add(key);

                        // extract player names: items before and after VS
                        const vsIdx = texts.indexOf('VS');
                        const before = texts.slice(0, vsIdx).filter(t =>
                            t !== 'VS' && t.length > 1 && !/^\\d+[-\\u2013]\\d+$/.test(t)
                        );
                        const after = texts.slice(vsIdx + 1).filter(t =>
                            t !== 'VS' && t.length > 1 && !/^\\d+[-\\u2013]\\d+$/.test(t)
                        );

                        containers.push({
                            cx: rect.x + rect.width / 2,
                            cy: rect.y + rect.height / 2,
                            player1: before.length >= 2 ? before[before.length - 2] : (before[0] || ''),
                            player2: after.length >= 1 ? after[0] : '',
                            texts: texts,
                        });
                    }
                }
                break;
            }
            if (texts.length > 28) break;
            el = el.parentElement;
        }
    }
    return containers;
}
"""

# ── JavaScript: extract modal leaf texts ────────────────────────────────────
_MODAL_TEXTS_JS = """
() => {
    function leafTexts(el) {
        const out = [];
        const tw = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
        let n;
        while ((n = tw.nextNode())) {
            const t = n.textContent.trim();
            if (t.length > 0) out.push(t);
        }
        return out;
    }

    // Try common modal selectors
    const modal =
        document.querySelector('[role="dialog"]') ||
        document.querySelector('[class*="modal" i]') ||
        document.querySelector('[class*="Modal"]') ||
        document.querySelector('[class*="overlay" i]') ||
        document.querySelector('[class*="Overlay"]') ||
        document.querySelector('[class*="popup" i]') ||
        document.querySelector('[class*="drawer" i]');

    if (modal) {
        return { found: true, texts: leafTexts(modal) };
    }

    // Fallback: return full body texts (modal might be inline)
    return { found: false, texts: leafTexts(document.body) };
}
"""

# ── modal text parser ────────────────────────────────────────────────────────

def _parse_modal(texts: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "h2h":   {},
        "form":  {"player1": [], "player2": []},
        "stats": {"player1": {}, "player2": {}},
    }

    upper = [t.upper() for t in texts]

    # ── Head to head ────────────────────────────────────────────────────────
    try:
        h2h_idx = next((i for i, t in enumerate(upper) if "HEAD TO HEAD" in t), None)
        if h2h_idx is not None:
            chunk = texts[h2h_idx : h2h_idx + 12]
            nums = re.findall(r"\b(\d+)\b", " ".join(chunk))
            if len(nums) >= 2:
                data["h2h"]["wins_player1"] = int(nums[0])
                data["h2h"]["wins_player2"] = int(nums[1])
            floats = re.findall(r"\b(\d+\.\d+)\b", " ".join(chunk))
            if floats:
                data["h2h"]["avg_goals_per_match"] = float(floats[0])
    except Exception as exc:
        logger.debug(f"H2H: {exc}")

    # ── Form ─────────────────────────────────────────────────────────────────
    try:
        form_idx = next((i for i, t in enumerate(upper) if t == "FORM"), None)
        if form_idx is not None:
            chunk = texts[form_idx + 1 : form_idx + 14]
            wld_blocks = [t for t in chunk if re.search(r"[WLD]", t.upper())]
            if len(wld_blocks) >= 2:
                data["form"]["player1"] = re.findall(r"[WLD]", wld_blocks[0].upper())
                data["form"]["player2"] = re.findall(r"[WLD]", wld_blocks[1].upper())
            elif len(wld_blocks) == 1:
                data["form"]["player1"] = re.findall(r"[WLD]", wld_blocks[0].upper())
    except Exception as exc:
        logger.debug(f"Form: {exc}")

    # ── Stats ────────────────────────────────────────────────────────────────
    STAT_MAP = {
        "WINS %":        "wins_pct",
        "WIN %":         "wins_pct",
        "GOALS FOR":     "goals_for",
        "GOALS AGAINST": "goals_against",
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


# ── main ─────────────────────────────────────────────────────────────────────

async def scrape_details(url: str = UPCOMING_URL) -> list[dict[str, Any]]:
    """
    Returns list of dicts:
      {match_id, player1, player2, h2h, form, stats}
    """
    results: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info(f"Details: navigating to {url}")
            await page.goto(url, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            # ── find match containers via JS ────────────────────────────────
            containers: list[dict] = await page.evaluate(_FIND_CONTAINERS_JS)
            logger.info(f"Details: found {len(containers)} match containers")

            for idx, c in enumerate(containers):
                player1 = c.get("player1", "")
                player2 = c.get("player2", "")
                cx      = c.get("cx", 0)
                cy      = c.get("cy", 0)

                if not player1:
                    logger.debug(f"  [{idx}] no player1, skip")
                    continue

                match_id = hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]

                try:
                    # ── KROK 4: click via mouse coordinates ─────────────────
                    logger.debug(f"  [{idx}] clicking {player1} vs {player2} @ ({cx:.0f},{cy:.0f})")
                    await page.mouse.click(cx, cy)

                    # ── KROK 5: wait for modal ───────────────────────────────
                    try:
                        await page.wait_for_selector(
                            '[role="dialog"], [class*="modal" i], [class*="Modal"], '
                            '[class*="overlay" i], [class*="popup" i], [class*="drawer" i], '
                            'text=Head to head',
                            timeout=6_000,
                        )
                    except Exception:
                        logger.debug(f"  [{idx}] modal timeout for {player1}, skip")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(600)
                        continue

                    # ── KROK 6: extract modal via JS ─────────────────────────
                    modal_result: dict = await page.evaluate(_MODAL_TEXTS_JS)
                    modal_texts = modal_result.get("texts", [])
                    modal_found = modal_result.get("found", False)

                    logger.debug(
                        f"  [{idx}] modal {'element' if modal_found else 'body fallback'}: "
                        f"{len(modal_texts)} leaf texts"
                    )

                    if len(modal_texts) < 5:
                        logger.debug(f"  [{idx}] modal too short, skip")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(600)
                        continue

                    modal_data = _parse_modal(modal_texts)

                    results.append({
                        "match_id": match_id,
                        "player1":  player1,
                        "player2":  player2,
                        **modal_data,
                    })
                    logger.info(f"  ✓ {player1} vs {player2} — H2H: {modal_data['h2h']}")

                except Exception as exc:
                    logger.debug(f"  [{idx}] error: {exc}")

                finally:
                    # ── KROK 7: close modal ──────────────────────────────────
                    await page.keyboard.press("Escape")
                    # ── KROK 8: delay ────────────────────────────────────────
                    await page.wait_for_timeout(1_000)

        except Exception as exc:
            logger.error(f"Details fatal: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details done: {len(results)} matches enriched")
    return results

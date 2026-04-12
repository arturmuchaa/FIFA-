"""
Upcoming scraper — drafted.gg/valhalla-cup/upcoming-matches

Strategy: identical to results.py but anchors on "VS" text nodes
instead of score nodes.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
logger = logging.getLogger(__name__)

_JUNK = {
    "vs", "results", "upcoming matches", "upcoming", "contact",
    "valhalla cup", "head to head", "form", "stats", "home",
}

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\d{{1,2}}\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}"
    rf"|\d{{2}}:\d{{2}}",
    re.IGNORECASE,
)

# ── JavaScript extractor ──────────────────────────────────────────────────────
_EXTRACT_JS = """
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

    // Find text nodes whose content is exactly "VS"
    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    const vsNodes = [];
    let n;
    while ((n = tw.nextNode())) {
        if (n.textContent.trim() === 'VS') vsNodes.push(n);
    }

    const cards = [];
    const seenKeys = new Set();

    for (const vsNode of vsNodes) {
        let el = vsNode.parentElement;
        let prev = el;

        while (el && el !== document.body) {
            const texts = leafTexts(el);

            if (texts.length >= 4 && texts.length <= 28) {
                const key = texts.slice(0, 4).join('|');
                if (!seenKeys.has(key)) {
                    seenKeys.add(key);
                    cards.push(texts);
                }
                break;
            }
            if (texts.length > 28) {
                const prevTexts = leafTexts(prev);
                if (prevTexts.length >= 4) {
                    const key = prevTexts.slice(0, 4).join('|');
                    if (!seenKeys.has(key)) {
                        seenKeys.add(key);
                        cards.push(prevTexts);
                    }
                }
                break;
            }
            prev = el;
            el = el.parentElement;
        }
    }
    return cards;
}
"""


def _is_junk(t: str) -> bool:
    return t.lower() in _JUNK or len(t.strip()) < 2


def _make_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _parse_card(texts: list[str]) -> dict[str, Any] | None:
    """
    Turn leaf-text array into a match dict.
    Layout: player1, team1, [match_id/date], VS, player2, team2, [date]
    """
    # find VS index
    vs_idx = next((i for i, t in enumerate(texts) if t.strip() == "VS"), None)
    if vs_idx is None:
        return None

    before = [t for t in texts[:vs_idx] if not _is_junk(t)]
    after  = [t for t in texts[vs_idx + 1:] if not _is_junk(t)]

    if not before or not after:
        return None

    # remove date strings from player candidates
    non_date_before = [t for t in before if not _DATE_RE.search(t) and not t.lower().startswith("match ")]
    non_date_after  = [t for t in after  if not _DATE_RE.search(t) and not t.lower().startswith("match ")]

    if not non_date_before or not non_date_after:
        return None

    player1 = non_date_before[-2] if len(non_date_before) >= 2 else non_date_before[-1]
    team1   = non_date_before[-1] if len(non_date_before) >= 2 else ""
    player2 = non_date_after[0]
    team2   = non_date_after[1] if len(non_date_after) >= 2 else ""

    date = next((t for t in texts if _DATE_RE.search(t)), "")
    raw_id = next((t for t in texts if t.lower().startswith("match ")), f"{player1}_{player2}")

    return {
        "match_id": _make_id(f"up_{raw_id}"),
        "raw_id":   raw_id,
        "player1":  player1,
        "team1":    team1,
        "player2":  player2,
        "team2":    team2,
        "date":     date,
        "source":   "upcoming",
        "status":   "scheduled",
    }


async def scrape_upcoming() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info("Upcoming: navigating…")
            await page.goto(UPCOMING_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            # ── debug: save HTML ────────────────────────────────────────────
            html = await page.content()
            Path("debug_upcoming.html").write_text(html, encoding="utf-8")
            logger.info(f"Upcoming: HTML saved ({len(html)} chars)")
            logger.info(f"Upcoming: HTML preview → {html[:300]!r}")

            # ── JS extraction ───────────────────────────────────────────────
            card_texts: list[list[str]] = await page.evaluate(_EXTRACT_JS)
            logger.info(f"Upcoming: JS found {len(card_texts)} candidate cards")

            for i, texts in enumerate(card_texts):
                logger.debug(f"  card[{i}]: {texts}")

            seen_ids: set[str] = set()

            for texts in card_texts:
                result = _parse_card(texts)
                if result is None:
                    continue
                if result["match_id"] in seen_ids:
                    continue
                seen_ids.add(result["match_id"])
                matches.append(result)
                logger.info(f"  + {result['player1']} vs {result['player2']} | {result['date']}")

        except Exception as exc:
            logger.error(f"Upcoming scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Upcoming done: {len(matches)} matches")
    return matches

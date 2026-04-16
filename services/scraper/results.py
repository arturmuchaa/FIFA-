"""
Results scraper — drafted.gg/valhalla-cup/results

Strategy:
  1. page.content() → save HTML for debugging
  2. page.evaluate() with JS TreeWalker — find score leaf nodes,
     walk up to the tightest container that looks like a match card,
     return leaf text arrays in DOM order
  3. Python parses each text array
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

RESULTS_URL = "https://drafted.gg/valhalla-cup/results"
logger = logging.getLogger(__name__)

# ── score pattern ────────────────────────────────────────────────────────────
_SCORE_FULL = re.compile(r"^(\d+)\s*[-–]\s*(\d+)$")   # "3-1" / "3 – 1"
_NUM_ONLY   = re.compile(r"^\d+$")
_SEP_ONLY   = re.compile(r"^[-–]$")

# junk nav labels to ignore in player slots
_JUNK = {
    "vs", "results", "upcoming matches", "upcoming", "contact",
    "valhalla cup", "head to head", "form", "stats", "home", "match history",
}

# ── JavaScript that extracts raw text arrays ─────────────────────────────────
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

    const scoreRe = /^\\d+\\s*[-\\u2013]\\s*\\d+$/;
    const numRe   = /^\\d+$/;
    const sepRe   = /^[-\\u2013]$/;

    // Collect score text nodes
    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    const scoreNodes = [];
    let n;
    while ((n = tw.nextNode())) {
        const t = n.textContent.trim();
        if (scoreRe.test(t)) scoreNodes.push(n);
    }

    const cards = [];
    const seenKeys = new Set();

    for (const sn of scoreNodes) {
        let el = sn.parentElement;
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
                // too big — use previous smaller container if it had ≥4 items
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_score(texts: list[str]) -> tuple[int, int, int] | None:
    """
    Return (index_in_texts, goals1, goals2) or None.
    Handles:
      - "3-1" in a single text node
      - "3", "-", "1" across three text nodes
    """
    for i, t in enumerate(texts):
        m = _SCORE_FULL.match(t)
        if m:
            return i, int(m.group(1)), int(m.group(2))

    for i in range(len(texts) - 2):
        if (
            _NUM_ONLY.match(texts[i])
            and _SEP_ONLY.match(texts[i + 1])
            and _NUM_ONLY.match(texts[i + 2])
        ):
            return i, int(texts[i]), int(texts[i + 2])

    return None


def _is_junk(t: str) -> bool:
    return t.lower() in _JUNK or len(t.strip()) < 2


def _make_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\d{{1,2}}\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}",
    re.IGNORECASE,
)


def _find_date(texts: list[str]) -> str:
    return next((t for t in texts if _DATE_RE.search(t)), "")


def _parse_card(texts: list[str]) -> dict[str, Any] | None:
    """
    Turn a leaf-text array into a match dict.
    Expected layout (flexible): player1, team1, [match_id], [date], score, player2, team2
    """
    score_result = _find_score(texts)
    if score_result is None:
        return None

    score_idx, g1, g2 = score_result
    score_str = texts[score_idx] if _SCORE_FULL.match(texts[score_idx]) else f"{g1}-{g2}"

    before = [t for t in texts[:score_idx] if not _is_junk(t)]
    after  = [t for t in texts[score_idx + (3 if score_idx + 2 < len(texts) and _SEP_ONLY.match(texts[score_idx + 1]) else 1):] if not _is_junk(t)]

    if not before or not after:
        return None

    # player1 = last non-junk before score that isn't a date or match-id
    date = _find_date(texts)
    candidates_before = [t for t in before if not _DATE_RE.search(t) and not t.lower().startswith("match ")]
    candidates_after  = [t for t in after  if not _DATE_RE.search(t) and not t.lower().startswith("match ")]

    if not candidates_before or not candidates_after:
        return None

    player1 = candidates_before[-2] if len(candidates_before) >= 2 else candidates_before[-1]
    team1   = candidates_before[-1] if len(candidates_before) >= 2 else ""
    player2 = candidates_after[0]
    team2   = candidates_after[1] if len(candidates_after) >= 2 else ""

    raw_id = next((t for t in texts if t.lower().startswith("match ")), f"{player1}_{player2}_{score_str}")

    return {
        "match_id":    _make_id(raw_id),
        "raw_id":      raw_id,
        "player1":     player1,
        "team1":       team1,
        "player2":     player2,
        "team2":       team2,
        "score":       score_str,
        "goals1":      g1,
        "goals2":      g2,
        "total_goals": g1 + g2,
        "date":        date,
        "source":      "results",
        "status":      "finished",
    }


# ── main ─────────────────────────────────────────────────────────────────────

async def scrape_results() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-gpu", "--single-process"],
        )
        context = await browser.new_context()
        page = await context.new_page()

        try:
            logger.info("Results: navigating…")
            await page.goto(RESULTS_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            # ── debug: save raw HTML ────────────────────────────────────────
            html = await page.content()
            Path("debug_results.html").write_text(html, encoding="utf-8")
            logger.info(f"Results: HTML saved ({len(html)} chars)")
            logger.info(f"Results: HTML preview → {html[:300]!r}")

            # ── extract match card text arrays via JS ───────────────────────
            card_texts: list[list[str]] = await page.evaluate(_EXTRACT_JS)
            logger.info(f"Results: JS found {len(card_texts)} candidate cards")

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
                logger.info(
                    f"  + {result['player1']} vs {result['player2']}"
                    f" | {result['score']} | {result['date']}"
                )

        except Exception as exc:
            logger.error(f"Results scraper error: {exc}")
        finally:
            await context.close()
            await browser.close()

    logger.info(f"Results done: {len(matches)} matches")
    return matches

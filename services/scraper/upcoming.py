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


def _is_junk(t: str) -> bool:
    return t.lower() in _JUNK or len(t.strip()) < 2


def _make_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _leaf_texts_from_html(html: str) -> list[str]:
    """Strip tags and return non-empty text tokens."""
    clean = re.sub(r"<[^>]+>", " ", html)
    return [t.strip() for t in re.split(r"\s{2,}|\n", clean) if t.strip()]


def _parse_texts(texts: list[str]) -> dict[str, Any] | None:
    vs_idx = next((i for i, t in enumerate(texts) if t.strip().upper() == "VS"), None)
    if vs_idx is None:
        return None

    before = [t for t in texts[:vs_idx] if not _is_junk(t)]
    after  = [t for t in texts[vs_idx + 1:] if not _is_junk(t)]

    if not before or not after:
        return None

    nb = [t for t in before if not _DATE_RE.search(t) and not t.lower().startswith("match ")]
    na = [t for t in after  if not _DATE_RE.search(t) and not t.lower().startswith("match ")]

    if not nb or not na:
        return None

    player1 = nb[-2] if len(nb) >= 2 else nb[-1]
    team1   = nb[-1] if len(nb) >= 2 else ""
    player2 = na[0]
    team2   = na[1] if len(na) >= 2 else ""

    date   = next((t for t in texts if _DATE_RE.search(t)), "")
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

            # ── wait for real DOM content ────────────────────────────────────
            try:
                await page.wait_for_selector("div", timeout=15_000)
            except Exception:
                logger.warning("Upcoming: timeout waiting for div elements")

            # ── trigger lazy-load via scroll ─────────────────────────────────
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2_000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1_000)

            # ── save debug HTML ──────────────────────────────────────────────
            html = await page.content()
            Path("debug_upcoming.html").write_text(html, encoding="utf-8")
            logger.info(f"Upcoming: HTML {len(html)} chars saved → debug_upcoming.html")

            # ── collect all divs, filter by VS + content ─────────────────────
            all_divs = await page.query_selector_all("div")
            logger.info(f"Upcoming: {len(all_divs)} divs found")

            match_cards = []
            for div in all_divs:
                try:
                    inner = await div.inner_html()
                    text  = await div.inner_text()
                    if "VS" not in text.upper():
                        continue
                    # must have player-like content on both sides of VS
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    vs_pos = [i for i, l in enumerate(lines) if l.upper() == "VS"]
                    if not vs_pos:
                        continue
                    vi = vs_pos[0]
                    if vi < 1 or vi >= len(lines) - 1:
                        continue
                    # skip huge wrapper divs (contain many VS)
                    if text.upper().count("VS") > 4:
                        continue
                    match_cards.append((lines, inner))
                except Exception:
                    continue

            # ── debug: print sample card ─────────────────────────────────────
            if match_cards:
                logger.info(f"Upcoming: {len(match_cards)} VS-containing divs")
                print("CARD SAMPLE:", match_cards[0][1][:500])
            else:
                logger.warning("Upcoming: 0 VS-containing divs — check debug_upcoming.html")

            seen_ids: set[str] = set()

            for lines, _inner in match_cards:
                result = _parse_texts(lines)
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

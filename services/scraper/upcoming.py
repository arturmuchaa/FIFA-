"""
Scraper for Valhalla Cup upcoming matches page.

Identyczna logika co results.py — anchor "Match XXXX".

Struktura okna wokół "Match XXXX":
  player1          ← i-2
  team1            ← i-1
  Match XXXX       ← i  (ANCHOR)
  date             ← i+1
  player2          ← i+2
  team2            ← i+3

Brak score — mecz jeszcze nie rozegrany.
"""

import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"

logger = logging.getLogger(__name__)

_SKIP = {
    "results", "upcoming matches", "upcoming", "contact", "valhalla cup",
    "vs", "head to head", "form", "stats", "match history", "home",
}

_MATCH_RE = re.compile(r"^Match\s+\S", re.IGNORECASE)

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\d{{1,2}}\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}:\d{{2}}",
    re.IGNORECASE,
)


def _is_junk(line: str) -> bool:
    return line.lower() in _SKIP or len(line) < 2


def _make_id(raw_match_id: str) -> str:
    return hashlib.md5(f"up_{raw_match_id}".encode()).hexdigest()[:12]


async def scrape_upcoming() -> list[dict[str, Any]]:
    """
    Returns scheduled matches:
      {match_id, raw_id, player1, team1, player2, team2, date, source, status}
    """
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info("Upcoming scraper: navigating…")
            await page.goto(UPCOMING_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            body_text = await page.inner_text("body")
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]

            logger.info(f"Upcoming: {len(lines)} body lines")
            for idx, l in enumerate(lines[:30]):
                logger.debug(f"  [{idx:02d}] {l!r}")

            seen_ids: set[str] = set()

            for i, line in enumerate(lines):
                if not _MATCH_RE.match(line):
                    continue

                raw_match_id = line
                lo = max(0, i - 4)
                hi = min(len(lines) - 1, i + 5)
                window = lines[lo : hi + 1]
                wi = i - lo

                if wi < 2:
                    continue

                player1 = window[wi - 2]
                team1   = window[wi - 1]

                if _is_junk(player1) or _is_junk(team1):
                    continue

                if wi + 1 >= len(window):
                    continue

                date = window[wi + 1]
                if not _DATE_RE.search(date):
                    if wi + 2 < len(window):
                        date = window[wi + 2]
                    if not _DATE_RE.search(date):
                        logger.debug(f"No date near {raw_match_id!r}, skip")
                        continue

                date_wi = window.index(date, wi + 1)
                if date_wi + 1 >= len(window):
                    continue

                player2 = window[date_wi + 1]
                team2   = window[date_wi + 2] if date_wi + 2 < len(window) else ""

                if _is_junk(player2):
                    continue

                match_id = _make_id(raw_match_id)
                if match_id in seen_ids:
                    continue
                seen_ids.add(match_id)

                matches.append({
                    "match_id": match_id,
                    "raw_id":   raw_match_id,
                    "player1":  player1,
                    "team1":    team1,
                    "player2":  player2,
                    "team2":    team2,
                    "date":     date,
                    "source":   "upcoming",
                    "status":   "scheduled",
                })
                logger.info(f"  + {player1} vs {player2} | {date}")

        except Exception as exc:
            logger.error(f"Upcoming scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Upcoming scraper done: {len(matches)} matches")
    return matches

"""
Scraper for Valhalla Cup upcoming matches page.

Identyczna metoda co results.py — pobierz inner_text("body") i parsuj linie.

Struktura meczu w tekście:
  player1
  team1
  match_id
  date
  VS
  player2
  team2
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"

logger = logging.getLogger(__name__)


def _stable_id(raw_id: str, player1: str, player2: str) -> str:
    if raw_id and len(raw_id) > 3 and not re.search(r"[WLD%]", raw_id):
        return hashlib.md5(f"up_{raw_id}".encode()).hexdigest()[:12]
    return hashlib.md5(f"up_{player1}_{player2}".encode()).hexdigest()[:12]


async def scrape_upcoming() -> list[dict[str, Any]]:
    """
    Returns list of scheduled matches:
      {match_id, player1, team1, player2, team2, date, source}
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

            # ── pobierz cały tekst po wyrenderowaniu JS ──────────────────────
            body_text = await page.inner_text("body")
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]

            logger.info(f"Upcoming: {len(lines)} lines of body text")

            seen_ids: set[str] = set()

            for i, line in enumerate(lines):
                if line != "VS":
                    continue

                if i < 4 or i + 2 >= len(lines):
                    continue

                player1 = lines[i - 4]
                team1   = lines[i - 3]
                raw_id  = lines[i - 2]
                date    = lines[i - 1]
                player2 = lines[i + 1]
                team2   = lines[i + 2]

                # sanity check
                if player1 in ("VS", "HEAD TO HEAD", "FORM") or player2 in ("VS",):
                    continue

                match_id = _stable_id(raw_id, player1, player2)

                if match_id in seen_ids:
                    continue
                seen_ids.add(match_id)

                matches.append({
                    "match_id": match_id,
                    "player1":  player1,
                    "team1":    team1,
                    "player2":  player2,
                    "team2":    team2,
                    "date":     date,
                    "source":   "upcoming",
                    "status":   "scheduled",
                })

        except Exception as exc:
            logger.error(f"Upcoming scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Upcoming scraper finished: {len(matches)} matches")
    return matches

"""
Scraper for Valhalla Cup upcoming matches page.
Extracts scheduled matches (player1, player2, date, match_id).
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"

logger = logging.getLogger(__name__)


def _make_upcoming_id(player1: str, player2: str, date_str: str) -> str:
    raw = f"upcoming_{player1}_{player2}_{date_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


async def scrape_upcoming() -> list[dict[str, Any]]:
    """
    Visit the upcoming-matches page and return a list of scheduled matches.
    Each dict: {match_id, player1, player2, date, source}
    """
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info("Navigating to upcoming matches page…")
            await page.goto(UPCOMING_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            match_containers = page.locator("div:has-text('VS')")
            count = await match_containers.count()
            logger.info(f"Found {count} potential upcoming rows")

            seen_ids: set[str] = set()

            for i in range(count):
                try:
                    container = match_containers.nth(i)
                    full_text = await container.inner_text()

                    if full_text.count("VS") > 3:
                        continue

                    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

                    vs_idx = next(
                        (j for j, l in enumerate(lines) if l.upper() == "VS"), None
                    )
                    if vs_idx is None or vs_idx == 0 or vs_idx >= len(lines) - 1:
                        continue

                    player1 = lines[vs_idx - 1]
                    player2 = lines[vs_idx + 1]

                    date_line = next(
                        (
                            l
                            for l in lines
                            if re.search(
                                r"\d{1,2}[\s./]\w+|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", l
                            )
                        ),
                        datetime.utcnow().strftime("%Y-%m-%d"),
                    )

                    match_id = _make_upcoming_id(player1, player2, date_line)
                    if match_id in seen_ids:
                        continue
                    seen_ids.add(match_id)

                    matches.append(
                        {
                            "match_id": match_id,
                            "player1": player1,
                            "player2": player2,
                            "date": date_line.strip(),
                            "source": "upcoming",
                        }
                    )

                except Exception as exc:
                    logger.debug(f"Skipping upcoming row {i}: {exc}")
                    continue

        except Exception as exc:
            logger.error(f"Upcoming scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Upcoming scraper finished: {len(matches)} matches")
    return matches

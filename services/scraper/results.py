"""
Scraper for Valhalla Cup results page.
Extracts completed match history (player1, player2, score, date, match_id).
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from typing import Any

from playwright.async_api import async_playwright

RESULTS_URL = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)


def _make_match_id(player1: str, player2: str, score: str, date_str: str) -> str:
    raw = f"{player1}_{player2}_{score}_{date_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _parse_score(score_text: str) -> tuple[int | None, int | None]:
    """Parse '3-1' → (3, 1). Returns (None, None) on failure."""
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", score_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


async def scrape_results() -> list[dict[str, Any]]:
    """
    Visit the results page and return a list of completed matches.
    Each dict: {match_id, player1, player2, score, goals1, goals2, date, source}
    """
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info("Navigating to results page…")
            await page.goto(RESULTS_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            # ── locate match rows ───────────────────────────────────────────
            # Each completed match row contains a "VS" divider between the two
            # player name spans.  We grab every container that holds "VS".
            match_containers = page.locator("div:has-text('VS')")
            count = await match_containers.count()
            logger.info(f"Found {count} potential result rows")

            seen_ids: set[str] = set()

            for i in range(count):
                try:
                    container = match_containers.nth(i)
                    full_text = await container.inner_text()

                    # skip containers that are just wrappers with many VS
                    if full_text.count("VS") > 3:
                        continue

                    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

                    # Extract players around "VS"
                    vs_idx = next(
                        (j for j, l in enumerate(lines) if l.upper() == "VS"), None
                    )
                    if vs_idx is None or vs_idx == 0 or vs_idx >= len(lines) - 1:
                        continue

                    player1 = lines[vs_idx - 1]
                    player2 = lines[vs_idx + 1]

                    # Find score (pattern d-d)
                    score_line = next(
                        (l for l in lines if re.search(r"\d\s*[-–]\s*\d", l)), None
                    )
                    if score_line is None:
                        continue

                    goals1, goals2 = _parse_score(score_line)
                    if goals1 is None:
                        continue

                    # Date: look for something like "12 Apr" / "2024-04-12"
                    date_line = next(
                        (
                            l
                            for l in lines
                            if re.search(
                                r"\d{1,2}[\s./]\w+|\d{4}-\d{2}-\d{2}", l
                            )
                            and l != score_line
                        ),
                        datetime.utcnow().strftime("%Y-%m-%d"),
                    )

                    match_id = _make_match_id(player1, player2, score_line, date_line)
                    if match_id in seen_ids:
                        continue
                    seen_ids.add(match_id)

                    matches.append(
                        {
                            "match_id": match_id,
                            "player1": player1,
                            "player2": player2,
                            "score": score_line.strip(),
                            "goals1": goals1,
                            "goals2": goals2,
                            "total_goals": goals1 + goals2,
                            "date": date_line.strip(),
                            "source": "results",
                        }
                    )

                except Exception as exc:
                    logger.debug(f"Skipping row {i}: {exc}")
                    continue

        except Exception as exc:
            logger.error(f"Results scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Results scraper finished: {len(matches)} matches")
    return matches

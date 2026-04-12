"""
Scraper for Valhalla Cup results page.

Strona renderuje dane dynamicznie przez JS — locatory DOM zwracają 0 elementów.
Rozwiązanie: pobierz cały tekst body po wyrenderowaniu, parsuj linia po linii.

Struktura każdego meczu w tekście:
  player1
  team1
  match_id        ← unikalny ID ze strony
  date
  VS
  player2
  team2
  score           ← np. "3 - 1"  (może być kilka linii dalej)
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Any

from playwright.async_api import async_playwright

RESULTS_URL = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

# Score pattern: "3 - 1" / "3-1" / "3–1"
_SCORE_RE = re.compile(r"^(\d+)\s*[-–]\s*(\d+)$")


def _is_score(line: str) -> tuple[int, int] | None:
    m = _SCORE_RE.match(line.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _stable_id(raw_id: str, player1: str, player2: str) -> str:
    """
    Prefer the match_id from the page (line i-2 relative to VS).
    Fall back to MD5 of players if it doesn't look like an ID.
    """
    if raw_id and len(raw_id) > 3 and not re.search(r"[WLD%]", raw_id):
        return hashlib.md5(raw_id.encode()).hexdigest()[:12]
    return hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]


async def scrape_results() -> list[dict[str, Any]]:
    """
    Returns list of completed matches:
      {match_id, player1, team1, player2, team2,
       score, goals1, goals2, total_goals, date, source}
    """
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            logger.info("Results scraper: navigating…")
            await page.goto(RESULTS_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            # ── pobierz cały tekst po wyrenderowaniu JS ──────────────────────
            body_text = await page.inner_text("body")
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]

            logger.info(f"Results: {len(lines)} lines of body text")

            seen_ids: set[str] = set()

            for i, line in enumerate(lines):
                if line != "VS":
                    continue

                # ── graceful bounds check ────────────────────────────────────
                if i < 4 or i + 2 >= len(lines):
                    continue

                # ── wyciągnij pola ───────────────────────────────────────────
                player1  = lines[i - 4]
                team1    = lines[i - 3]
                raw_id   = lines[i - 2]   # match_id ze strony
                date     = lines[i - 1]
                player2  = lines[i + 1]
                team2    = lines[i + 2]

                # ── wynik: szukaj w kolejnych ~5 liniach po VS ───────────────
                goals1 = goals2 = None
                score_str = ""
                for offset in range(3, 8):
                    if i + offset >= len(lines):
                        break
                    parsed = _is_score(lines[i + offset])
                    if parsed:
                        goals1, goals2 = parsed
                        score_str = lines[i + offset].strip()
                        break

                if goals1 is None:
                    # no score found → skip (match may not have score yet)
                    continue

                # basic sanity: player names should look like names, not keywords
                if player1 in ("VS", "HEAD TO HEAD", "FORM") or player2 in ("VS",):
                    continue

                match_id = _stable_id(raw_id, player1, player2)

                if match_id in seen_ids:
                    continue
                seen_ids.add(match_id)

                matches.append({
                    "match_id":    match_id,
                    "player1":     player1,
                    "team1":       team1,
                    "player2":     player2,
                    "team2":       team2,
                    "score":       score_str,
                    "goals1":      goals1,
                    "goals2":      goals2,
                    "total_goals": goals1 + goals2,
                    "date":        date,
                    "source":      "results",
                    "status":      "finished",
                })

        except Exception as exc:
            logger.error(f"Results scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Results scraper finished: {len(matches)} matches")
    return matches

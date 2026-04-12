"""
Scraper for Valhalla Cup results page.

Strona renderuje JS dynamicznie — body text zawiera ~85 linii.
W wynikach NIE MA "VS" — anchor to "Match XXXX".

Struktura okna wokół "Match XXXX":
  ...
  player1          ← 2 linie przed match_id
  team1            ← 1 linia przed match_id
  Match XXXX       ← ANCHOR (linia i)
  date             ← i+1
  player2          ← i+2
  team2            ← i+3
  score            ← i+4 .. i+7  (np. "3 - 1" / "3-1")
  ...

Ignorowane linie (SKIP_LINES): nawigacja, nagłówki strony.
"""

import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

RESULTS_URL = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

# Linie, które NIE są nazwami graczy / teamów
_SKIP = {
    "results", "upcoming matches", "upcoming", "contact", "valhalla cup",
    "vs", "head to head", "form", "stats", "match history", "home",
}

# Score: "3 - 1" / "3-1" / "3–1"  (wyłącznie cyfry i separator)
_SCORE_RE = re.compile(r"^(\d+)\s*[-–]\s*(\d+)$")

# Match ID: "Match " + cokolwiek, lub samo "Match\s+\S+"
_MATCH_RE = re.compile(r"^Match\s+\S", re.IGNORECASE)

# Data: "12 Apr 2024" / "2024-04-12" / "Apr 12" / zawiera miesiąc
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\d{{1,2}}\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}|\d{{4}}-\d{{2}}-\d{{2}}",
    re.IGNORECASE,
)


def _is_score(line: str) -> tuple[int, int] | None:
    m = _SCORE_RE.match(line.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _is_junk(line: str) -> bool:
    return line.lower() in _SKIP or len(line) < 2


def _make_id(raw_match_id: str) -> str:
    return hashlib.md5(raw_match_id.encode()).hexdigest()[:12]


async def scrape_results() -> list[dict[str, Any]]:
    """
    Returns completed matches:
      {match_id, player1, team1, player2, team2,
       score, goals1, goals2, total_goals, date, source, status}
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

            body_text = await page.inner_text("body")
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]

            logger.info(f"Results: {len(lines)} body lines")
            # debug — pierwsze 30 linii żeby zobaczyć strukturę
            for idx, l in enumerate(lines[:30]):
                logger.debug(f"  [{idx:02d}] {l!r}")

            seen_ids: set[str] = set()

            for i, line in enumerate(lines):
                # ── ANCHOR: linia zawierająca "Match " ───────────────────────
                if not _MATCH_RE.match(line):
                    continue

                raw_match_id = line  # np. "Match 1234"

                # ── okno: i-4 .. i+7 (granice bezpieczne) ────────────────────
                lo = max(0, i - 4)
                hi = min(len(lines) - 1, i + 7)
                window = lines[lo : hi + 1]
                # indeks match_id wewnątrz window
                wi = i - lo   # lines[i] == window[wi]

                # ── player1 = 2 linie przed match_id ─────────────────────────
                if wi < 2:
                    continue
                player1 = window[wi - 2]
                team1   = window[wi - 1]

                if _is_junk(player1) or _is_junk(team1):
                    continue

                # ── date = 1 linia po match_id ────────────────────────────────
                if wi + 1 >= len(window):
                    continue
                date = window[wi + 1]

                # jeśli linia po match_id nie wygląda jak data → spróbuj i+2
                if not _DATE_RE.search(date):
                    if wi + 2 < len(window):
                        date = window[wi + 2]
                    if not _DATE_RE.search(date):
                        logger.debug(f"No date near {raw_match_id!r}, skip")
                        continue

                # ── player2 = linia tuż po date ──────────────────────────────
                date_wi = window.index(date, wi + 1)
                if date_wi + 1 >= len(window):
                    continue
                player2 = window[date_wi + 1]
                team2   = window[date_wi + 2] if date_wi + 2 < len(window) else ""

                if _is_junk(player2):
                    continue

                # ── score: szukaj "d-d" w liniach po team2 ───────────────────
                goals1 = goals2 = None
                score_str = ""
                search_from = date_wi + 3
                for offset in range(search_from, min(search_from + 5, len(window))):
                    parsed = _is_score(window[offset])
                    if parsed:
                        goals1, goals2 = parsed
                        score_str = window[offset].strip()
                        break

                if goals1 is None:
                    logger.debug(f"No score for {raw_match_id!r} — skip")
                    continue

                match_id = _make_id(raw_match_id)
                if match_id in seen_ids:
                    continue
                seen_ids.add(match_id)

                matches.append({
                    "match_id":    match_id,
                    "raw_id":      raw_match_id,
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
                logger.info(
                    f"  + {player1} vs {player2} | {score_str} | {date}"
                )

        except Exception as exc:
            logger.error(f"Results scraper error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Results scraper done: {len(matches)} matches")
    return matches

"""
Detail scraper — klika każdy mecz, czeka na modal, scrapuje dane.

KROK 1  — nawiguj, poczekaj na JS
KROK 2  — pobierz inner_text("body"), znajdź pozycje "VS"
KROK 3  — dla każdego VS: wyciągnij player1/player2 z linii
KROK 4  — kliknij element klikalny w pobliżu (locator po tekście gracza)
KROK 5  — czekaj na modal (wait_for_selector)
KROK 6  — pobierz inner_text modalu, parsuj H2H / Form / Stats
KROK 7  — zamknij (Escape)
KROK 8  — delay 1s
"""

import hashlib
import logging
import re
from typing import Any

from playwright.async_api import Page, async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)


# ─────────────────────────── modal parser ───────────────────────────────────

def _parse_modal_text(modal_text: str) -> dict[str, Any]:
    """
    Parse the full inner text of the open modal.
    Returns {h2h, form, stats}.
    """
    lines = [l.strip() for l in modal_text.split("\n") if l.strip()]

    data: dict[str, Any] = {
        "h2h":  {},
        "form": {"player1": [], "player2": []},
        "stats": {"player1": {}, "player2": {}},
    }

    # ── Head to head ─────────────────────────────────────────────────────────
    try:
        h2h_idx = next(
            (i for i, l in enumerate(lines) if "head to head" in l.lower()), None
        )
        if h2h_idx is not None:
            # grab the next ~10 lines for numbers
            h2h_chunk = "\n".join(lines[h2h_idx : h2h_idx + 10])
            wins = re.findall(r"\b(\d+)\b", h2h_chunk)
            if len(wins) >= 2:
                data["h2h"]["wins_player1"] = int(wins[0])
                data["h2h"]["wins_player2"] = int(wins[1])

            avg = re.search(
                r"[Tt]otal\s+average\s+goals\s+per\s+match[:\s]*([\d.]+)",
                h2h_chunk,
            )
            if avg:
                data["h2h"]["avg_goals_per_match"] = float(avg.group(1))

            # also grab a standalone float in the chunk (avg goals)
            floats = re.findall(r"\b(\d+\.\d+)\b", h2h_chunk)
            if floats and "avg_goals_per_match" not in data["h2h"]:
                data["h2h"]["avg_goals_per_match"] = float(floats[0])
    except Exception as exc:
        logger.debug(f"H2H parse: {exc}")

    # ── Form ─────────────────────────────────────────────────────────────────
    try:
        form_idx = next(
            (i for i, l in enumerate(lines) if l.lower() == "form"), None
        )
        if form_idx is not None:
            form_chunk = lines[form_idx + 1 : form_idx + 12]
            wld_lines = [l for l in form_chunk if re.search(r"[WLD]", l.upper())]
            if len(wld_lines) >= 2:
                data["form"]["player1"] = re.findall(r"[WLD]", wld_lines[0].upper())
                data["form"]["player2"] = re.findall(r"[WLD]", wld_lines[1].upper())
            elif len(wld_lines) == 1:
                data["form"]["player1"] = re.findall(r"[WLD]", wld_lines[0].upper())
    except Exception as exc:
        logger.debug(f"Form parse: {exc}")

    # ── Stats (Wins %, Goals for, Goals against) ──────────────────────────────
    stat_keys = {
        "wins %": "wins_pct",
        "goals for": "goals_for",
        "goals against": "goals_against",
    }
    try:
        for i, l in enumerate(lines):
            key = stat_keys.get(l.lower())
            if key is None:
                continue
            # numbers usually on the same line or the next 1-2 lines
            chunk = " ".join(lines[i : i + 3])
            nums = re.findall(r"\b[\d.]+\b", chunk)
            # skip the label's own digits if any
            nums = [n for n in nums if "." in n or int(float(n)) < 200]
            if len(nums) >= 2:
                data["stats"]["player1"][key] = float(nums[0])
                data["stats"]["player2"][key] = float(nums[1])
            elif len(nums) == 1:
                data["stats"]["player1"][key] = float(nums[0])
    except Exception as exc:
        logger.debug(f"Stats parse: {exc}")

    return data


# ─────────────────────────── main scraper ───────────────────────────────────

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
            # KROK 1 — nawigacja
            logger.info(f"Details scraper: navigating to {url}")
            await page.goto(url, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5_000)

            # KROK 2 — pobierz body text, znajdź mecze
            body_text = await page.inner_text("body")
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]
            logger.info(f"Details: {len(lines)} body lines")

            # zbierz (player1, player2) z pozycji VS
            match_pairs: list[tuple[str, str]] = []
            seen: set[str] = set()

            for i, line in enumerate(lines):
                if line != "VS":
                    continue
                if i < 4 or i + 2 >= len(lines):
                    continue

                player1 = lines[i - 4]
                player2 = lines[i + 1]

                if player1 in ("VS", "HEAD TO HEAD", "FORM") or player2 == "VS":
                    continue

                key = f"{player1}|{player2}"
                if key not in seen:
                    seen.add(key)
                    match_pairs.append((player1, player2))

            logger.info(f"Details: {len(match_pairs)} unique match pairs to click")

            # KROK 3 — iteracja po parach
            for player1, player2 in match_pairs:
                match_id = hashlib.md5(
                    f"{player1}_{player2}".encode()
                ).hexdigest()[:12]

                try:
                    # KROK 4 — klik na klikalny element zawierający player1
                    # Szukamy elementu, który zawiera tekst gracza i jest klikalny
                    clickable = page.locator(f"text='{player1}'").first
                    await clickable.click(timeout=5_000)

                except Exception as exc:
                    logger.debug(f"Click failed for {player1}: {exc}")
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(500)
                    continue

                # KROK 5 — czekaj na modal
                try:
                    await page.wait_for_selector(
                        "text=Head to head", timeout=6_000
                    )
                except Exception:
                    logger.debug(f"Modal not found for {player1} vs {player2}, skip")
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(500)
                    continue

                # KROK 6 — scrapuj modal (inner_text całej strony po otwarciu modalu)
                try:
                    modal_text = await page.inner_text("body")
                    modal_data = _parse_modal_text(modal_text)

                    results.append({
                        "match_id": match_id,
                        "player1":  player1,
                        "player2":  player2,
                        **modal_data,
                    })
                    logger.info(f"Details: got data for {player1} vs {player2}")

                except Exception as exc:
                    logger.debug(f"Modal parse error for {player1}: {exc}")

                # KROK 7 — zamknij modal
                await page.keyboard.press("Escape")
                # KROK 8 — delay
                await page.wait_for_timeout(1_000)

        except Exception as exc:
            logger.error(f"Details scraper fatal: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details scraper done: {len(results)} matches enriched")
    return results

"""
Detail scraper — clicks every match row on the upcoming page,
waits for the modal, and extracts Head-to-Head + Form + Stats data.

Rules:
  ✅ uses locator(), click(), wait_for_selector()
  ❌ no network intercept, no JSON API, no inner_text of full page
"""

import asyncio
import logging
import re
from typing import Any

from playwright.async_api import Page, async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)


# ─────────────────────────── helpers ────────────────────────────────────────

def _safe_float(text: str) -> float | None:
    m = re.search(r"[\d.]+", text)
    return float(m.group()) if m else None


def _parse_form(text: str) -> list[str]:
    """Extract W/L/D characters from a form string."""
    return re.findall(r"[WLD]", text.upper())


async def _safe_text(locator, default: str = "") -> str:
    try:
        return (await locator.first.inner_text()).strip()
    except Exception:
        return default


# ─────────────────────────── modal scraper ──────────────────────────────────

async def _scrape_modal(page: Page) -> dict[str, Any]:
    """
    Extract data from the currently open match detail modal.
    Returns a dict with h2h, form, and stats sub-dicts.
    """
    data: dict[str, Any] = {
        "h2h": {},
        "form": {"player1": [], "player2": []},
        "stats": {"player1": {}, "player2": {}},
    }

    # ── Head-to-Head ────────────────────────────────────────────────────────
    try:
        # wins for each player (usually two numbers flanking "H2H" or similar)
        h2h_section = page.locator("text=Head to head").locator("..")
        h2h_text = await h2h_section.inner_text()

        wins = re.findall(r"\b(\d+)\b", h2h_text)
        if len(wins) >= 2:
            data["h2h"]["wins_player1"] = int(wins[0])
            data["h2h"]["wins_player2"] = int(wins[1])

        # Total average goals per match
        avg_match = re.search(
            r"[Tt]otal\s+average\s+goals\s+per\s+match[:\s]*([\d.]+)", h2h_text
        )
        if avg_match:
            data["h2h"]["avg_goals_per_match"] = float(avg_match.group(1))
    except Exception as exc:
        logger.debug(f"H2H parse error: {exc}")

    # ── Form ────────────────────────────────────────────────────────────────
    try:
        form_rows = page.locator("text=Form").locator("..").locator("span, div")
        form_count = await form_rows.count()
        form_texts: list[str] = []
        for j in range(min(form_count, 20)):
            t = await form_rows.nth(j).inner_text()
            if re.search(r"[WLD]", t.upper()):
                form_texts.append(t.strip().upper())

        if len(form_texts) >= 2:
            data["form"]["player1"] = _parse_form(form_texts[0])
            data["form"]["player2"] = _parse_form(form_texts[1])
        elif len(form_texts) == 1:
            data["form"]["player1"] = _parse_form(form_texts[0])
    except Exception as exc:
        logger.debug(f"Form parse error: {exc}")

    # ── Stats ────────────────────────────────────────────────────────────────
    try:
        stats_labels = ["Wins %", "Goals for", "Goals against"]
        for label in stats_labels:
            try:
                stat_row = page.locator(f"text={label}").locator("..")
                stat_text = await stat_row.inner_text()
                numbers = re.findall(r"[\d.]+", stat_text)
                key = label.lower().replace(" ", "_").replace("%", "pct")
                if len(numbers) >= 2:
                    data["stats"]["player1"][key] = float(numbers[0])
                    data["stats"]["player2"][key] = float(numbers[1])
                elif len(numbers) == 1:
                    data["stats"]["player1"][key] = float(numbers[0])
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"Stats parse error: {exc}")

    return data


# ─────────────────────────── public entry point ─────────────────────────────

async def scrape_details(url: str = UPCOMING_URL) -> list[dict[str, Any]]:
    """
    Visit *url*, click each match row that has a VS divider,
    scrape the modal detail, close it, move to next.

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

            # KROK 2 — znajdź mecze
            matches = page.locator("div:has-text('VS')")
            count = await matches.count()
            logger.info(f"Details scraper: found {count} match containers")

            # KROK 3 — iteracja
            for i in range(count):
                container = matches.nth(i)

                try:
                    container_text = await container.inner_text()

                    # skip large wrapper divs
                    if container_text.count("VS") > 2:
                        continue

                    lines = [
                        ln.strip()
                        for ln in container_text.splitlines()
                        if ln.strip()
                    ]
                    vs_idx = next(
                        (j for j, l in enumerate(lines) if l.upper() == "VS"), None
                    )
                    if vs_idx is None or vs_idx == 0 or vs_idx >= len(lines) - 1:
                        continue

                    player1 = lines[vs_idx - 1]
                    player2 = lines[vs_idx + 1]

                    import hashlib
                    match_id = hashlib.md5(
                        f"{player1}_{player2}".encode()
                    ).hexdigest()[:12]

                    # KROK 4 — klik
                    await container.click()

                    # KROK 5 — czekaj na modal
                    try:
                        await page.wait_for_selector(
                            "text=Head to head", timeout=5_000
                        )
                    except Exception:
                        logger.debug(
                            f"Modal didn't open for {player1} vs {player2}, skipping"
                        )
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(500)
                        continue

                    # KROK 6 — scrapuj modal
                    modal_data = await _scrape_modal(page)

                    results.append(
                        {
                            "match_id": match_id,
                            "player1": player1,
                            "player2": player2,
                            **modal_data,
                        }
                    )
                    logger.info(f"Scraped details for {player1} vs {player2}")

                except Exception as exc:
                    logger.debug(f"Error processing match {i}: {exc}")

                finally:
                    # KROK 7 — zamknij modal
                    await page.keyboard.press("Escape")
                    # KROK 8 — delay
                    await page.wait_for_timeout(1_000)

        except Exception as exc:
            logger.error(f"Details scraper fatal error: {exc}")
        finally:
            await browser.close()

    logger.info(f"Details scraper finished: {len(results)} matches with details")
    return results

"""
Entry point — runs the scrape/model loop every 60 seconds
and starts the FastAPI server.

Usage:
  python main.py
  # or
  uvicorn api.app:app --host 0.0.0.0 --port 8000
  # (then start the loop separately: python main.py --loop-only)
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

LOOP_INTERVAL = 60  # seconds


# ─────────────────────────── scrape cycle ───────────────────────────────────

async def run_cycle() -> None:
    """
    One full scrape + model cycle:
      1. scrape results
      2. update database
      3. scrape upcoming
      4. scrape details (modal click)
      5. rebuild player stats
    """
    logger.info("═" * 50)
    logger.info(f"Cycle start: {datetime.utcnow().isoformat()} UTC")

    from services.scraper.results import scrape_results
    from services.scraper.upcoming import scrape_upcoming
    from services.scraper.details import scrape_details, UPCOMING_URL
    from core.database import (
        upsert_matches,
        upsert_match_details,
        rebuild_player_stats,
    )

    # 1 — results
    try:
        logger.info("Step 1/4 — scraping results…")
        results = await scrape_results()
        added = upsert_matches(results)
        logger.info(f"  → {len(results)} results scraped, {added} new")
    except Exception as exc:
        logger.error(f"  Results scraper failed: {exc}")

    # 2 — rebuild stats after new results
    try:
        players = rebuild_player_stats()
        logger.info(f"  → {len(players)} player profiles rebuilt")
    except Exception as exc:
        logger.error(f"  Stats rebuild failed: {exc}")

    # 3 — upcoming
    try:
        logger.info("Step 2/4 — scraping upcoming…")
        upcoming = await scrape_upcoming()
        added = upsert_matches(upcoming)
        logger.info(f"  → {len(upcoming)} upcoming matches, {added} new")
    except Exception as exc:
        logger.error(f"  Upcoming scraper failed: {exc}")

    # 4 — details (modal): prefer upcoming; fall back to results when empty
    try:
        logger.info("Step 3/4 — scraping match details (modal)…")
        from services.scraper.details import RESULTS_URL
        details = await scrape_details(UPCOMING_URL)
        if not details:
            logger.info("  → 0 upcoming cards; trying results page for H2H data…")
            details = await scrape_details(RESULTS_URL)
        upsert_match_details(details)
        logger.info(f"  → {len(details)} matches enriched with modal data")
    except Exception as exc:
        logger.error(f"  Details scraper failed: {exc}")

    # 5 — rebuild again with any new detail-derived stats
    try:
        logger.info("Step 4/4 — final stats rebuild…")
        players = rebuild_player_stats()
        logger.info(f"  → {len(players)} player profiles (final)")
    except Exception as exc:
        logger.error(f"  Final stats rebuild failed: {exc}")

    logger.info("Cycle complete.")


# ─────────────────────────── main loop ──────────────────────────────────────

async def loop() -> None:
    while True:
        try:
            await run_cycle()
        except Exception as exc:
            logger.error(f"Unhandled cycle error: {exc}")
        logger.info(f"Sleeping {LOOP_INTERVAL}s until next cycle…")
        await asyncio.sleep(LOOP_INTERVAL)


# ─────────────────────────── server + loop ──────────────────────────────────

async def main() -> None:
    import uvicorn
    from api.app import app

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    # run server and the scrape loop concurrently
    await asyncio.gather(
        server.serve(),
        loop(),
    )


if __name__ == "__main__":
    loop_only = "--loop-only" in sys.argv

    if loop_only:
        # headless loop without uvicorn (use when uvicorn is started separately)
        asyncio.run(loop())
    else:
        asyncio.run(main())

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
    from services.scraper.bookmaker import scrape_bookmaker_odds
    from services.bookmaker_matcher import match_bookmaker_to_predictions
    from services.predictor_v2 import run_predictions_v2 as run_predictions
    from core.database import (
        load_matches,
        upsert_matches,
        upsert_match_details,
        rebuild_player_stats,
    )
    from core.db_sqlite import (
        init_db,
        sync_matches as _sqlite_sync,
        auto_settle_predictions,
        backfill_match_info,
        save_bookmaker_odds,
    )
    from core.database import load_predictions

    # 0 — ensure SQLite schema is current (idempotent) + backfill player names
    try:
        init_db()
        backfill_match_info(load_predictions())
    except Exception as exc:
        logger.error(f"  DB init failed: {exc}")

    # 1 — results
    try:
        logger.info("Step 1/4 — scraping results…")
        results = await scrape_results()
        added = upsert_matches(results)
        logger.info(f"  → {len(results)} results scraped, {added} new")
        # Sync to SQLite and auto-settle any pending predictions
        _sqlite_sync(results)
        settled = auto_settle_predictions()
        if settled:
            logger.info(f"  → {settled} prediction rows auto-settled")
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
        logger.info("Step 4/5 — final stats rebuild…")
        players = rebuild_player_stats()
        logger.info(f"  → {len(players)} player profiles (final)")
    except Exception as exc:
        logger.error(f"  Final stats rebuild failed: {exc}")

    # 5 — bookmaker odds (shuffle.vip) — fetched BEFORE predictions so the
    # best-bet selector can align real lines with our model output.
    try:
        logger.info("Step 5/6 — scraping bookmaker odds (shuffle.vip)…")
        bm_entries = await scrape_bookmaker_odds()
        if bm_entries:
            upcoming_now = [
                m for m in load_matches() if m.get("source") == "upcoming"
            ]
            aligned = match_bookmaker_to_predictions(bm_entries, upcoming_now)
            written = 0
            for mid, bm in aligned.items():
                totals = bm.get("totals") or {}
                try:
                    n = save_bookmaker_odds(
                        match_id     = mid,
                        totals       = totals,
                        match_winner = bm.get("match_winner") or None,
                    )
                    written += n
                    logger.info(
                        f"  bookmaker save: match_id={mid} "
                        f"({bm.get('player1')} vs {bm.get('player2')}) → "
                        f"{n} rows, lines={sorted(totals.keys())}"
                    )
                except Exception as sexc:
                    logger.warning(f"  save_bookmaker_odds {mid}: {sexc}")
            logger.info(
                f"  → {len(bm_entries)} bookmaker cards, "
                f"{len(aligned)} matched to upcoming, {written} total-line rows stored"
            )
        else:
            logger.info("  → bookmaker scraper returned no entries (non-fatal)")
    except Exception as exc:
        logger.error(f"  Bookmaker scraper failed: {exc}")

    # 6 — predictions
    try:
        logger.info("Step 6/6 — computing predictions…")
        preds = run_predictions()
        logger.info(f"  → {len(preds)} predictions computed")
    except Exception as exc:
        logger.error(f"  Predictions failed: {exc}")

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

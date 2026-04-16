"""
One-off script: wipe the prediction/typy history so statistics restart
from scratch with bookmaker lines & odds.

Run from project root:
    python -m scripts.reset_history

Clears (SQLite):
    - predictions
    - model_performance
    - match_info
    - bookmaker_odds
    - bookmaker_1x2

Also removes the JSON prediction cache used by the UI fallback:
    - data/predictions.json

The raw `matches` table (completed results used for model fitting) and the
scraped JSON match files are left intact on purpose — they are the historical
fuel for the model, not the typy ledger.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reset_history")


def main() -> None:
    from core.db_sqlite import init_db, reset_prediction_history

    init_db()
    reset_prediction_history()

    # Wipe the predictions.json cache served by the /matches endpoint fallback.
    preds_json = Path("data") / "predictions.json"
    if preds_json.exists():
        preds_json.write_text(json.dumps([], ensure_ascii=False))
        log.info("emptied %s", preds_json)

    log.info("done.")


if __name__ == "__main__":
    main()

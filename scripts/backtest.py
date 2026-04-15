#!/usr/bin/env python3
"""
Backtest: evaluate hit rate, ROI, Brier score, and calibration buckets.

Usage:
  cd /opt/FIFA
  python scripts/backtest.py            # default line 6.5
  python scripts/backtest.py --line 5.5
  python scripts/backtest.py --line all

Requires settled predictions (actual_over IS NOT NULL in SQLite).
Run core/db_sqlite.py update_actual_result() as matches finish.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.db_sqlite import get_backtest_rows, init_db


def backtest(line: float) -> None:
    rows = get_backtest_rows(line)
    if not rows:
        print(f"  No settled predictions for O/U {line} — run matches first.")
        return

    n          = len(rows)
    n_hit      = sum(1 for r in rows
                     if r["actual_over"] == (1 if r["prob_calibrated"] >= 0.5 else 0))
    hit_rate   = n_hit / n

    # Brier score (lower = better; perfect = 0)
    brier = sum((r["prob_calibrated"] - r["actual_over"]) ** 2 for r in rows) / n

    # Flat-stake ROI at calibrated odds
    pnl = 0.0
    for r in rows:
        p    = r["prob_calibrated"]
        odds = 1.0 / p if p > 0 else 0.0
        pnl += (odds - 1.0) if r["actual_over"] == 1 else -1.0
    roi = pnl / n

    print(f"\n{'═'*50}")
    print(f"  Backtest: O/U {line}   ({n} settled predictions)")
    print(f"{'═'*50}")
    print(f"  Hit rate   : {hit_rate * 100:.1f}%  ({n_hit}/{n})")
    print(f"  ROI        : {roi * 100:+.1f}%")
    print(f"  Net P&L    : {pnl:+.2f} units")
    print(f"  Brier score: {brier:.4f}  (0=perfect, 0.25=random)")

    # Calibration buckets
    print(f"\n  Calibration (predicted prob vs actual frequency):")
    print(f"  {'Bucket':<12} {'n':>4}  {'pred':>6}  {'actual':>7}  {'error':>7}")
    for lo in [i * 0.1 for i in range(10)]:
        hi     = round(lo + 0.1, 1)
        bucket = [r for r in rows if lo <= r["prob_calibrated"] < hi]
        if not bucket:
            continue
        avg_pred   = sum(r["prob_calibrated"] for r in bucket) / len(bucket)
        avg_actual = sum(r["actual_over"]     for r in bucket) / len(bucket)
        err        = avg_pred - avg_actual
        print(f"  [{lo:.1f}–{hi:.1f})     {len(bucket):>4}  {avg_pred:>6.3f}  "
              f"{avg_actual:>7.3f}  {err:>+7.3f}")

    # Top misses
    misses = sorted(
        [r for r in rows if r["actual_over"] != (1 if r["prob_calibrated"] >= 0.5 else 0)],
        key=lambda r: abs(r["prob_calibrated"] - 0.5), reverse=True
    )[:10]
    if misses:
        print(f"\n  Top {len(misses)} worst misses (highest confidence, wrong):")
        for r in misses:
            direction = "OVER  predicted" if r["prob_calibrated"] >= 0.5 else "UNDER predicted"
            outcome   = "but UNDER" if r["actual_over"] == 0 else "but OVER"
            print(f"    {r['match_id'][:12]}  {direction} ({r['prob_calibrated']:.2f}) {outcome}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Valhalla Cup predictor backtest")
    ap.add_argument(
        "--line", default="6.5",
        help="Line to evaluate (e.g. 5.5, 6.5, 7.5, or 'all')"
    )
    args = ap.parse_args()

    init_db()

    if args.line.lower() == "all":
        for line in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            backtest(line)
    else:
        backtest(float(args.line))


if __name__ == "__main__":
    main()

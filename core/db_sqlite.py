"""
SQLite persistence layer — historical matches, predictions, model performance.

data/valhalla.db
  matches           — completed match records (synced from JSON each cycle)
  predictions       — per-match per-line model outputs with actuals
  model_performance — settled bets for backtesting
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "valhalla.db"


# ── Connection ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS matches (
    id          TEXT    PRIMARY KEY,
    player_a    TEXT    NOT NULL,
    player_b    TEXT    NOT NULL,
    goals_a     INTEGER,
    goals_b     INTEGER,
    total_goals INTEGER,
    played_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_m_pa  ON matches(player_a);
CREATE INDEX IF NOT EXISTS idx_m_pb  ON matches(player_b);
CREATE INDEX IF NOT EXISTS idx_m_at  ON matches(played_at);

CREATE TABLE IF NOT EXISTS match_info (
    match_id   TEXT PRIMARY KEY,
    player1    TEXT NOT NULL,
    player2    TEXT NOT NULL,
    date       TEXT,
    lambda_val REAL,
    tempo      REAL,
    h2h        REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        TEXT    NOT NULL,
    line            REAL    NOT NULL,
    lambda_raw      REAL,
    lambda_capped   REAL,
    lambda_final    REAL,
    tempo           REAL,
    asymmetry       REAL,
    variance_factor REAL,
    h2h_weighted    REAL,
    h2h_source      TEXT,
    prob_raw        REAL,
    prob_calibrated REAL,
    value_edge      REAL,
    created_at      TEXT,
    actual_over     INTEGER,
    actual_goals    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_p_mid  ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_p_line ON predictions(line);
CREATE INDEX IF NOT EXISTS idx_predictions_match_time ON predictions(match_id, created_at);

CREATE TABLE IF NOT EXISTS model_performance (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id       TEXT NOT NULL,
    line           REAL NOT NULL,
    predicted_prob REAL,
    actual_result  INTEGER,
    created_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_mp_mid ON model_performance(match_id);
"""


def init_db() -> None:
    with _conn() as c:
        c.executescript(_DDL)
        # Migration: add columns introduced after initial schema
        for stmt in [
            "ALTER TABLE predictions ADD COLUMN actual_goals INTEGER",
            "ALTER TABLE predictions ADD COLUMN is_best_bet INTEGER DEFAULT 0",
        ]:
            try:
                c.execute(stmt)
            except Exception:
                pass  # column already exists
    logger.debug("SQLite DB ready: %s", DB_PATH)


# ── Sync JSON → SQLite ────────────────────────────────────────────────────────

def sync_matches(matches: list[dict]) -> int:
    """
    Import completed matches from the JSON list into SQLite.
    Idempotent — skips already-imported records.
    Returns number of newly inserted rows.
    """
    completed = [
        m for m in matches
        if m.get("source") == "results"
        and m.get("goals1") is not None
        and m.get("goals2") is not None
    ]
    if not completed:
        return 0

    inserted = 0
    with _conn() as c:
        existing = {r[0] for r in c.execute("SELECT id FROM matches").fetchall()}
        for m in completed:
            mid = m["match_id"]
            if mid in existing:
                continue
            g1, g2 = int(m["goals1"]), int(m["goals2"])
            c.execute(
                "INSERT INTO matches (id,player_a,player_b,goals_a,goals_b,total_goals,played_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (mid,
                 m["player1"].upper(), m["player2"].upper(),
                 g1, g2, g1 + g2,
                 m.get("date")),
            )
            inserted += 1

    if inserted:
        logger.info("db_sqlite: %d new matches synced", inserted)
    return inserted


# ── H2H weighted average ──────────────────────────────────────────────────────

def get_h2h_weighted(player_a: str, player_b: str) -> tuple[float | None, str]:
    """
    Rank-based weighted average total goals for this player pair.

    Weights (most-recent first):
      rank  0–9   → 1.0
      rank 10–29  → 0.7
      rank 30–99  → 0.4
      rank 100+   → 0.2

    Returns (None, 'none') if fewer than 3 matches found.
    """
    pa, pb = player_a.upper(), player_b.upper()
    with _conn() as c:
        rows = c.execute(
            "SELECT total_goals FROM matches"
            " WHERE (player_a=? AND player_b=?) OR (player_a=? AND player_b=?)"
            " ORDER BY played_at DESC",
            (pa, pb, pb, pa),
        ).fetchall()

    if len(rows) < 3:
        return None, "none"

    total_w = total_wg = 0.0
    for i, row in enumerate(rows):
        g = row["total_goals"]
        if g is None:
            continue
        w = 1.0 if i < 10 else 0.7 if i < 30 else 0.4 if i < 100 else 0.2
        total_w  += w
        total_wg += g * w

    if total_w == 0:
        return None, "none"
    return round(total_wg / total_w, 2), "db_weighted"


# ── Player weighted stats ─────────────────────────────────────────────────────

def get_player_weighted_stats(player: str) -> dict | None:
    """
    Rank-based weighted goals scored / conceded (last 50 matches).

    rank  0–9   → 1.0
    rank 10–19  → 0.7
    rank 20–49  → 0.4

    Returns None if no data found.
    """
    p = player.upper()
    with _conn() as c:
        rows = c.execute(
            """SELECT
                 CASE WHEN player_a=? THEN goals_a ELSE goals_b END AS scored,
                 CASE WHEN player_a=? THEN goals_b ELSE goals_a END AS conceded
               FROM matches
               WHERE player_a=? OR player_b=?
               ORDER BY played_at DESC LIMIT 50""",
            (p, p, p, p),
        ).fetchall()

    if not rows:
        return None

    ws = wc = tw = 0.0
    for i, row in enumerate(rows):
        if row["scored"] is None:
            continue
        w   = 1.0 if i < 10 else 0.7 if i < 20 else 0.4
        ws += row["scored"]   * w
        wc += row["conceded"] * w
        tw += w

    if tw == 0:
        return None
    return {
        "avg_scored_w":   round(ws / tw, 3),
        "avg_conceded_w": round(wc / tw, 3),
        "n": len(rows),
    }


# ── Prediction storage ────────────────────────────────────────────────────────

def save_predictions_batch(
    *,
    match_id:        str,
    lambda_raw:      float,
    lambda_final:    float,
    tempo:           float,
    asymmetry:       float,
    h2h_weighted:    float | None,
    h2h_source:      str,
    predictions:     dict,         # {line_str: {p_over_raw, p_over, p_under, ...}}
    best_bet_line:   str | None = None,  # line string of the chosen best bet
) -> int:
    """
    Insert one row per line for this match.

    Always INSERT — never UPDATE or REPLACE.
    Every prediction cycle builds history; count must grow every run.
    The best_bet_line row is marked with is_best_bet=1 for flat-bet tracking.
    Returns number of rows inserted.
    """
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with _conn() as c:
        for line_str, v in predictions.items():
            line = float(line_str)
            is_bb = 1 if best_bet_line is not None and line_str == best_bet_line else 0
            c.execute(
                """INSERT INTO predictions
                   (match_id, line, lambda_raw, lambda_capped, lambda_final,
                    tempo, asymmetry, variance_factor, h2h_weighted, h2h_source,
                    prob_raw, prob_calibrated, value_edge, created_at, is_best_bet)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (match_id, line,
                 lambda_raw, lambda_final, lambda_final,
                 tempo, asymmetry, 1.0,
                 h2h_weighted, h2h_source,
                 v.get("p_over_raw", v.get("p_over")), v["p_over"],
                 None, now, is_bb),
            )
            inserted += 1
            logger.info(
                "Inserted prediction: %s line=%.1f prob_raw=%.3f prob_cal=%.3f%s",
                match_id, line, v.get("p_over_raw", 0), v["p_over"],
                " [BEST BET]" if is_bb else "",
            )
    return inserted


def save_prediction(
    *,
    match_id:        str,
    line:            float,
    lambda_raw:      float,
    lambda_capped:   float,
    lambda_final:    float,
    tempo:           float,
    asymmetry:       float,
    variance_factor: float,
    h2h_weighted:    float | None,
    h2h_source:      str,
    prob_raw:        float,
    prob_calibrated: float,
    value_edge:      float | None = None,
) -> None:
    """Single-line insert. Kept for compatibility; prefer save_predictions_batch."""
    with _conn() as c:
        c.execute(
            """INSERT INTO predictions
               (match_id,line,lambda_raw,lambda_capped,lambda_final,
                tempo,asymmetry,variance_factor,h2h_weighted,h2h_source,
                prob_raw,prob_calibrated,value_edge,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (match_id, line, lambda_raw, lambda_capped, lambda_final,
             tempo, asymmetry, variance_factor, h2h_weighted, h2h_source,
             prob_raw, prob_calibrated, value_edge,
             datetime.now(timezone.utc).isoformat()),
        )
    logger.info(
        "Inserted prediction: %s line=%.1f prob_cal=%.3f",
        match_id, line, prob_calibrated,
    )


def update_actual_result(match_id: str, line: float, actual_over: bool) -> None:
    """Update after a match settles — closes the loop for backtesting."""
    with _conn() as c:
        c.execute(
            "UPDATE predictions SET actual_over=? WHERE match_id=? AND line=?",
            (int(actual_over), match_id, line),
        )
        c.execute(
            "INSERT INTO model_performance (match_id,line,predicted_prob,actual_result,created_at)"
            " SELECT match_id,line,prob_calibrated,?,?"
            " FROM predictions WHERE match_id=? AND line=?",
            (int(actual_over), datetime.now(timezone.utc).isoformat(),
             match_id, line),
        )


# ── Historical goals data (for v2 model fitting) ─────────────────────────────

def get_all_total_goals(limit: int = 500) -> list[int]:
    """
    Return total_goals for the most recent `limit` completed matches,
    ordered newest-first. Used by predictor_v2 to fit regime parameters.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT total_goals FROM matches"
            " WHERE total_goals IS NOT NULL"
            " ORDER BY played_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [r["total_goals"] for r in rows]


# ── Backtest data ─────────────────────────────────────────────────────────────

def get_backtest_rows(line: float) -> list[dict]:
    """Return all settled predictions for a given line."""
    with _conn() as c:
        rows = c.execute(
            """SELECT match_id, prob_raw, prob_calibrated, actual_over,
                      lambda_final, created_at
               FROM predictions
               WHERE line=? AND actual_over IS NOT NULL
               ORDER BY created_at""",
            (line,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Match info ────────────────────────────────────────────────────────────────

def upsert_match_info(
    match_id: str,
    player1:  str,
    player2:  str,
    date:     str | None,
    lambda_val: float | None,
    tempo:    float | None,
    h2h:      float | None,
) -> None:
    """Store human-readable match metadata keyed by match_id."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO match_info
               (match_id, player1, player2, date, lambda_val, tempo, h2h, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (match_id, player1.upper(), player2.upper(), date,
             lambda_val, tempo, h2h, now),
        )


# ── Settle match (manual result entry) ───────────────────────────────────────

def settle_match(match_id: str, actual_goals: int) -> int:
    """
    Record the actual total goals for a match and mark all predictions as settled.
    Inserts into model_performance for backtesting.
    Returns number of prediction rows updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    with _conn() as c:
        rows = c.execute(
            "SELECT id, line, prob_calibrated FROM predictions WHERE match_id=?",
            (match_id,),
        ).fetchall()
        for row in rows:
            over = 1 if actual_goals > row["line"] else 0
            c.execute(
                "UPDATE predictions SET actual_over=?, actual_goals=? WHERE id=?",
                (over, actual_goals, row["id"]),
            )
            c.execute(
                """INSERT INTO model_performance
                   (match_id, line, predicted_prob, actual_result, created_at)
                   VALUES (?,?,?,?,?)""",
                (match_id, row["line"], row["prob_calibrated"], over, now),
            )
            updated += 1
    logger.info("settle_match: %s total_goals=%d → %d rows settled", match_id, actual_goals, updated)
    return updated


def backfill_match_info(predictions: list[dict]) -> int:
    """
    Populate match_info from a list of prediction dicts (e.g. loaded from
    data/predictions.json). Called on startup to fix missing player names
    for predictions stored before match_info table existed.
    Returns number of rows inserted.
    """
    inserted = 0
    with _conn() as c:
        existing = {r[0] for r in c.execute("SELECT match_id FROM match_info").fetchall()}
        for p in predictions:
            mid = p.get("match_id")
            if not mid or mid in existing:
                continue
            c.execute(
                """INSERT OR IGNORE INTO match_info
                   (match_id, player1, player2, date, lambda_val, tempo, h2h, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (mid,
                 (p.get("player1") or "?").upper(),
                 (p.get("player2") or "?").upper(),
                 p.get("date"),
                 p.get("lambda_total") or p.get("lambda_raw"),
                 p.get("tempo_avg"),
                 p.get("h2h_avg_goals"),
                 p.get("created_at")),
            )
            inserted += 1
    if inserted:
        logger.info("backfill_match_info: %d entries added", inserted)
    return inserted


# ── Automatic prediction settlement ──────────────────────────────────────────

def auto_settle_predictions() -> int:
    """
    Automatically match stored predictions against scraped results.

    Logic:
      1. Find all match_ids in `predictions` with no actual_over yet.
      2. Look up player1/player2/date in match_info.
      3. Search the `matches` table for a completed result with the same
         player pair AND date within ±3 hours.
      4. Call settle_match() to fill actual_over/actual_goals.

    Called every scrape cycle after sync_matches() so calibration is
    continuous — no manual input required.
    Returns total prediction rows settled this run.
    """
    settled_total = 0
    with _conn() as c:
        unsettled = c.execute(
            """SELECT DISTINCT p.match_id, mi.player1, mi.player2, mi.date
               FROM predictions p
               JOIN match_info mi ON p.match_id = mi.match_id
               WHERE p.actual_over IS NULL""",
        ).fetchall()

    for row in unsettled:
        p1       = row["player1"].upper()
        p2       = row["player2"].upper()
        pred_date = row["date"]   # "DD/MM/YYYY HH:MM" from upcoming scraper

        with _conn() as c:
            result = c.execute(
                """SELECT total_goals, played_at FROM matches
                   WHERE ((player_a=? AND player_b=?) OR (player_a=? AND player_b=?))
                   AND total_goals IS NOT NULL
                   ORDER BY played_at DESC LIMIT 1""",
                (p1, p2, p2, p1),
            ).fetchone()

        if not result or result["total_goals"] is None:
            continue

        # Date safety check:
        #   - result must be played AFTER (or within 30 min before) the prediction
        #     date — guards against settling a prediction against an old match
        #   - result must be within 3 hours of prediction date — guards against
        #     a future match played days later being matched
        #   - if date parsing fails → SKIP (safer to miss than to mis-settle)
        if pred_date and result["played_at"]:
            try:
                fmt = "%d/%m/%Y %H:%M"
                pd = datetime.strptime(pred_date.strip()[:16], fmt)
                rd = datetime.strptime(result["played_at"].strip()[:16], fmt)
                diff_sec = (rd - pd).total_seconds()
                # rd must be >= pd − 30min (allow slight scheduling variance)
                # AND rd must be <= pd + 3h (not a completely different match)
                if diff_sec < -1800 or diff_sec > 3 * 3600:
                    continue
            except Exception:
                continue  # date unknown → skip (do NOT risk a false settle)

        n = settle_match(row["match_id"], result["total_goals"])
        if n:
            logger.info(
                "auto_settle: %s vs %s → %d goals (%d rows)",
                p1, p2, result["total_goals"], n,
            )
            settled_total += n

    if settled_total:
        logger.info("auto_settle_predictions: %d rows settled this cycle", settled_total)
    return settled_total


# ── Per-line calibration from settled data ────────────────────────────────────

def get_line_calibration(min_samples: int = 15) -> dict[float, float]:
    """
    Compute mean prediction error per line from settled predictions.
    mean_error = mean(prob_calibrated − actual_over)
    Returns {line: offset} for lines with >= min_samples settled rows.
    A positive offset means the model over-estimates → subtract from p_over.
    """
    with _conn() as c:
        rows = c.execute(
            """SELECT line,
                      AVG(prob_calibrated - actual_over) AS mean_error,
                      COUNT(*) AS n
               FROM predictions
               WHERE actual_over IS NOT NULL
               GROUP BY line
               HAVING COUNT(*) >= ?""",
            (min_samples,),
        ).fetchall()
    result = {}
    for r in rows:
        result[float(r["line"])] = round(float(r["mean_error"]), 4)
        logger.info(
            "calibration: line=%.1f  mean_error=%+.4f  n=%d",
            r["line"], r["mean_error"], r["n"],
        )
    return result


# ── Model accuracy statistics ─────────────────────────────────────────────────

_FLAT_STAKE = 100.0  # PLN per best-bet


def get_model_stats() -> dict:
    """
    Compute model accuracy and flat-bet P&L from all settled predictions.

    Returns a dict with:
      by_line      — per-line stats (total, correct, accuracy)
      best_bets    — stats only for is_best_bet=1 rows, with full P&L:
                     bets, wins, losses, profit (PLN), yield_pct, win_rate
                     + by_label breakdown (PEWNY/DOBRY/OK)
      flat_stake   — stake per bet (100 PLN)
      total_settled — total rows with actual_over recorded
    """
    with _conn() as c:
        rows = c.execute(
            """SELECT line, prob_calibrated, actual_over,
                      COALESCE(is_best_bet, 0) AS is_best_bet
               FROM predictions
               WHERE actual_over IS NOT NULL""",
        ).fetchall()

    if not rows:
        return {"by_line": {}, "best_bets": {}, "flat_stake": _FLAT_STAKE, "total_settled": 0}

    # ── Per-line direction accuracy ───────────────────────────────────────
    by_line: dict[float, dict] = {}
    for r in rows:
        line   = float(r["line"])
        prob   = float(r["prob_calibrated"])
        actual = int(r["actual_over"])
        correct = 1 if (1 if prob > 0.5 else 0) == actual else 0

        if line not in by_line:
            by_line[line] = {"total": 0, "correct": 0}
        by_line[line]["total"]   += 1
        by_line[line]["correct"] += correct

    for s in by_line.values():
        s["accuracy"] = round(s["correct"] / s["total"], 3) if s["total"] else 0.0

    # ── Best-bet flat-stake P&L (only is_best_bet=1 rows) ────────────────
    # Label inferred from prob (same thresholds as _best_bet in predictor):
    #   prob ≥ 0.65 or ≤ 0.35  → PEWNY
    #   prob ≥ 0.60 or ≤ 0.40  → DOBRY
    #   prob ≥ 0.55 or ≤ 0.45  → OK
    def _bb_label(prob: float) -> str:
        p = prob if prob >= 0.5 else 1.0 - prob
        if p >= 0.65: return "PEWNY"
        if p >= 0.60: return "DOBRY"
        return "OK"

    bb_totals: dict[str, dict] = {
        "PEWNY": {"bets": 0, "wins": 0, "profit": 0.0},
        "DOBRY": {"bets": 0, "wins": 0, "profit": 0.0},
        "OK":    {"bets": 0, "wins": 0, "profit": 0.0},
    }
    bb_all = {"bets": 0, "wins": 0, "profit": 0.0}

    for r in rows:
        if not r["is_best_bet"]:
            continue
        prob   = float(r["prob_calibrated"])
        actual = int(r["actual_over"])

        # Bet direction: prob > 0.5 → bet OVER, else bet UNDER
        bet_over  = prob >= 0.5
        bet_prob  = prob if bet_over else 1.0 - prob  # confidence ∈ [0.55, 0.70]
        model_odds = 1.0 / bet_prob if bet_prob > 0 else 2.0
        won       = (bet_over and actual == 1) or (not bet_over and actual == 0)

        profit = round(_FLAT_STAKE * (model_odds - 1), 2) if won else -_FLAT_STAKE
        label  = _bb_label(prob)

        bb_totals[label]["bets"]   += 1
        bb_totals[label]["wins"]   += int(won)
        bb_totals[label]["profit"] += profit
        bb_all["bets"]   += 1
        bb_all["wins"]   += int(won)
        bb_all["profit"] += profit

    # Compute derived fields
    for s in bb_totals.values():
        n = s["bets"]
        s["losses"]    = n - s["wins"]
        s["staked"]    = round(n * _FLAT_STAKE, 2)
        s["profit"]    = round(s["profit"], 2)
        s["win_rate"]  = round(s["wins"] / n, 3) if n else 0.0
        s["yield_pct"] = round(s["profit"] / s["staked"] * 100, 2) if s["staked"] else 0.0

    n_all = bb_all["bets"]
    bb_all["losses"]    = n_all - bb_all["wins"]
    bb_all["staked"]    = round(n_all * _FLAT_STAKE, 2)
    bb_all["profit"]    = round(bb_all["profit"], 2)
    bb_all["win_rate"]  = round(bb_all["wins"] / n_all, 3) if n_all else 0.0
    bb_all["yield_pct"] = round(bb_all["profit"] / bb_all["staked"] * 100, 2) if bb_all["staked"] else 0.0

    return {
        "by_line":       {str(k): v for k, v in sorted(by_line.items())},
        "best_bets":     {**bb_all, "by_label": bb_totals},
        "flat_stake":    _FLAT_STAKE,
        "total_settled": len(rows),
    }


# ── Prediction history for UI ─────────────────────────────────────────────────

def get_prediction_history(limit: int = 60) -> list[dict]:
    """
    Return recent predictions grouped by match for the history/settle UI.
    Each entry has: match_id, player1, player2, date, predictions per line,
    and settlement status (actual_goals if settled).
    """
    with _conn() as c:
        # Latest prediction row per (match_id, line) using a subquery
        match_ids = [
            r["match_id"] for r in c.execute(
                """SELECT DISTINCT match_id FROM predictions
                   ORDER BY rowid DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        ]
        if not match_ids:
            return []

        result = []
        for mid in match_ids:
            info = c.execute(
                "SELECT player1, player2, date, lambda_val, tempo, h2h "
                "FROM match_info WHERE match_id=?",
                (mid,),
            ).fetchone()

            lines_rows = c.execute(
                """SELECT p.line, p.prob_calibrated, p.prob_raw,
                          p.actual_over, p.actual_goals, p.created_at
                   FROM predictions p
                   INNER JOIN (
                       SELECT line, MAX(created_at) AS mc
                       FROM predictions WHERE match_id=? GROUP BY line
                   ) latest ON p.line=latest.line AND p.created_at=latest.mc
                             AND p.match_id=?
                   ORDER BY p.line""",
                (mid, mid),
            ).fetchall()

            if not lines_rows:
                continue

            actual_goals = lines_rows[0]["actual_goals"]
            is_settled   = actual_goals is not None
            lines_data   = {
                str(r["line"]): {
                    "p_over":     round(r["prob_calibrated"], 4),
                    "p_over_raw": round(r["prob_raw"] or 0, 4),
                    "actual_over": r["actual_over"],
                }
                for r in lines_rows
            }

            result.append({
                "match_id":    mid,
                "player1":     info["player1"]    if info else "?",
                "player2":     info["player2"]    if info else "?",
                "date":        info["date"]        if info else None,
                "lambda_val":  info["lambda_val"] if info else None,
                "tempo":       info["tempo"]       if info else None,
                "h2h":         info["h2h"]         if info else None,
                "is_settled":  is_settled,
                "actual_goals": actual_goals,
                "predictions": lines_data,
                "created_at":  lines_rows[0]["created_at"],
            })
        return result

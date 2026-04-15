"""
Enhanced Poisson predictor for Valhalla Cup FIFA matches.

Pipeline
────────
1.  Base lambda       — Poisson formula with tier-1/2/3 stats
                        (widget → computed history → global default)
2.  Recent form       — SQLite rank-weighted stats (last 50 matches)
3.  H2H weighted      — scraped avg (priority) OR SQLite rank-weighted fallback
4.  Composition       — 0.5 × base + 0.3 × recent_form + 0.2 × h2h
5.  Style / tempo     — style_factor (1.08/0.92/1.0) × tempo_factor
6.  Asymmetry         — |λ_a − λ_b|
7.  Variance model    — dynamic multiplier from tempo + asymmetry + attack
8.  Lambda soft cap   — compress values above 9.5
9.  Poisson CDF       — raw probabilities per line
10. Sigmoid calibration — 1 / (1 + exp(−4 × (p − 0.5)))
11. Blowout detection  — reduce under confidence by 12% for HIGH_VAR matches
12. Value detection    — edge > 0.05 AND prob in [0.45, 0.70]
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

LINES      = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
_DEFAULT   = 3.5    # goals fallback when no stats available
_CAP_PIVOT = 9.5    # lambda soft-cap pivot
_CAP_SLOPE = 0.3    # compression factor above pivot


# ── Date utility ──────────────────────────────────────────────────────────────

def _is_future(date_str: str | None) -> bool:
    if not date_str:
        return True
    try:
        dt = datetime.strptime(date_str.strip()[:16], "%d/%m/%Y %H:%M")
        return dt.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)
    except Exception:
        return True


# ── Poisson ───────────────────────────────────────────────────────────────────

def _poisson_cdf(lam: float, k_max: int) -> float:
    """P(X ≤ k_max) for X ~ Poisson(lam). Pure Python, no scipy."""
    if lam <= 0:
        return 1.0
    total = log_fact = 0.0
    log_lam = math.log(lam)
    for k in range(k_max + 1):
        if k > 0:
            log_fact += math.log(k)
        total += math.exp(k * log_lam - lam - log_fact)
    return min(total, 1.0)


# ── Lambda cap ────────────────────────────────────────────────────────────────

def _cap(lam: float) -> float:
    """Soft cap: values above 9.5 are compressed by factor 0.3."""
    if lam <= _CAP_PIVOT:
        return lam
    return _CAP_PIVOT + (lam - _CAP_PIVOT) * _CAP_SLOPE


# ── Sigmoid calibration ───────────────────────────────────────────────────────

def _calibrate(p: float) -> float:
    """
    Sigmoid sharpening around 0.5, clamped to [0.01, 0.99].

    Effect on extremes:
      p=0.90 → 0.832   p=0.95 → 0.858
      p=0.10 → 0.168   p=0.05 → 0.142
    """
    c = 1.0 / (1.0 + math.exp(-4.0 * (p - 0.5)))
    return max(0.01, min(0.99, c))


# ── Style classification ──────────────────────────────────────────────────────

def _style(tempo: float) -> str:
    return "over" if tempo > 6.5 else "under" if tempo < 5.0 else "neutral"


def _style_factor(sa: str, sb: str) -> float:
    if sa == "over"  and sb == "over":  return 1.08
    if sa == "under" and sb == "under": return 0.92
    return 1.0


# ── Over/Under table ──────────────────────────────────────────────────────────

def _over_under(lam: float, high_variance: bool = False) -> dict[str, dict]:
    """
    Return calibrated {line: {p_over, p_under, over, under, p_over_raw}}.
    For HIGH_VAR matches, under confidence is reduced by 12%.
    """
    out: dict[str, dict] = {}
    for line in LINES:
        k      = int(line)
        pu_raw = _poisson_cdf(lam, k)
        po_raw = 1.0 - pu_raw
        po_cal = _calibrate(po_raw)
        pu_cal = _calibrate(pu_raw)
        if high_variance:
            pu_cal = max(0.01, pu_cal * 0.88)
        out[str(line)] = {
            "p_over":     round(po_cal, 4),
            "p_under":    round(pu_cal, 4),
            "over":       round(1.0 / po_cal, 2),
            "under":      round(1.0 / pu_cal, 2),
            "p_over_raw": round(po_raw, 4),   # stored for logging & DB
        }
    return out


# ── Value detection ───────────────────────────────────────────────────────────

def _value_bets(predictions: dict, high_variance: bool) -> list[dict]:
    """
    Flag value bets where:
      - edge (prob − implied_prob) > 0.05
      - prob in [0.45, 0.70]
      - skip under bets in HIGH_VAR matches
    """
    results = []
    for line, v in predictions.items():
        for side in ("over", "under"):
            if side == "under" and high_variance:
                continue
            prob = v[f"p_{side}"]
            odds = v[side]
            if odds <= 0 or not (0.45 <= prob <= 0.70):
                continue
            edge = prob - 1.0 / odds
            if edge > 0.05:
                results.append({
                    "line": line,
                    "side": side,
                    "prob": round(prob, 4),
                    "odds": odds,
                    "edge": round(edge, 4),
                })
    return results


# ── Stat resolution ───────────────────────────────────────────────────────────

def _resolve_stats(
    name: str,
    players: dict,
    matches: list,
) -> tuple[float, float, str]:
    """
    Tier-1: oddin.gg widget stats from most-recent enriched match.
    Tier-2: computed averages from players.json.
    Tier-3: global default (3.5 / 3.5).
    """
    name_up = name.upper()
    for m in reversed(matches):
        mstats = m.get("stats", {})
        if not mstats:
            continue
        if   m.get("player1", "").upper() == name_up: blk = mstats.get("player1", {})
        elif m.get("player2", "").upper() == name_up: blk = mstats.get("player2", {})
        else: continue
        gf, ga = blk.get("goals_for"), blk.get("goals_against")
        if gf is not None and ga is not None:
            return float(gf), float(ga), "widget"

    key = next((k for k in players if k.upper() == name_up), None)
    if key:
        s = players[key]
        return (
            float(s.get("avg_goals_scored",   _DEFAULT)),
            float(s.get("avg_goals_conceded", _DEFAULT)),
            "history",
        )
    return _DEFAULT, _DEFAULT, "default"


def _db_stats(name: str) -> dict | None:
    """SQLite rank-weighted recent stats. Returns None on any failure."""
    try:
        from core.db_sqlite import get_player_weighted_stats
        return get_player_weighted_stats(name)
    except Exception:
        return None


# ── H2H resolution ────────────────────────────────────────────────────────────

def _h2h_goals(match: dict, matches: list) -> tuple[float | None, str]:
    """
    1. Scraped avg_goals_per_match from this match or any prior record.
    2. SQLite rank-weighted fallback.
    Returns (None, 'none') if unavailable.
    """
    h = match.get("h2h", {})
    if h.get("avg_goals_per_match"):
        return float(h["avg_goals_per_match"]), "scraped"

    p1 = match["player1"].upper()
    p2 = match["player2"].upper()
    for m in reversed(matches):
        m1, m2 = m.get("player1","").upper(), m.get("player2","").upper()
        if (m1 == p1 and m2 == p2) or (m1 == p2 and m2 == p1):
            h2 = m.get("h2h", {})
            if h2.get("avg_goals_per_match"):
                return float(h2["avg_goals_per_match"]), "scraped"

    try:
        from core.db_sqlite import get_h2h_weighted
        val, src = get_h2h_weighted(p1, p2)
        if val is not None:
            return val, src
    except Exception:
        pass

    return None, "none"


# ── Variance model ────────────────────────────────────────────────────────────

def _variance(
    tempo: float,
    asym: float,
    gf_a: float,
    ga_b: float,
) -> tuple[float, bool]:
    """
    Returns (variance_factor, high_variance_flag).

    variance_factor components:
      tempo > 5.0          → +0.20
      asymmetry > 0.8      → +0.30
      attacker gf_a > 3.0  → +0.20

    high_variance: both high attack (gf_a > 3.0) AND porous defence (ga_b > 3.0)
    """
    vf = 1.0
    if tempo > 5.0: vf += 0.20
    if asym  > 0.8: vf += 0.30
    if gf_a  > 3.0: vf += 0.20
    return round(vf, 3), (gf_a > 3.0 and ga_b > 3.0)


# ── Single-match prediction ───────────────────────────────────────────────────

def _predict_one(
    match: dict,
    players: dict,
    matches: list,
) -> dict[str, Any]:
    p1, p2 = match["player1"], match["player2"]

    # ── Tier-1/2/3 stats ──────────────────────────────────────────────────────
    gf_a, ga_a, src_a = _resolve_stats(p1, players, matches)
    gf_b, ga_b, src_b = _resolve_stats(p2, players, matches)

    # ── SQLite weighted stats (optional enrichment) ───────────────────────────
    db_a = _db_stats(p1)
    db_b = _db_stats(p2)
    egf_a = db_a["avg_scored_w"]   if db_a else gf_a
    ega_a = db_a["avg_conceded_w"] if db_a else ga_a
    egf_b = db_b["avg_scored_w"]   if db_b else gf_b
    ega_b = db_b["avg_conceded_w"] if db_b else ga_b

    # ── Lambda components ─────────────────────────────────────────────────────
    lam_a    = (gf_a  + ga_b)  / 2.0
    lam_b    = (gf_b  + ga_a)  / 2.0
    lam_base = lam_a + lam_b                          # component 1

    lam_rf   = (egf_a + ega_b) / 2.0 + (egf_b + ega_a) / 2.0  # component 2

    h2h_val, h2h_src = _h2h_goals(match, matches)
    lam_h2h  = h2h_val if h2h_val is not None else lam_base    # component 3

    # ── Weighted composition ──────────────────────────────────────────────────
    lam_comp = 0.5 * lam_base + 0.3 * lam_rf + 0.2 * lam_h2h

    # ── Style / tempo ──────────────────────────────────────────────────────────
    tempo_a  = egf_a + ega_a
    tempo_b  = egf_b + ega_b
    tempo    = (tempo_a + tempo_b) / 2.0
    sa, sb   = _style(tempo_a), _style(tempo_b)
    sf       = _style_factor(sa, sb)
    tf       = tempo / 6.0

    # ── Caps & variance ───────────────────────────────────────────────────────
    lam_styled = max(lam_comp * sf * tf, 0.5)   # pre-cap (logged as λ_raw)
    lam_capped = _cap(lam_styled)               # after soft cap

    asym        = round(abs(lam_a - lam_b), 3)
    vf, hi_var  = _variance(tempo, asym, egf_a, ega_b)

    lam_final  = _cap(lam_capped * vf)          # after variance + second cap

    # ── Probabilities ──────────────────────────────────────────────────────────
    preds = _over_under(lam_final, hi_var)

    # ── Persist 6.5 line to SQLite (non-fatal) ────────────────────────────────
    try:
        from core.db_sqlite import save_prediction
        p65 = preds["6.5"]
        save_prediction(
            match_id        = match["match_id"],
            line            = 6.5,
            lambda_raw      = lam_styled,
            lambda_capped   = lam_capped,
            lambda_final    = lam_final,
            tempo           = tempo,
            asymmetry       = asym,
            variance_factor = vf,
            h2h_weighted    = h2h_val,
            h2h_source      = h2h_src,
            prob_raw        = p65["p_over_raw"],
            prob_calibrated = p65["p_over"],
        )
    except Exception as exc:
        logger.debug("SQLite save_prediction skipped: %s", exc)

    return {
        "match_id":         match["match_id"],
        "player1":          p1,
        "player2":          p2,
        "date":             match.get("date"),
        # ── backward-compat fields (HTML template) ────────────────────────────
        "lambda1":          round(lam_a, 3),
        "lambda2":          round(lam_b, 3),
        "lambda_total":     round(lam_final, 3),
        # ── diagnostic fields ─────────────────────────────────────────────────
        "lambda_raw":       round(lam_styled, 3),
        "lambda_capped":    round(lam_capped, 3),
        "base_lambda":      round(lam_base, 3),
        "recent_lambda":    round(lam_rf, 3),
        "h2h_lambda":       round(lam_h2h, 3),
        "tempo_avg":        round(tempo, 2),
        "style_a":          sa,
        "style_b":          sb,
        "style_factor":     sf,
        "tempo_factor":     round(tf, 4),
        "asymmetry":        asym,
        "variance_factor":  vf,
        "high_variance":    hi_var,
        "h2h_avg_goals":    h2h_val,
        "h2h_source":       h2h_src,
        "stat_src_a":       src_a,
        "stat_src_b":       src_b,
        "predictions":      preds,
        "value_bets":       _value_bets(preds, hi_var),
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── Prediction log line ───────────────────────────────────────────────────────

def _log_pred(pred: dict) -> None:
    p65     = pred["predictions"]["6.5"]
    hv_str  = " ⚠ HIGH_VAR" if pred.get("high_variance") else ""
    val_str = ""
    if pred.get("value_bets"):
        val_str = " | ★ VALUE: " + ", ".join(
            f"{v['side'].upper()} {v['line']} @{v['odds']} (edge={v['edge']:.3f})"
            for v in pred["value_bets"]
        )
    logger.info(
        "  %s vs %s"
        " | λ_raw=%.2f → λ_final=%.2f"
        " | h2h=%.1f (%s)"
        " | tempo=%.1f [%sv%s]"
        " | asym=%.2f vf=%.2f%s"
        " | O6.5=%d%%→%d%% [%s/%s]%s",
        pred["player1"], pred["player2"],
        pred["lambda_raw"],  pred["lambda_total"],
        pred["h2h_avg_goals"] or 6.0, pred["h2h_source"],
        pred["tempo_avg"], pred["style_a"], pred["style_b"],
        pred["asymmetry"],  pred["variance_factor"], hv_str,
        round(p65["p_over_raw"] * 100), round(p65["p_over"] * 100),
        pred["stat_src_a"], pred["stat_src_b"],
        val_str,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_predictions() -> list[dict[str, Any]]:
    """
    Compute over/under predictions for every upcoming future match,
    persist to data/predictions.json and SQLite, and return the list.
    """
    from core.database import load_matches, load_players, save_predictions

    matches = load_matches()
    players = load_players()

    # Sync completed matches to SQLite (needed for weighted H2H / recent-form)
    try:
        from core.db_sqlite import init_db, sync_matches
        init_db()
        n = sync_matches(matches)
        if n:
            logger.info("Predictor: %d new matches synced to SQLite", n)
    except Exception as exc:
        logger.warning("SQLite sync skipped (non-fatal): %s", exc)

    upcoming = [
        m for m in matches
        if m.get("source") == "upcoming" and _is_future(m.get("date"))
    ]

    if not upcoming:
        logger.info("Predictor: no upcoming matches, skipping")
        return []

    results: list[dict] = []
    for m in upcoming:
        try:
            pred = _predict_one(m, players, matches)
            results.append(pred)
            _log_pred(pred)
        except Exception as exc:
            logger.warning(
                "Predictor error %s vs %s: %s",
                m.get("player1"), m.get("player2"), exc,
            )

    save_predictions(results)
    logger.info("Predictor: %d predictions written", len(results))
    return results

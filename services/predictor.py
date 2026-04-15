"""
Poisson over/under predictor for Valhalla Cup FIFA matches.

Feature pipeline
────────────────
1.  Player stats  — from oddin.gg widget (goals_for / goals_against stored in
    match["stats"]["player1|2"]) when available; otherwise fall back to our
    computed averages from players.json; otherwise global defaults.

2.  Tempo         — gf + ga per player; classifies each player's style as
    "over" / "under" / "neutral" and applies a style_factor to lambda.

3.  H2H           — avg_goals_per_match from the most recent enriched record
    for this pairing; if missing, factor = 1.0 (no adjustment).

Lambda formula
──────────────
    lambda_A     = (attack_A + defense_B) / 2
    lambda_B     = (attack_B + defense_A) / 2
    lambda_base  = lambda_A + lambda_B
    tempo_factor = tempo_avg / 6.0
    h2h_factor   = 1 + (h2h_goals - 6.0) / 10      [1.0 when no H2H data]
    lambda_total = lambda_base * tempo_factor * h2h_factor * style_factor

Predictions: P(over X.5) = 1 - Poisson_CDF(lambda_total, floor(X.5))
Lines evaluated: 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
_DEFAULT_GOALS = 3.5   # used when a player has no recorded stats at all
_H2H_NEUTRAL   = 6.0   # reference point; H2H == 6.0 → no correction


# ── Poisson maths ─────────────────────────────────────────────────────────────

def _poisson_cdf(lam: float, k_max: int) -> float:
    """P(X ≤ k_max) for X ~ Poisson(lam); pure Python, no scipy required."""
    if lam <= 0:
        return 1.0
    total = 0.0
    log_lam = math.log(lam)
    log_fact = 0.0
    for k in range(k_max + 1):
        if k > 0:
            log_fact += math.log(k)
        total += math.exp(k * log_lam - lam - log_fact)
    return min(total, 1.0)


def _over_under(lam: float) -> dict[str, dict[str, float]]:
    """Return {line: {p_over, p_under, over_odds, under_odds}} for every line."""
    out: dict[str, dict[str, float]] = {}
    for line in LINES:
        k = int(line)          # e.g. 3 for 3.5
        pu = _poisson_cdf(lam, k)
        po = max(1.0 - pu, 0.0001)
        pu = max(pu, 0.0001)
        out[str(line)] = {
            "p_over":  round(po, 4),
            "p_under": round(pu, 4),
            "over":    round(1.0 / po, 2),   # decimal odds
            "under":   round(1.0 / pu, 2),
        }
    return out


# ── Style classification ──────────────────────────────────────────────────────

def _style(tempo: float) -> str:
    if tempo > 6.5:
        return "over"
    if tempo < 5.0:
        return "under"
    return "neutral"


def _style_factor(sa: str, sb: str) -> float:
    if sa == "over"  and sb == "over":  return 1.15
    if sa == "under" and sb == "under": return 0.85
    return 1.0


# ── Stat resolution ───────────────────────────────────────────────────────────

def _resolve_stats(
    name: str,
    players: dict[str, dict],
    matches: list[dict],
) -> tuple[float, float, str]:
    """
    Return (gf_avg, ga_avg, source_label) using three-tier fallback:

      1. oddin.gg widget stats stored in the most recent enriched match record
         (match["stats"]["player1|2"]["goals_for|against"]).
         These cover the last 2 months across all opponents — most reliable.

      2. Our own computed per-player averages from players.json
         (avg_goals_scored, avg_goals_conceded).

      3. Global default (3.5 / 3.5).
    """
    name_up = name.upper()

    # Tier 1 — widget stats (most recent match that has them)
    for m in reversed(matches):
        p1_up = m.get("player1", "").upper()
        p2_up = m.get("player2", "").upper()
        mstats = m.get("stats", {})
        if not mstats:
            continue
        if p1_up == name_up:
            blk = mstats.get("player1", {})
        elif p2_up == name_up:
            blk = mstats.get("player2", {})
        else:
            continue
        gf = blk.get("goals_for")
        ga = blk.get("goals_against")
        if gf is not None and ga is not None:
            return float(gf), float(ga), "widget"

    # Tier 2 — computed historical averages (case-insensitive key lookup)
    key = next((k for k in players if k.upper() == name_up), None)
    if key:
        s = players[key]
        return (
            float(s.get("avg_goals_scored",   _DEFAULT_GOALS)),
            float(s.get("avg_goals_conceded", _DEFAULT_GOALS)),
            "history",
        )

    # Tier 3 — defaults
    return _DEFAULT_GOALS, _DEFAULT_GOALS, "default"


def _find_h2h_goals(match: dict, matches: list[dict]) -> float | None:
    """
    Return avg_goals_per_match for the player pair from:
      1. The match itself (if already enriched by the details scraper).
      2. The most recent other match record for the same pair (either direction).
    """
    h = match.get("h2h", {})
    if h.get("avg_goals_per_match"):
        return float(h["avg_goals_per_match"])

    p1 = match["player1"].upper()
    p2 = match["player2"].upper()
    for m in reversed(matches):
        m1 = m.get("player1", "").upper()
        m2 = m.get("player2", "").upper()
        if (m1 == p1 and m2 == p2) or (m1 == p2 and m2 == p1):
            h2 = m.get("h2h", {})
            if h2.get("avg_goals_per_match"):
                return float(h2["avg_goals_per_match"])
    return None


# ── Single-match prediction ───────────────────────────────────────────────────

def _predict_one(
    match: dict,
    players: dict[str, dict],
    matches: list[dict],
) -> dict[str, Any]:
    p1, p2 = match["player1"], match["player2"]

    gf_a, ga_a, src_a = _resolve_stats(p1, players, matches)
    gf_b, ga_b, src_b = _resolve_stats(p2, players, matches)

    tempo_a   = gf_a + ga_a
    tempo_b   = gf_b + ga_b
    tempo_avg = (tempo_a + tempo_b) / 2.0

    sa = _style(tempo_a)
    sb = _style(tempo_b)
    sf = _style_factor(sa, sb)

    h2h = _find_h2h_goals(match, matches)

    lam_a    = (gf_a + ga_b) / 2.0
    lam_b    = (gf_b + ga_a) / 2.0
    lam_base = lam_a + lam_b

    tempo_f  = tempo_avg / 6.0
    h2h_f    = 1.0 + (h2h - _H2H_NEUTRAL) / 10.0 if h2h is not None else 1.0
    lam_total = max(lam_base * tempo_f * h2h_f * sf, 0.5)

    return {
        "match_id":      match["match_id"],
        "player1":       p1,
        "player2":       p2,
        "date":          match.get("date"),
        # lambda1/lambda2 kept for backward compat with existing HTML template
        "lambda1":       round(lam_a, 3),
        "lambda2":       round(lam_b, 3),
        "lambda_total":  round(lam_total, 3),
        "tempo_avg":     round(tempo_avg, 2),
        "style_a":       sa,
        "style_b":       sb,
        "style_factor":  sf,
        "h2h_avg_goals": h2h,
        "h2h_factor":    round(h2h_f, 4),
        "tempo_factor":  round(tempo_f, 4),
        "stat_src_a":    src_a,
        "stat_src_b":    src_b,
        "predictions":   _over_under(lam_total),
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_predictions() -> list[dict[str, Any]]:
    """
    Compute over/under predictions for every upcoming match, persist to
    data/predictions.json, and return the list.
    """
    from core.database import load_matches, load_players, save_predictions

    matches = load_matches()
    players = load_players()
    upcoming = [m for m in matches if m.get("source") == "upcoming"]

    if not upcoming:
        logger.info("Predictor: no upcoming matches, skipping")
        return []

    results: list[dict] = []
    for m in upcoming:
        try:
            pred = _predict_one(m, players, matches)
            results.append(pred)
            p = pred["predictions"]
            logger.info(
                f"  {pred['player1']} vs {pred['player2']}"
                f" | λ={pred['lambda_total']:.2f}"
                f" | tempo={pred['tempo_avg']:.1f}"
                f" [{pred['style_a']}v{pred['style_b']}]"
                f" | h2h={pred['h2h_avg_goals']}"
                f" | O5.5={p['5.5']['p_over']*100:.0f}%"
                f" O6.5={p['6.5']['p_over']*100:.0f}%"
                f" [src:{pred['stat_src_a']}/{pred['stat_src_b']}]"
            )
        except Exception as exc:
            logger.warning(
                f"  Predictor error {m.get('player1')} vs {m.get('player2')}: {exc}"
            )

    save_predictions(results)
    logger.info(f"Predictor: {len(results)} predictions written")
    return results

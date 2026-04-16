"""
services/predictor_v2.py

Mixture Negative Binomial predictor for Valhalla Cup FIFA matches.

Pipeline
────────
1.  Data analysis      — query SQLite matches, compute global mean/variance/k_global
2.  Two-regime fit     — low cluster (total_goals ≤ 6) and high cluster (> 6)
                         NegBin(μ, k) per cluster via method of moments
3.  Dynamic weight     — w = f(base_rate, tempo, h2h_avg, asymmetry)
                         w = weight on the high-chaos (high-goal) regime
4.  Mixture CDF        — P(over L.5) = 1 − [(1−w)·CDF_low(L) + w·CDF_high(L)]
5.  Sigmoid calibration — steepness=2.5, clamped [0.05, 0.95]
6.  Tempo adjustment   — high_tempo (>6.5): p_under × 0.88
7.  Value detection    — edge > 0.05 AND prob in [0.42, 0.78]
8.  SQLite persist     — save_predictions_batch (always INSERT, builds time-series)

Stat resolution (three-tier, mirrors predictor.py):
  Tier-1: oddin.gg widget stats from most-recent enriched match
  Tier-2: computed averages from players.json
  Tier-3: global default (3.5 goals each direction)
  SQLite:  rank-weighted recent stats (last 50 matches, optional enrichment)

Note on the NegBin parameterisation used throughout:
  X ~ NegBin(μ, k)
  Mean     = μ
  Variance = μ + μ²/k
  As k → ∞ the distribution converges to Poisson(μ).
  k is fitted by method of moments: k̂ = μ̂² / (s² − μ̂)
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

LINES    = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
_DEFAULT = 3.5   # fallback goals when no stats available
_SPLIT   = 6     # total_goals ≤ _SPLIT → low regime, > _SPLIT → high regime


# ── Negative Binomial math (pure Python — no scipy) ──────────────────────────

def _negbin_pmf(x: int, mu: float, k: float) -> float:
    """
    P(X = x) for NegBin(μ, k).

    log P = log Γ(x+k) − log Γ(k) − log Γ(x+1)
            + k·log(k/(k+μ)) + x·log(μ/(k+μ))
    """
    if mu <= 0 or k <= 0 or x < 0:
        return 1.0 if x == 0 else 0.0
    try:
        log_coeff = (
            math.lgamma(x + k)
            - math.lgamma(k)
            - math.lgamma(x + 1)
        )
        p_k  = k  / (k + mu)
        p_mu = mu / (k + mu)
        log_prob = log_coeff + k * math.log(p_k) + x * math.log(p_mu)
        return math.exp(log_prob)
    except (ValueError, OverflowError):
        return 0.0


def _negbin_cdf(x_max: int, mu: float, k: float) -> float:
    """P(X ≤ x_max) — accumulated PMF sum."""
    total = 0.0
    for x in range(x_max + 1):
        pmf = _negbin_pmf(x, mu, k)
        total += pmf
        # early exit when tail is negligible
        if pmf < 1e-9 and x > max(x_max, int(mu) + 5):
            break
    return min(total, 1.0)


# ── Regime parameter container ────────────────────────────────────────────────

class RegimeParams:
    """Fitted NegBin parameters for one scoring regime."""

    __slots__ = ("mu", "k", "n", "mean", "variance")

    def __init__(
        self,
        mu:       float,
        k:        float,
        n:        int,
        mean:     float,
        variance: float,
    ) -> None:
        self.mu       = mu
        self.k        = k
        self.n        = n
        self.mean     = mean
        self.variance = variance

    def __repr__(self) -> str:
        return f"NegBin(μ={self.mu:.2f}, k={self.k:.2f}, n={self.n})"


def _fit_negbin(goals: list[int]) -> RegimeParams:
    """
    Fit NegBin(μ, k) by method of moments from a list of observed totals.

    μ̂ = sample mean
    k̂ = μ̂² / (s² − μ̂)   [if s² > μ̂]
       = 200              [if s² ≤ μ̂, distribution is not overdispersed → near-Poisson]

    k is clamped to [0.5, 200].
    """
    n = len(goals)
    if n < 2:
        return RegimeParams(mu=_DEFAULT, k=5.0, n=n, mean=_DEFAULT, variance=1.0)

    mean     = sum(goals) / n
    variance = sum((g - mean) ** 2 for g in goals) / max(n - 1, 1)

    if variance <= mean or variance <= 0:
        k = 200.0   # essentially Poisson
    else:
        k = mean ** 2 / (variance - mean)

    k  = max(0.5, min(k, 200.0))
    mu = max(0.5, mean)

    return RegimeParams(mu=mu, k=k, n=n, mean=mean, variance=variance)


# ── Data analysis + regime fitting ────────────────────────────────────────────

def analyze_and_fit() -> tuple[RegimeParams, RegimeParams, float, dict]:
    """
    Pull historical total_goals from SQLite, split into two regimes,
    and return fitted NegBin parameters plus meta-diagnostics.

    Returns
    -------
    low_params       : RegimeParams  — NegBin fit for low-scoring games (≤ _SPLIT)
    high_params      : RegimeParams  — NegBin fit for high-scoring games (> _SPLIT)
    base_chaos_w     : float         — fraction of historical games in high regime
    meta             : dict          — diagnostic data (logged, not used for prediction)
    """
    all_goals: list[int] = []
    try:
        from core.db_sqlite import get_all_total_goals
        all_goals = get_all_total_goals(limit=500)
    except Exception as exc:
        logger.warning("v2: SQLite query failed, using hardcoded fallback: %s", exc)

    if len(all_goals) < 10:
        logger.warning(
            "v2: insufficient historical data (%d matches) — using prior defaults",
            len(all_goals),
        )
        low  = RegimeParams(mu=4.5, k=3.0,  n=0, mean=4.5, variance=3.5)
        high = RegimeParams(mu=8.5, k=2.0,  n=0, mean=8.5, variance=9.0)
        return low, high, 0.35, {"data_source": "fallback", "n_total": 0}

    low_goals  = [g for g in all_goals if g <= _SPLIT]
    high_goals = [g for g in all_goals if g >  _SPLIT]

    low  = _fit_negbin(low_goals)
    high = (
        _fit_negbin(high_goals)
        if len(high_goals) >= 3
        else RegimeParams(mu=8.5, k=2.0, n=0, mean=8.5, variance=9.0)
    )

    n_total  = len(all_goals)
    base_w   = len(high_goals) / n_total if n_total > 0 else 0.35

    # Global overdispersion check (diagnostic only)
    g_mean   = sum(all_goals) / n_total
    g_var    = sum((g - g_mean) ** 2 for g in all_goals) / max(n_total - 1, 1)
    k_global = (
        round(g_mean ** 2 / (g_var - g_mean), 3)
        if g_var > g_mean else None
    )

    meta = {
        "data_source": "sqlite",
        "n_total":     n_total,
        "n_low":       len(low_goals),
        "n_high":      len(high_goals),
        "global_mean": round(g_mean,  3),
        "global_var":  round(g_var,   3),
        "k_global":    k_global,
        "low":         repr(low),
        "high":        repr(high),
        "base_w":      round(base_w, 3),
    }

    logger.info(
        "v2: fit — %s  |  %s  |  base_w=%.2f  k_global=%s  n=%d",
        low, high, base_w, k_global, n_total,
    )
    return low, high, base_w, meta


# ── Dynamic mixture weight ────────────────────────────────────────────────────

def _dynamic_weight(
    base_w:    float,
    tempo:     float,
    h2h_avg:   float | None,
    asymmetry: float,
) -> float:
    """
    Adjust the base high-regime weight by match context signals.

    Boosts (more chaos expected):
      tempo > 7.0          → +0.12
      6.5 < tempo ≤ 7.0   → +0.06
      h2h_avg > 7.0        → +0.15  (pair historically high-scoring)
      6.5 < h2h_avg ≤ 7.0 → +0.08

    Dampeners (more predictable, lower totals):
      asymmetry > 2.0      → −0.08  (one player dominates)
      1.5 < asym ≤ 2.0    → −0.04

    Result clamped to [0.10, 0.90].
    """
    w = base_w

    if tempo > 7.0:
        w += 0.12
    elif tempo > 6.5:
        w += 0.06

    if h2h_avg is not None:
        if h2h_avg > 7.0:
            w += 0.15
        elif h2h_avg > 6.5:
            w += 0.08

    if asymmetry > 2.0:
        w -= 0.08
    elif asymmetry > 1.5:
        w -= 0.04

    return max(0.10, min(0.90, w))


# ── Mixture CDF ───────────────────────────────────────────────────────────────

def _mixture_cdf(
    x_max: int,
    low:   RegimeParams,
    high:  RegimeParams,
    w:     float,
) -> float:
    """
    P(X ≤ x_max) = (1−w)·CDF_low(x_max) + w·CDF_high(x_max)

    w is the weight on the high-scoring (chaos) regime.
    """
    cdf_low  = _negbin_cdf(x_max, low.mu,  low.k)
    cdf_high = _negbin_cdf(x_max, high.mu, high.k)
    return min((1.0 - w) * cdf_low + w * cdf_high, 1.0)


# ── Sigmoid calibration ───────────────────────────────────────────────────────

def _calibrate(p: float, steepness: float = 2.5) -> float:
    """Sigmoid calibration, clamped to [0.05, 0.95]."""
    c = 1.0 / (1.0 + math.exp(-steepness * (p - 0.5)))
    return max(0.05, min(0.95, c))


# ── Over/Under table (mixture NegBin) ────────────────────────────────────────

def _over_under_v2(
    low:        RegimeParams,
    high:       RegimeParams,
    w:          float,
    high_tempo: bool = False,
) -> dict[str, dict]:
    """
    Compute calibrated over/under probabilities for every line.

    For line L.5 (e.g. 6.5):
      k_floor = int(L.5) = 6
      P(under) = P(X ≤ 6) = mixture CDF at 6
      P(over)  = 1 − P(under)

    High-tempo adjustment: p_under × 0.88 (under bets less reliable).
    """
    out: dict[str, dict] = {}
    for line in LINES:
        k_floor = int(line)        # e.g. 6.5 → 6
        pu_raw  = _mixture_cdf(k_floor, low, high, w)
        po_raw  = 1.0 - pu_raw

        po_cal  = _calibrate(po_raw)
        pu_cal  = _calibrate(pu_raw)

        if high_tempo:
            pu_cal = max(0.05, pu_cal * 0.88)
            po_cal = max(0.05, min(0.95, 1.0 - pu_cal))

        out[str(line)] = {
            "p_over":     round(po_cal, 4),
            "p_under":    round(pu_cal, 4),
            "over":       round(1.0 / po_cal, 2),
            "under":      round(1.0 / pu_cal, 2),
            "p_over_raw": round(po_raw, 4),
        }
    return out


# ── Value detection ───────────────────────────────────────────────────────────

def _value_bets(predictions: dict) -> list[dict]:
    """
    Flag value bets where edge > 0.05 AND prob in [0.42, 0.78].
    Wider range than v1 to capture more mixture-model opportunities.
    """
    results = []
    for line, v in predictions.items():
        for side in ("over", "under"):
            prob = v[f"p_{side}"]
            odds = v[side]
            if odds <= 0 or not (0.42 <= prob <= 0.78):
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


# ── Stat resolution (three-tier) ──────────────────────────────────────────────

def _resolve_stats(
    name:    str,
    players: dict,
    matches: list,
) -> tuple[float, float, str]:
    """
    Tier-1: oddin.gg widget stats (goals_for / goals_against) from most-recent
            enriched match record.
    Tier-2: computed averages from players.json.
    Tier-3: global default (_DEFAULT / _DEFAULT).
    """
    name_up = name.upper()
    for m in reversed(matches):
        mstats = m.get("stats", {})
        if not mstats:
            continue
        if   m.get("player1", "").upper() == name_up:
            blk = mstats.get("player1", {})
        elif m.get("player2", "").upper() == name_up:
            blk = mstats.get("player2", {})
        else:
            continue
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


def _h2h_goals(match: dict, matches: list) -> tuple[float | None, str]:
    """
    1. Scraped avg_goals_per_match from this match's h2h block.
    2. Same field from any prior match between these two players.
    3. SQLite rank-weighted fallback.
    Returns (None, 'none') if nothing found.
    """
    h = match.get("h2h", {})
    if h.get("avg_goals_per_match"):
        return float(h["avg_goals_per_match"]), "scraped"

    p1 = match["player1"].upper()
    p2 = match["player2"].upper()
    for m in reversed(matches):
        m1 = m.get("player1", "").upper()
        m2 = m.get("player2", "").upper()
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


# ── Style helper ──────────────────────────────────────────────────────────────

def _style(tempo: float) -> str:
    return "over" if tempo > 6.5 else "under" if tempo < 5.0 else "neutral"


# ── Date filter ───────────────────────────────────────────────────────────────

def _is_future(date_str: str | None) -> bool:
    if not date_str:
        return True
    try:
        dt = datetime.strptime(date_str.strip()[:16], "%d/%m/%Y %H:%M")
        return dt.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)
    except Exception:
        return True


# ── Single-match prediction ───────────────────────────────────────────────────

def _predict_one_v2(
    match:   dict,
    players: dict,
    matches: list,
    low:     RegimeParams,
    high:    RegimeParams,
    base_w:  float,
) -> dict[str, Any]:
    p1, p2 = match["player1"], match["player2"]

    # ── Tier-1/2/3 stats ──────────────────────────────────────────────────
    gf_a, ga_a, src_a = _resolve_stats(p1, players, matches)
    gf_b, ga_b, src_b = _resolve_stats(p2, players, matches)

    # ── SQLite weighted stats (optional enrichment) ───────────────────────
    db_a  = _db_stats(p1)
    db_b  = _db_stats(p2)
    egf_a = db_a["avg_scored_w"]   if db_a else gf_a
    ega_a = db_a["avg_conceded_w"] if db_a else ga_a
    egf_b = db_b["avg_scored_w"]   if db_b else gf_b
    ega_b = db_b["avg_conceded_w"] if db_b else ga_b

    # ── Context signal: expected total goals ──────────────────────────────
    # Used to drive the dynamic weight; not used directly as a distribution mean.
    lam_a    = (gf_a  + ga_b)  / 2.0
    lam_b    = (gf_b  + ga_a)  / 2.0
    lam_base = lam_a + lam_b

    lam_rf   = (egf_a + ega_b) / 2.0 + (egf_b + ega_a) / 2.0

    h2h_val, h2h_src = _h2h_goals(match, matches)
    lam_h2h  = h2h_val if h2h_val is not None else lam_base

    lam_ctx  = max(0.5 * lam_base + 0.3 * lam_rf + 0.2 * lam_h2h, 0.5)

    # ── Tempo / asymmetry ─────────────────────────────────────────────────
    tempo_a    = egf_a + ega_a
    tempo_b    = egf_b + ega_b
    tempo      = (tempo_a + tempo_b) / 2.0
    high_tempo = tempo > 6.5
    asym       = round(abs(lam_a - lam_b), 3)
    sa, sb     = _style(tempo_a), _style(tempo_b)

    # ── Dynamic mixture weight ────────────────────────────────────────────
    w = _dynamic_weight(base_w, tempo, h2h_val, asym)

    # ── Mixture probabilities ─────────────────────────────────────────────
    preds = _over_under_v2(low, high, w, high_tempo)

    # ── Diagnostic: P(X ≥ 9) — log when extreme ──────────────────────────
    p_extreme = 1.0 - _mixture_cdf(8, low, high, w)
    if p_extreme > 0.10:
        logger.debug(
            "v2: %s vs %s — P(≥9 goals)=%.1f%%  w=%.2f  tempo=%.2f  h2h=%s",
            p1, p2, p_extreme * 100, w, tempo,
            f"{h2h_val:.1f}" if h2h_val else "—",
        )

    # ── Persist all lines to SQLite ───────────────────────────────────────
    try:
        from core.db_sqlite import save_predictions_batch
        n = save_predictions_batch(
            match_id     = match["match_id"],
            lambda_raw   = lam_ctx,
            lambda_final = lam_ctx,
            tempo        = tempo,
            asymmetry    = asym,
            h2h_weighted = h2h_val,
            h2h_source   = h2h_src,
            predictions  = preds,
        )
        logger.info("v2 SQLite: %d rows inserted for %s", n, match["match_id"])
    except Exception as exc:
        logger.warning("v2 SQLite save failed for %s: %s", match["match_id"], exc)

    return {
        "match_id":      match["match_id"],
        "player1":       p1,
        "player2":       p2,
        "date":          match.get("date"),
        # ── model identity ────────────────────────────────────────────────
        "model":         "mixture_negbin_v2",
        # ── regime parameters (for inspection / backtest) ─────────────────
        "low_mu":        round(low.mu,  3),
        "low_k":         round(low.k,   3),
        "high_mu":       round(high.mu, 3),
        "high_k":        round(high.k,  3),
        "mixture_w":     round(w, 3),
        # ── backward-compat fields expected by the HTML template ──────────
        "lambda1":       round(lam_a,   3),
        "lambda2":       round(lam_b,   3),
        "lambda_total":  round(lam_ctx, 3),
        # ── diagnostic context ────────────────────────────────────────────
        "lambda_raw":    round(lam_ctx, 3),
        "base_lambda":   round(lam_base, 3),
        "recent_lambda": round(lam_rf,   3),
        "h2h_lambda":    round(lam_h2h,  3),
        "tempo_avg":     round(tempo, 2),
        "style_a":       sa,
        "style_b":       sb,
        "high_tempo":    high_tempo,
        "asymmetry":     asym,
        "h2h_avg_goals": h2h_val,
        "h2h_source":    h2h_src,
        "stat_src_a":    src_a,
        "stat_src_b":    src_b,
        "p_extreme":     round(p_extreme, 4),
        # ── main outputs ──────────────────────────────────────────────────
        "predictions":   preds,
        "value_bets":    _value_bets(preds),
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }


# ── Prediction log line ───────────────────────────────────────────────────────

def _log_pred(pred: dict) -> None:
    p65    = pred["predictions"]["6.5"]
    ht_str = " HIGH_TEMPO" if pred.get("high_tempo") else ""
    val_str = ""
    if pred.get("value_bets"):
        val_str = " | ★ VALUE: " + ", ".join(
            f"{v['side'].upper()} {v['line']} @{v['odds']} (edge={v['edge']:.3f})"
            for v in pred["value_bets"]
        )
    logger.info("  MATCH: %s vs %s  [MNB v2]", pred["player1"], pred["player2"])
    logger.info(
        "    μ_low=%.2f(k=%.1f)  μ_high=%.2f(k=%.1f)  w=%.2f  P(≥9)=%.1f%%",
        pred["low_mu"], pred["low_k"],
        pred["high_mu"], pred["high_k"],
        pred["mixture_w"], pred.get("p_extreme", 0) * 100,
    )
    logger.info(
        "    tempo=%.1f [%sv%s]%s  asym=%.2f  h2h=%s (%s)  [src:%s/%s]",
        pred["tempo_avg"], pred["style_a"], pred["style_b"], ht_str,
        pred["asymmetry"],
        f"{pred['h2h_avg_goals']:.1f}" if pred["h2h_avg_goals"] else "—",
        pred["h2h_source"],
        pred["stat_src_a"], pred["stat_src_b"],
    )
    logger.info(
        "    p_raw=%.3f  p_final=%.3f"
        "  O5.5=%d%%  O6.5=%d%%  O7.5=%d%%%s",
        p65["p_over_raw"], p65["p_over"],
        round(pred["predictions"]["5.5"]["p_over"] * 100),
        round(p65["p_over"] * 100),
        round(pred["predictions"]["7.5"]["p_over"] * 100),
        val_str,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_predictions_v2() -> list[dict[str, Any]]:
    """
    Mixture Negative Binomial predictions for every upcoming future match.

    Steps:
      1. Sync completed matches to SQLite.
      2. Fit two-regime NegBin from historical data.
      3. For each upcoming match: compute dynamic w, mixture probs, value bets.
      4. Persist to data/predictions.json and SQLite (always INSERT).
      5. Return list of prediction dicts.
    """
    from core.database import load_matches, load_players, save_predictions

    matches = load_matches()
    players = load_players()

    # Sync completed matches to SQLite (needed for regime fitting + H2H)
    try:
        from core.db_sqlite import init_db, sync_matches
        init_db()
        n = sync_matches(matches)
        if n:
            logger.info("v2: %d new matches synced to SQLite", n)
    except Exception as exc:
        logger.warning("v2: SQLite sync skipped (non-fatal): %s", exc)

    # Fit regime models from historical totals
    low, high, base_w, _meta = analyze_and_fit()

    upcoming = [
        m for m in matches
        if m.get("source") == "upcoming" and _is_future(m.get("date"))
    ]

    if not upcoming:
        logger.info("v2: no upcoming future matches — skipping")
        return []

    results: list[dict] = []
    for m in upcoming:
        try:
            pred = _predict_one_v2(m, players, matches, low, high, base_w)
            results.append(pred)
            _log_pred(pred)
        except Exception as exc:
            logger.warning(
                "v2 error %s vs %s: %s",
                m.get("player1"), m.get("player2"), exc,
            )

    save_predictions(results)
    logger.info("v2: %d predictions written", len(results))
    return results

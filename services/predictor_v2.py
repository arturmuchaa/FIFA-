"""
services/predictor_v2.py

Mixture Negative Binomial predictor for Valhalla Cup FIFA matches.

Pipeline
────────
1.  Data analysis      — query SQLite matches, compute global mean/variance/k_global
2.  Two-regime fit     — low cluster (total_goals ≤ 7) and high cluster (> 7)
                         NegBin(μ, k) per cluster via method of moments
                         Split=7 chosen: more balanced regimes (≈47/53)
3.  Dynamic weight     — w = f(base_rate, tempo, h2h_avg, asymmetry)
                         w = weight on the high-chaos (high-goal) regime
4.  Mixture CDF        — P(over L.5) = 1 − [(1−w)·CDF_low(L) + w·CDF_high(L)]
5.  Clamp calibration  — clamp to [0.05, 0.95] only; no sigmoid squeeze
                         Raw NegBin mixture probabilities are already principled
6.  Value detection    — edge > 0.08 → VALUE; edge > 0.12 → STRONG_VALUE
7.  SQLite persist     — save_predictions_batch (always INSERT, builds time-series)

Removed:
  - Sigmoid calibration (was squeezing P(>=7) down by ~6%, causing under-bias)
  - p_under × 0.88 hack (replaced by proper dynamic weight)

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

LINES    = [round(2.5 + 0.25 * i, 2) for i in range(53)]  # 2.5 → 15.5 every 0.25
_DEFAULT = 3.5   # fallback goals when no stats available
_SPLIT   = 7     # total_goals ≤ _SPLIT → low regime, > _SPLIT → high regime
                 # Split=7 chosen: analysis shows 47/53 balance, better P(>=8) fit


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
        k = 200.0
    else:
        k = mean ** 2 / (variance - mean)

    k  = max(0.5, min(k, 200.0))
    mu = max(0.5, mean)

    return RegimeParams(mu=mu, k=k, n=n, mean=mean, variance=variance)


def _fit_negbin_weighted(goals_weights: list[tuple[int, float]]) -> RegimeParams:
    """
    Fit NegBin(μ, k) by WEIGHTED method of moments.

    Time-decay: recent matches carry weight 1.0, older ones less.
    This makes regime parameters respond to current form rather than
    being anchored to matches from months ago.
    """
    n = len(goals_weights)
    if n < 2:
        return RegimeParams(mu=_DEFAULT, k=5.0, n=n, mean=_DEFAULT, variance=1.0)

    total_w = sum(w for _, w in goals_weights)
    if total_w == 0:
        return RegimeParams(mu=_DEFAULT, k=5.0, n=n, mean=_DEFAULT, variance=1.0)

    mean = sum(g * w for g, w in goals_weights) / total_w
    # Reliability-weighted variance
    sum_w2   = sum(w * w for _, w in goals_weights)
    denom    = total_w - sum_w2 / total_w  # Bessel-equivalent for weighted data
    variance = (
        sum(w * (g - mean) ** 2 for g, w in goals_weights) / denom
        if denom > 0 else 0.0
    )

    if variance <= mean or variance <= 0:
        k = 200.0
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
            "v2: insufficient historical data (%d matches) — using FIFA esports priors",
            len(all_goals),
        )
        low  = RegimeParams(mu=6.0, k=200.0, n=0, mean=6.0, variance=6.0)
        high = RegimeParams(mu=9.5, k=200.0, n=0, mean=9.5, variance=9.5)
        return low, high, 0.47, {"data_source": "fallback_fifa_prior", "n_total": 0}

    # ── Time-decay weights (rank 0 = most recent match) ───────────────────
    # Matches ordered newest-first by get_all_total_goals().
    # Recent form matters more: if a player switched to aggressive play
    # last month, old defensive stats shouldn't drag the estimate down.
    def _rank_w(i: int) -> float:
        if i < 50:  return 1.00
        if i < 150: return 0.60
        if i < 300: return 0.30
        return 0.15

    goals_w     = [(g, _rank_w(i)) for i, g in enumerate(all_goals)]
    low_gw      = [(g, w) for g, w in goals_w if g <= _SPLIT]
    high_gw     = [(g, w) for g, w in goals_w if g >  _SPLIT]

    low  = _fit_negbin_weighted(low_gw)
    high = (
        _fit_negbin_weighted(high_gw)
        if len(high_gw) >= 3
        else RegimeParams(mu=8.5, k=2.0, n=0, mean=8.5, variance=9.0)
    )

    n_total = len(all_goals)
    total_w = sum(w for _, w in goals_w)
    high_w  = sum(w for g, w in goals_w if g > _SPLIT)
    base_w_long = high_w / total_w if total_w > 0 else 0.35

    # ── Short-horizon base_w (recent 80 matches) ──────────────────────────
    # The historical high-regime share is computed over 500 matches and
    # moves slowly. If the current tournament skews higher than the
    # long-term mean (e.g. 7.3 vs 6.5), the model would keep under-weighting
    # the high regime. Mix a short window (80 recent games, band 60-100)
    # at 60% so the predictor adapts to current tournament dynamics.
    RECENT_WINDOW = 80
    recent = all_goals[:RECENT_WINDOW]
    if len(recent) >= 20:
        high_cnt_r = sum(1 for g in recent if g > _SPLIT)
        base_w_recent = high_cnt_r / len(recent)
        base_w = 0.6 * base_w_recent + 0.4 * base_w_long
    else:
        base_w_recent = base_w_long
        base_w = base_w_long

    # Global overdispersion check (diagnostic only)
    g_mean   = sum(all_goals) / n_total
    g_var    = sum((g - g_mean) ** 2 for g in all_goals) / max(n_total - 1, 1)
    k_global = (
        round(g_mean ** 2 / (g_var - g_mean), 3)
        if g_var > g_mean else None
    )

    meta = {
        "data_source":   "sqlite",
        "n_total":       n_total,
        "n_low":         len(low_gw),
        "n_high":        len(high_gw),
        "global_mean":   round(g_mean,  3),
        "global_var":    round(g_var,   3),
        "k_global":      k_global,
        "low":           repr(low),
        "high":          repr(high),
        "base_w":        round(base_w, 3),
        "base_w_long":   round(base_w_long, 3),
        "base_w_recent": round(base_w_recent, 3),
        "recent_n":      len(recent),
    }

    logger.info(
        "v2: fit — %s  |  %s  |  base_w=%.2f (long=%.2f recent=%.2f)  k_global=%s  n=%d",
        low, high, base_w, base_w_long, base_w_recent, k_global, n_total,
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

    Boosts (more high-scoring expected):
      tempo > 7.5          → +0.20  (both players very aggressive)
      7.0 < tempo ≤ 7.5   → +0.13
      6.5 < tempo ≤ 7.0   → +0.06
      h2h_avg > 8.0        → +0.18  (pair consistently high-scoring)
      7.0 < h2h_avg ≤ 8.0 → +0.12
      6.5 < h2h_avg ≤ 7.0 → +0.06

    Dampeners (more contained, lower totals):
      asymmetry > 2.5      → −0.12  (one player strongly dominant)
      2.0 < asym ≤ 2.5    → −0.08
      1.5 < asym ≤ 2.0    → −0.04

    Result clamped to [0.10, 0.92].

    Design rationale (validated against synthetic FIFA distribution, mean≈7.6):
      With split=7 and base_w≈0.47:
        avg match (tempo=6.5):        w≈0.47  → P(over6.5)≈0.63
        high tempo (tempo=7.5):       w≈0.60  → P(over6.5)≈0.69
        aggressive (tempo=7.8,h2h=8): w≈0.79  → P(over6.5)≈0.77
    """
    w = base_w

    # Tempo tiers
    if tempo > 7.5:
        w += 0.20
    elif tempo > 7.0:
        w += 0.13
    elif tempo > 6.5:
        w += 0.06

    # H2H tiers
    if h2h_avg is not None:
        if h2h_avg > 8.0:
            w += 0.18
        elif h2h_avg > 7.0:
            w += 0.12
        elif h2h_avg > 6.5:
            w += 0.06

    # Asymmetry dampener
    if asymmetry > 2.5:
        w -= 0.12
    elif asymmetry > 2.0:
        w -= 0.08
    elif asymmetry > 1.5:
        w -= 0.04

    return max(0.10, min(0.92, w))


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


# ── Calibration ───────────────────────────────────────────────────────────────

def _calibrate(p: float) -> float:
    """
    Clamp-only calibration. No sigmoid squeeze.

    The raw mixture NegBin probabilities are already principled — applying a
    sigmoid toward 0.5 introduces a systematic under-bias: P(>=7) was being
    suppressed by ~6 percentage points. We preserve the raw CDF output and
    only guard against exact 0 or 1.

    When enough actual_over data accumulates in SQLite, replace with
    isotonic regression: fit on (prob_raw, actual_over) pairs per line.
    """
    return max(0.05, min(0.95, p))


# ── Over/Under table (mixture NegBin) ────────────────────────────────────────

def _over_under_v2(
    low: RegimeParams,
    high: RegimeParams,
    w:   float,
) -> dict[str, dict]:
    """
    Compute over/under probabilities for every line via mixture NegBin CDF.

    Uses the standard Asian-bookmaker convention so quarter/half/integer
    lines all price distinctly:

      k.0  (integer, push on X=k):   P(over) = P(X>=k+1) / (1 − P(X=k))
      k.5  (half, no push):          P(over) = P(X>=k+1) = 1 − CDF(k)
      k.25 (split 0.0 / 0.5):        P(over) = 0.5·p(k.0) + 0.5·p(k.5)
      k.75 (split 0.5 / 1.0):        P(over) = 0.5·p(k.5) + 0.5·p((k+1).0)

    Probabilities are clamped to [0.05, 0.95]. High-tempo signal is already
    encoded in w (via _dynamic_weight).
    """
    cdf_cache: dict[int, float] = {}

    def cdf(k: int) -> float:
        if k < 0:
            return 0.0
        if k not in cdf_cache:
            cdf_cache[k] = _mixture_cdf(k, low, high, w)
        return cdf_cache[k]

    def p_over_half(k: int) -> float:
        """P(X >= k+1) — used for half lines like k+0.5."""
        return max(0.0, min(1.0, 1.0 - cdf(k)))

    def p_over_push(k: int) -> float:
        """Push-adjusted P(over k.0) = P(X>=k+1) / (1 - P(X=k))."""
        pmf_k = max(0.0, cdf(k) - cdf(k - 1))
        denom = 1.0 - pmf_k
        if denom <= 1e-9:
            return p_over_half(k)
        return max(0.0, min(1.0, (1.0 - cdf(k)) / denom))

    out: dict[str, dict] = {}
    for line in LINES:
        k = int(line)
        frac = round(line - k, 2)
        if frac == 0.0:
            po_raw = p_over_push(k)
        elif frac == 0.25:
            po_raw = 0.5 * p_over_push(k) + 0.5 * p_over_half(k)
        elif frac == 0.5:
            po_raw = p_over_half(k)
        elif frac == 0.75:
            po_raw = 0.5 * p_over_half(k) + 0.5 * p_over_push(k + 1)
        else:
            po_raw = p_over_half(k)

        pu_raw  = 1.0 - po_raw
        po_cal  = _calibrate(po_raw)
        pu_cal  = _calibrate(pu_raw)

        out[str(line)] = {
            "p_over":     round(po_cal, 4),
            "p_under":    round(pu_cal, 4),
            "over":       round(1.0 / po_cal, 2),
            "under":      round(1.0 / pu_cal, 2),
            "p_over_raw": round(po_raw, 4),
        }
    return out


# ── Best bet selection ────────────────────────────────────────────────────────

def _label_for_prob(prob: float) -> tuple[str, str]:
    if prob >= 0.65:
        return "PEWNY", "#34d399"
    if prob >= 0.60:
        return "DOBRY", "#60a5fa"
    return "OK", "#94a3b8"


def _best_bets(
    predictions: dict,
    book_odds:   dict[str, dict[str, float]] | None,
    lam_ctx:     float,
) -> tuple[dict | None, dict | None]:
    """
    Select TWO independent bets for a match:
      * value_bet — every +EV wager (wide coverage, higher volume)
      * safe_bet  — narrow "pewniak" criteria (high-confidence only)

    Both are evaluated strictly against bookmaker odds. When no bookmaker
    market is available for this match, both picks return None — we never
    place a bet without a real posted price.

    VALUE (old EV-based best bet):
      * candidate side must have a bookmaker price
      * prob ∈ [0.52, 0.90]
      * EV = prob · odds − 1 > 0.02

    SAFE (pewniak):
      * prob ≥ 0.65
      * EV ≥ 0.08
      * |line − λ_ctx| ≤ 1.5   (line close to expected total)
      * book_odds ∈ [1.45, 2.20]   (narrow spread → less risk from bad pricing)

    Returns (safe_bet, value_bet). Either can be None.
    """
    if not book_odds:
        return None, None  # No-bookmaker = no-bet.

    def _candidates():
        for line_str, v in predictions.items():
            b = book_odds.get(str(line_str)) or book_odds.get(str(float(line_str)))
            if not b:
                continue
            try:
                line_f = float(line_str)
            except (TypeError, ValueError):
                continue
            for side, key, odd_key in (
                ("over",  "p_over",  "over"),
                ("under", "p_under", "under"),
            ):
                prob = v.get(key)
                odd  = b.get(odd_key)
                if prob is None or odd is None or odd <= 1.0:
                    continue
                yield line_str, line_f, v, side, float(prob), float(odd)

    def _make(line_str, side, prob, odd, source):
        label, color = _label_for_prob(prob)
        return {
            "line":            line_str,
            "side":            side,
            "side_pl":         "OVER" if side == "over" else "UNDER",
            "prob":            round(prob, 4),
            "model_odds":      round(1.0 / prob, 2) if prob > 0 else None,
            "bookmaker_odds":  round(odd, 2),
            "edge":            round(prob * odd - 1.0, 4),
            "label":           label,
            "color":           color,
            "source":          source,
        }

    # ── VALUE: highest-EV pick in [0.52, 0.90] with EV > 0.02 ─────────────
    value: dict | None = None
    best_ev = -1.0
    for line_str, _line_f, _v, side, prob, odd in _candidates():
        if not (0.52 <= prob <= 0.90):
            continue
        ev = prob * odd - 1.0
        if ev > best_ev:
            best_ev = ev
            value = _make(line_str, side, prob, odd, "value")
    if value is None or best_ev <= 0.02:
        value = None

    # ── SAFE: strictest pewniak — prob ≥ 0.70, EV ≥ 0.12,
    #          line in ±1.0 of λ_ctx, odds in [1.70, 2.20],
    #          skip lines with known |mean_error| > 0.08.
    # Previous thresholds (prob 0.65, EV 0.08, odds 1.45+, dist 1.5) produced
    # a 50% hit rate on a 1.45–2.20 band — mathematically losing. New band
    # requires break-even hit rate ~60% which matches realistic performance
    # for strongly-supported picks.
    try:
        from core.db_sqlite import get_line_calibration
        line_errors = get_line_calibration(min_samples=15)
    except Exception:
        line_errors = {}

    safe: dict | None = None
    safe_ev = -1.0
    for line_str, line_f, _v, side, prob, odd in _candidates():
        if prob < 0.70:
            continue
        if not (1.70 <= odd <= 2.20):
            continue
        if abs(line_f - lam_ctx) > 1.0:
            continue
        ev = prob * odd - 1.0
        if ev < 0.12:
            continue
        line_err = line_errors.get(line_f)
        if line_err is not None and abs(line_err) > 0.08:
            continue  # known-miscalibrated line — skip SAFE
        if ev > safe_ev:
            safe_ev = ev
            safe = _make(line_str, side, prob, odd, "safe")

    return safe, value


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

    # ── Adaptive h2h weighting ────────────────────────────────────────────
    # For consistently high-scoring matchups (h2h > 8.0), the h2h average
    # is the strongest single predictor — increase its weight so lam_ctx
    # reflects the true expected total rather than being pulled toward the
    # global average by stats from different opponents.
    if h2h_val is not None and h2h_val > 8.0:
        lam_ctx = max(0.30 * lam_base + 0.20 * lam_rf + 0.50 * lam_h2h, 0.5)
    elif h2h_val is not None and h2h_val > 7.0:
        lam_ctx = max(0.40 * lam_base + 0.25 * lam_rf + 0.35 * lam_h2h, 0.5)
    else:
        lam_ctx = max(0.5 * lam_base + 0.3 * lam_rf + 0.2 * lam_h2h, 0.5)

    # ── Warm-start λ multiplier (decays once we have settled data) ───────
    # Historical audit showed every settled line had negative mean_error
    # (model under-estimates over). Until get_global_calibration has enough
    # data to correct this bias post-hoc, pre-bias λ upward by up to +6%.
    # The boost linearly fades to zero as settled count approaches 20,
    # after which global_offset takes over — no double correction.
    try:
        from core.db_sqlite import get_settled_count
        n_settled = get_settled_count()
    except Exception:
        n_settled = 0
    warm = max(0.0, (20 - n_settled) / 20.0)
    lam_mult = 1.0 + 0.06 * warm
    if lam_mult > 1.001:
        lam_ctx *= lam_mult

    # ── Tempo / asymmetry ─────────────────────────────────────────────────
    tempo_a = egf_a + ega_a
    tempo_b = egf_b + ega_b
    tempo   = (tempo_a + tempo_b) / 2.0
    asym    = round(abs(lam_a - lam_b), 3)
    sa, sb  = _style(tempo_a), _style(tempo_b)
    # high_tempo kept for output labelling; no longer adjusts probabilities
    high_tempo = tempo > 6.5

    # ── Dynamic mixture weight ────────────────────────────────────────────
    # Tempo signal is fully encoded in w — no separate p_under adjustment.
    w = _dynamic_weight(base_w, tempo, h2h_val, asym)

    # ── Extreme-match adaptation ──────────────────────────────────────────
    # Problem: model uses fixed historical regime means (e.g. mu_high ≈ 8.0)
    # even when a specific matchup consistently scores 9-10 goals (h2h=9.3).
    # Fix: for confirmed "overvover" pairs with strong h2h, anchor the
    # high-regime distribution center to the h2h average (not historical mean)
    # and allow a higher weight cap so the output matches the h2h prior.
    #
    # Condition: both styles "over" (tempo_x > 6.5) + h2h > 8.0 + tempo > 7.0
    # Validated: Frenkie vs Kevin (h2h=9.3, tempo=7.35):
    #   Before fix: O8.5 = 35%  (market: 58%, error -23pp)
    #   After fix:  O8.5 ≈ 55%  (error -3pp)
    is_extreme = (
        sa == "over"
        and sb == "over"
        and h2h_val is not None and h2h_val > 8.0
        and tempo > 7.0
    )
    if is_extreme:
        mu_high_eff = h2h_val          # anchor to H2H mean, not historical fit
        w = min(0.97, w + 0.15)        # raise cap; these pairs are "in" the high bucket
    else:
        mu_high_eff = high.mu

    # ── Lambda-anchored regime scaling ────────────────────────────────────
    # BUG FIX: Two matches with different λ (e.g. 4.6 vs 6.0) but neutral
    # contextual signals (tempo < 6.5, h2h < 6.5) got IDENTICAL probabilities
    # because the model used fixed mu_low/mu_high for all matches, varying only
    # the mixture weight w. When w is the same (no signals fired), every match
    # in the cycle collapses to the same CDF.
    #
    # Fix: scale both regime means proportionally by (lam_ctx / global_mean).
    # This anchors the mixture center to lam_ctx while preserving the
    # mu_low/mu_high ratio (overdispersion structure unchanged).
    #
    # Extreme matches (is_extreme=True): the high regime is already anchored
    # to h2h_val — skip scaling to avoid double-counting.
    global_mean = (1.0 - base_w) * low.mu + base_w * high.mu

    if is_extreme:
        # h2h_val already anchors the high regime correctly
        low_for_match  = low
        high_for_match = RegimeParams(
            mu=mu_high_eff, k=high.k, n=high.n,
            mean=mu_high_eff, variance=high.variance,
        )
    else:
        # Proportional scaling so mixture mean ≈ lam_ctx
        scale          = lam_ctx / global_mean if global_mean > 0.5 else 1.0
        mu_ls          = max(0.5, low.mu  * scale)
        mu_hs          = max(0.5, high.mu * scale)
        low_for_match  = RegimeParams(mu=mu_ls, k=low.k,  n=low.n,  mean=mu_ls,  variance=low.variance)
        high_for_match = RegimeParams(mu=mu_hs, k=high.k, n=high.n, mean=mu_hs, variance=high.variance)

    # ── Mixture probabilities (no sigmoid squeeze, no hack) ───────────────
    preds = _over_under_v2(low_for_match, high_for_match, w)

    # ── Calibration: per-line → band → global fallback chain ──────────
    # Plan II.B/C: per-line threshold lowered to 25 (from 50) because we
    # now have enough settled bets for 13+ lines. Band-level offsets
    # (low <5, mid 5-7, high ≥7) fill the gap for under-sampled lines
    # and break the bimodal cancellation that kept global_offset ≈ 0.
    try:
        from core.db_sqlite import (
            get_line_calibration,
            get_band_calibration,
            get_global_calibration,
        )
        cal_offsets   = get_line_calibration(min_samples=25)
        band_offsets  = get_band_calibration(min_samples=15)
        global_offset = get_global_calibration(min_samples=10)

        def _band_for(line_f: float) -> str:
            if line_f < 5.0:
                return "low"
            if line_f < 7.0:
                return "mid"
            return "high"

        for line_str, v in preds.items():
            line_f = float(line_str)
            offset = cal_offsets.get(line_f)
            if offset is None:
                offset = band_offsets.get(_band_for(line_f))
            if offset is None:
                offset = global_offset
            if offset is not None:
                offset = max(-0.15, min(0.15, offset))  # safety cap
            if offset is not None and abs(offset) > 0.001:
                po = max(0.05, min(0.95, v["p_over"] - offset))
                pu = max(0.05, min(0.95, 1.0 - po))
                v["p_over"]  = round(po, 4)
                v["p_under"] = round(pu, 4)
                v["over"]    = round(1.0 / po,  2)
                v["under"]   = round(1.0 / pu,  2)
    except Exception as exc:
        logger.debug("v2: calibration step skipped: %s", exc)

    # ── Enforce monotonic p_over across lines ─────────────────────────────
    # Per-line calibration above can leave gaps where a higher line has a
    # higher p_over than its lower neighbour (e.g. 6.5 calibrated but 6.75
    # not). That's nonsensical — Over probability must weakly decrease as
    # the line rises. Walk in ascending line order and clamp each entry
    # to <= the previous one, then recompute the dependent fields.
    try:
        ordered = sorted(preds.keys(), key=float)
        prev_po: float | None = None
        for ls in ordered:
            v = preds[ls]
            po = v.get("p_over")
            if po is None:
                continue
            if prev_po is not None and po > prev_po:
                po = prev_po
                pu = max(0.05, min(0.95, 1.0 - po))
                v["p_over"]  = round(po, 4)
                v["p_under"] = round(pu, 4)
                v["over"]    = round(1.0 / po, 2)
                v["under"]   = round(1.0 / pu, 2)
            prev_po = po
    except Exception:
        pass

    # ── Diagnostic: P(X ≥ 9) — always compute, log when notable ──────────
    p_extreme = 1.0 - _mixture_cdf(8, low_for_match, high_for_match, w)
    if p_extreme > 0.10:
        logger.debug(
            "v2: %s vs %s — P(≥9 goals)=%.1f%%  w=%.2f  tempo=%.2f  h2h=%s",
            p1, p2, p_extreme * 100, w, tempo,
            f"{h2h_val:.1f}" if h2h_val else "—",
        )

    # ── Bookmaker odds (optional) ────────────────────────────────────────
    # When the bookmaker scraper has populated odds for this match, restrict
    # best-bet selection to lines the bookmaker actually offers and pick the
    # highest-EV wager at the real price. Silent fallback to legacy selection
    # when nothing is stored.
    book_odds: dict[str, dict[str, float]] = {}
    book_1x2:  dict[str, float] | None     = None
    try:
        from core.db_sqlite import get_bookmaker_odds, get_bookmaker_1x2
        book_odds = get_bookmaker_odds(match["match_id"]) or {}
        book_1x2  = get_bookmaker_1x2(match["match_id"])
    except Exception as exc:
        logger.warning("book odds fetch failed for %s: %s", match["match_id"], exc)

    logger.info(
        "v2 book-odds fetch: match_id=%s → %d lines (keys=%s) 1x2=%s",
        match["match_id"], len(book_odds), sorted(book_odds.keys()), book_1x2,
    )

    # Sanity check: if the scraped book market is obviously the wrong one
    # (e.g. half-time totals bleeding into our match_id) reject it so the
    # UI shows "brak bukm." instead of confidently-priced nonsense.
    #
    # Two independent guards — fail either and the book data is dropped:
    #   1. Median bookmaker line must sit within ±2.0 of λ_total. Half-time
    #      markets usually centre around λ/2, so for any match with λ ≥ 6
    #      the half-time median (~3) is > 2.0 away from λ.
    #   2. Highest captured line must be ≥ λ − 2 — if the entire ladder is
    #      well below the expected total, those are first-half / handicap
    #      lines mislabeled as full-match.
    if book_odds:
        try:
            bk_lines  = sorted(float(k) for k in book_odds.keys())
            bk_median = bk_lines[len(bk_lines) // 2]
            bk_max    = bk_lines[-1]
            lam_v     = float(lam_ctx)
            reason    = None
            if abs(bk_median - lam_v) > 2.0:
                reason = f"median Δ {abs(bk_median - lam_v):.2f} > 2.0"
            elif bk_max < lam_v - 2.0:
                reason = f"max line {bk_max:.2f} < λ−2 ({lam_v - 2.0:.2f})"
            if reason:
                logger.warning(
                    "v2 book-odds rejected: %s λ=%.2f lines=%s — %s",
                    match["match_id"], lam_v,
                    [round(x, 2) for x in bk_lines], reason,
                )
                book_odds = {}
                # Purge the bad rows from SQLite too, otherwise every cycle
                # re-fetches the same stale lines and re-rejects them. The
                # scraper will re-populate on the next successful capture.
                try:
                    from core.db_sqlite import _conn
                    with _conn() as c:
                        c.execute(
                            "DELETE FROM bookmaker_odds WHERE match_id=?",
                            (match["match_id"],),
                        )
                except Exception as exc:
                    logger.debug("book-odds purge failed: %s", exc)
        except Exception as exc:
            logger.debug("book-odds sanity check skipped: %s", exc)

    # Annotate each line with its bookmaker price so the UI can render both
    # our implied odds and the real market odds side by side.
    matched = 0
    for line_str, v in preds.items():
        b = book_odds.get(str(line_str)) or book_odds.get(str(float(line_str)))
        if b:
            v["book_over"]  = round(float(b["over"]),  2)
            v["book_under"] = round(float(b["under"]), 2)
            matched += 1
        else:
            v["book_over"]  = None
            v["book_under"] = None
    if book_odds:
        logger.info(
            "v2 book-odds annotate: %s → %d/%d pred lines matched (pred keys=%s)",
            match["match_id"], matched, len(preds), sorted(preds.keys()),
        )

        # User request: show ONLY the lines the bookmaker actually posts for
        # this match. Predictions for lines without a real book price get
        # dropped from the persisted output and from the UI grid.
        book_keys = set(book_odds.keys()) | {str(float(k)) for k in book_odds.keys()}
        preds = {
            k: v for k, v in preds.items()
            if k in book_keys or str(float(k)) in book_keys
        }
    else:
        # No bookmaker data — trim the full ladder (2.5 → 15.5) to a window
        # centred on λ_total so the UI isn't flooded with 50 irrelevant rows.
        try:
            lam_center = float(lam_ctx)
            preds = {
                k: v for k, v in preds.items()
                if abs(float(k) - lam_center) <= 2.5
            }
        except Exception:
            pass

    # ── Compute dual-mode bets before saving ────────────────────────────
    # Both picks are evaluated independently against the same calibrated
    # probabilities. The VALUE pick is our everyday wager (wider criteria);
    # the SAFE pick only fires when the wager is a genuine "pewniak". Labels
    # are attached from the post-calibration probabilities so PEWNY/DOBRY/OK
    # reflects the same number shown in the UI (bug F).
    safe_bet_info, value_bet_info = _best_bets(
        preds, book_odds=book_odds or None, lam_ctx=float(lam_ctx),
    )
    value_line = value_bet_info["line"] if value_bet_info else None
    value_side = value_bet_info["side"] if value_bet_info else None
    value_odds = value_bet_info.get("bookmaker_odds") if value_bet_info else None
    safe_line  = safe_bet_info["line"]  if safe_bet_info  else None
    safe_side  = safe_bet_info["side"]  if safe_bet_info  else None
    safe_odds  = safe_bet_info.get("bookmaker_odds") if safe_bet_info else None

    # ── Persist all lines to SQLite ───────────────────────────────────────
    try:
        from core.db_sqlite import save_predictions_batch, upsert_match_info
        upsert_match_info(
            match_id   = match["match_id"],
            player1    = p1,
            player2    = p2,
            date       = match.get("date"),
            lambda_val = lam_ctx,
            tempo      = tempo,
            h2h        = h2h_val,
        )
        n = save_predictions_batch(
            match_id        = match["match_id"],
            lambda_raw      = lam_ctx,
            lambda_final    = lam_ctx,
            tempo           = tempo,
            asymmetry       = asym,
            h2h_weighted    = h2h_val,
            h2h_source      = h2h_src,
            predictions     = preds,
            value_bet_line  = value_line,
            value_bet_side  = value_side,
            value_bet_odds  = value_odds,
            safe_bet_line   = safe_line,
            safe_bet_side   = safe_side,
            safe_bet_odds   = safe_odds,
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
        "low_mu":        round(low_for_match.mu,  3),
        "low_k":         round(low_for_match.k,   3),
        "high_mu":       round(high_for_match.mu, 3),
        "high_k":        round(high.k,  3),
        "mixture_w":     round(w, 3),
        "extreme_match": is_extreme,
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
        "predictions":    preds,
        "best_bet":       value_bet_info,  # backward-compat: main dashboard uses VALUE
        "value_bet":      value_bet_info,
        "safe_bet":       safe_bet_info,
        "bookmaker_odds": book_odds or {},
        "bookmaker_1x2":  book_1x2,
        "has_bookmaker":  bool(book_odds),
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }


# ── Prediction log line ───────────────────────────────────────────────────────

def _log_pred(pred: dict) -> None:
    # With the bookmaker filter active, preds may not contain 5.5 / 6.5 / 7.5
    # for very-high-scoring matches. Pick the line closest to λ_total as an
    # anchor, and fall back to "—" for any of the fixed log columns that are
    # missing.
    preds = pred.get("predictions") or {}
    lam   = float(pred.get("lambda_total") or 6.5)
    anchor_key = None
    if preds:
        try:
            anchor_key = min(preds.keys(), key=lambda k: abs(float(k) - lam))
        except Exception:
            anchor_key = next(iter(preds))
    anchor = preds.get(anchor_key, {}) if anchor_key else {}

    def _pct(key: str) -> str:
        v = preds.get(key)
        return f"{round(v['p_over'] * 100)}%" if v else "—"

    ht_str = " HIGH_TEMPO" if pred.get("high_tempo") else ""
    val_str = ""
    if pred.get("best_bet"):
        bb = pred["best_bet"]
        val_str = f" | ★ TYP: {bb['side_pl']} {bb['line']} {bb['prob']*100:.1f}% [{bb['label']}]"
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
        "    anchor=%s p_raw=%.3f  p_final=%.3f"
        "  O5.5=%s  O6.5=%s  O7.5=%s%s",
        anchor_key or "—",
        anchor.get("p_over_raw", 0.0), anchor.get("p_over", 0.0),
        _pct("5.5"), _pct("6.5"), _pct("7.5"),
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

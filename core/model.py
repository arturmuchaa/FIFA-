"""
Poisson-based over/under model for FIFA-style matches.

For a match between player1 and player2:

  λ1 = (attack_p1 + defense_p2) / 2
  λ2 = (attack_p2 + defense_p1) / 2

  attack  = avg_goals_scored
  defense = avg_goals_conceded

Total goals T = X1 + X2  where X1 ~ Poisson(λ1), X2 ~ Poisson(λ2)

P(T > line) is computed via the CDF of a Poisson(λ1 + λ2).

Lines evaluated: 3.5, 4.5, 5.5, 6.5, 7.5, 8.5
"""

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

# Fallback lambda when no stats available
DEFAULT_LAMBDA = 3.0


def _poisson_cdf(lam: float, k_max: int) -> float:
    """P(X ≤ k_max) for X ~ Poisson(lam)."""
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


def _odds(prob: float) -> float:
    """Convert probability to decimal odds, clipped to [1.01, 100]."""
    if prob <= 0:
        return 100.0
    if prob >= 1:
        return 1.01
    return round(1 / prob, 2)


def predict_match(
    player1: str,
    player2: str,
    players: dict[str, dict],
) -> dict[str, Any]:
    """
    Build over/under predictions for a single match.

    Returns:
    {
      "player1": ...,
      "player2": ...,
      "lambda1": ...,
      "lambda2": ...,
      "lambda_total": ...,
      "predictions": {
        "3.5": {"over": 1.80, "under": 2.10, "p_over": 0.556, "p_under": 0.444},
        ...
      }
    }
    """
    s1 = players.get(player1, {})
    s2 = players.get(player2, {})

    attack1 = s1.get("avg_goals_scored", DEFAULT_LAMBDA)
    defense1 = s1.get("avg_goals_conceded", DEFAULT_LAMBDA)
    attack2 = s2.get("avg_goals_scored", DEFAULT_LAMBDA)
    defense2 = s2.get("avg_goals_conceded", DEFAULT_LAMBDA)

    lambda1 = (attack1 + defense2) / 2
    lambda2 = (attack2 + defense1) / 2
    lambda_total = lambda1 + lambda2

    predictions: dict[str, dict] = {}
    for line in LINES:
        k = int(line)          # floor of the line (e.g. 3 for 3.5)
        p_under = _poisson_cdf(lambda_total, k)
        p_over = 1.0 - p_under

        # guard against zero / near-zero
        p_over = max(p_over, 0.001)
        p_under = max(p_under, 0.001)

        predictions[str(line)] = {
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "over": _odds(p_over),
            "under": _odds(p_under),
        }

    return {
        "player1": player1,
        "player2": player2,
        "lambda1": round(lambda1, 3),
        "lambda2": round(lambda2, 3),
        "lambda_total": round(lambda_total, 3),
        "predictions": predictions,
    }


def predict_all_upcoming(
    matches: list[dict],
    players: dict[str, dict],
) -> list[dict]:
    """
    Run predict_match for every upcoming match.
    Attaches predictions directly to each match dict (shallow copy returned).
    """
    results = []
    for m in matches:
        if m.get("source") != "upcoming":
            continue
        try:
            pred = predict_match(m["player1"], m["player2"], players)
            enriched = {**m, **pred}
            results.append(enriched)
        except Exception as exc:
            logger.warning(f"Prediction failed for match {m.get('match_id')}: {exc}")
    return results

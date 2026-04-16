"""
Bookmaker-to-prediction matcher.

Given:
  * a list of scraped bookmaker entries (player1/player2/totals/1x2)
  * the current list of upcoming-match dicts (from drafted.gg)

attempt to align each bookmaker entry with one of our internal match_ids.
Shuffle.vip and drafted.gg don't always spell names identically (diacritics,
team suffixes, caps), so we fuzzy-match on normalised surnames.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_SPECIAL = {
    "Ł": "L", "ł": "l",
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "Þ": "TH", "þ": "th",
    "Ð": "D", "đ": "d", "Đ": "D",
    "ß": "ss",
}


def _norm(name: str) -> str:
    """Lowercase, strip punctuation, drop non-ASCII accents."""
    if not name:
        return ""
    import unicodedata
    s = "".join(_SPECIAL.get(ch, ch) for ch in name)
    # NFKD decomposes accented letters into base + combining marks, then we
    # drop the combining marks. Handles all Latin diacritics including Polish.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).strip().lower()
    return s


def _tokens(name: str) -> set[str]:
    return {t for t in _norm(name).split() if len(t) >= 2}


def _similarity(a: str, b: str) -> float:
    """
    Jaccard over normalised tokens, with a bonus when either name is fully
    contained in the other (e.g. "Lucas" vs "Lucas Barça").
    Score ∈ [0.0, 1.0].
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    jac = len(inter) / len(union)
    na, nb = _norm(a), _norm(b)
    if na and nb and (na in nb or nb in na):
        jac = max(jac, 0.85)
    return jac


def _pair_score(
    bm_p1: str, bm_p2: str,
    pred_p1: str, pred_p2: str,
) -> float:
    """Best of both orderings — bookmaker home/away may be swapped."""
    straight = (_similarity(bm_p1, pred_p1) + _similarity(bm_p2, pred_p2)) / 2.0
    swapped  = (_similarity(bm_p1, pred_p2) + _similarity(bm_p2, pred_p1)) / 2.0
    return max(straight, swapped)


def match_bookmaker_to_predictions(
    bookmaker_entries: list[dict[str, Any]],
    upcoming_matches:  Iterable[dict[str, Any]],
    min_score:         float = 0.55,
) -> dict[str, dict[str, Any]]:
    """
    Returns {match_id: bookmaker_entry} for every bookmaker entry that clears
    `min_score`.  Each upcoming match gets at most one bookmaker entry (the
    highest-scoring match).
    """
    upcoming = [
        m for m in upcoming_matches
        if m.get("source") == "upcoming" and m.get("match_id")
    ]
    if not upcoming or not bookmaker_entries:
        return {}

    matches_by_id: dict[str, dict[str, Any]] = {}
    best_score: dict[str, float] = {}

    for bm in bookmaker_entries:
        best_mid   = None
        best_val   = 0.0
        for m in upcoming:
            score = _pair_score(
                bm.get("player1", ""), bm.get("player2", ""),
                m.get("player1",  ""), m.get("player2",  ""),
            )
            if score > best_val:
                best_val = score
                best_mid = m["match_id"]
        if best_mid is None or best_val < min_score:
            logger.debug(
                "Bookmaker: no match for %s vs %s (best score %.2f)",
                bm.get("player1"), bm.get("player2"), best_val,
            )
            continue

        # Keep the strongest bookmaker entry per internal match_id
        if best_val > best_score.get(best_mid, 0.0):
            best_score[best_mid]   = best_val
            matches_by_id[best_mid] = bm

    logger.info(
        "Bookmaker match: %d/%d bookmaker entries aligned to upcoming matches",
        len(matches_by_id), len(bookmaker_entries),
    )
    return matches_by_id

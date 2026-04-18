"""
Bookmaker-to-prediction matcher.

Given:
  * a list of scraped bookmaker entries (player1/player2/totals/1x2)
  * the current list of upcoming-match dicts (from drafted.gg)

attempt to align each bookmaker entry with one of our internal match_ids.
Shuffle.vip and drafted.gg don't always spell names identically (diacritics,
team suffixes, caps), so we fuzzy-match on normalised surnames and break
ties using kick-off time proximity.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
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


# ── Date / time parsing for proximity tie-break ──────────────────────────────

_DATE_FORMATS = (
    "%d/%m/%Y %H:%M",
    "%d.%m.%Y %H:%M",
    "%d-%m-%Y %H:%M",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
)


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()[:16]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Try loose search for "HH:MM" alone
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if m:
        try:
            return datetime.strptime(f"{m.group(1).zfill(2)}:{m.group(2)}", "%H:%M")
        except Exception:
            return None
    return None


def _time_delta_minutes(bm_date: str | None, pred_date: str | None) -> float | None:
    """
    Minutes between two date strings — None when either side fails to parse
    or when only one has a date part (we compare time-of-day in that case).
    """
    bm_d  = _parse_date(bm_date)
    pr_d  = _parse_date(pred_date)
    if bm_d is None or pr_d is None:
        return None

    # When either has no real date (only time of day, year==1900) fall back to
    # HH:MM comparison — tournaments run multiple matches per hour so this is
    # still a useful disambiguator.
    if bm_d.year == 1900 or pr_d.year == 1900:
        bm_m = bm_d.hour * 60 + bm_d.minute
        pr_m = pr_d.hour * 60 + pr_d.minute
        diff = abs(bm_m - pr_m)
        return min(diff, 24 * 60 - diff)  # wrap around midnight

    return abs((bm_d - pr_d).total_seconds()) / 60.0


def _time_bonus(delta_min: float | None) -> float:
    """
    Convert a time delta (minutes) into a small score bonus ∈ [0, 0.15].
    Closer kick-off → higher bonus.  Returns 0 when delta is unknown.
    """
    if delta_min is None:
        return 0.0
    if delta_min <= 5:   return 0.15
    if delta_min <= 15:  return 0.10
    if delta_min <= 30:  return 0.06
    if delta_min <= 60:  return 0.02
    return 0.0


def match_bookmaker_to_predictions(
    bookmaker_entries: list[dict[str, Any]],
    upcoming_matches:  Iterable[dict[str, Any]],
    min_score:         float = 0.55,
) -> dict[str, dict[str, Any]]:
    """
    Returns {match_id: bookmaker_entry} for every bookmaker entry that clears
    `min_score`.  Each upcoming match gets at most one bookmaker entry (the
    highest-combined-score match — name similarity + kick-off time proximity).
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
        best_mid       = None
        best_combined  = 0.0
        best_name      = 0.0
        best_delta     = None
        for m in upcoming:
            name_s = _pair_score(
                bm.get("player1", ""), bm.get("player2", ""),
                m.get("player1",  ""), m.get("player2",  ""),
            )
            if name_s < min_score:
                continue
            delta = _time_delta_minutes(bm.get("date"), m.get("date"))
            combined = name_s + _time_bonus(delta)
            if combined > best_combined:
                best_combined = combined
                best_name     = name_s
                best_delta    = delta
                best_mid      = m["match_id"]

        if best_mid is None:
            logger.debug(
                "Bookmaker: no match for %s vs %s",
                bm.get("player1"), bm.get("player2"),
            )
            continue

        logger.debug(
            "Bookmaker pick: %s vs %s → %s (name=%.2f Δt=%s combined=%.2f)",
            bm.get("player1"), bm.get("player2"), best_mid,
            best_name,
            f"{best_delta:.0f}m" if best_delta is not None else "?",
            best_combined,
        )

        # Keep the strongest bookmaker entry per internal match_id
        if best_combined > best_score.get(best_mid, 0.0):
            best_score[best_mid]    = best_combined
            matches_by_id[best_mid] = bm

    logger.info(
        "Bookmaker match: %d/%d bookmaker entries aligned to upcoming matches",
        len(matches_by_id), len(bookmaker_entries),
    )
    return matches_by_id

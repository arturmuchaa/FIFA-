"""
JSON-file database.

matches.json  — list of match dicts (results + upcoming)
players.json  — dict of player → aggregate stats
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MATCHES_PATH     = Path(__file__).parent.parent / "data" / "matches.json"
PLAYERS_PATH     = Path(__file__).parent.parent / "data" / "players.json"
PREDICTIONS_PATH = Path(__file__).parent.parent / "data" / "predictions.json"


# ────────────────────────── low-level IO ────────────────────────────────────

def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists() and path.stat().st_size > 2:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not load {path}: {exc}")
    return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ────────────────────────── matches ─────────────────────────────────────────

def load_matches() -> list[dict]:
    return _load_json(MATCHES_PATH, [])


def save_matches(matches: list[dict]) -> None:
    _save_json(MATCHES_PATH, matches)


def upsert_matches(new_matches: list[dict]) -> int:
    """
    Add new matches to the store, deduplicating by match_id.
    Returns number of newly inserted records.
    """
    existing = load_matches()
    existing_ids = {m["match_id"] for m in existing}

    added = 0
    for m in new_matches:
        if m["match_id"] not in existing_ids:
            existing.append(m)
            existing_ids.add(m["match_id"])
            added += 1

    save_matches(existing)
    logger.info(f"upsert_matches: {added} new records (total {len(existing)})")
    return added


def upsert_match_details(details: list[dict]) -> None:
    """
    Merge modal detail data into existing match records (matched by player names).
    If no existing match is found, the detail is stored as a new upcoming record.
    """
    existing = load_matches()
    detail_map = {
        (d["player1"].lower(), d["player2"].lower()): d for d in details
    }

    for match in existing:
        key = (match["player1"].lower(), match["player2"].lower())
        if key in detail_map:
            detail = detail_map[key]
            match["h2h"] = detail.get("h2h", {})
            match["form"] = detail.get("form", {})
            match["stats"] = detail.get("stats", {})

    save_matches(existing)


# ────────────────────────── players ─────────────────────────────────────────

def load_players() -> dict:
    return _load_json(PLAYERS_PATH, {})


def save_players(players: dict) -> None:
    _save_json(PLAYERS_PATH, players)


def load_predictions() -> list[dict]:
    return _load_json(PREDICTIONS_PATH, [])


def save_predictions(predictions: list[dict]) -> None:
    _save_json(PREDICTIONS_PATH, predictions)


def rebuild_player_stats() -> dict[str, dict]:
    """
    Iterate all completed matches (source='results', has goals1/goals2)
    and compute per-player aggregate stats.

    Fields per player:
      total_matches, wins, losses, draws,
      goals_scored, goals_conceded,
      avg_goals_scored, avg_goals_conceded, winrate
    """
    matches = load_matches()
    completed = [
        m for m in matches
        if m.get("source") == "results"
        and m.get("goals1") is not None
        and m.get("goals2") is not None
    ]

    stats: dict[str, dict] = {}

    def _ensure(name: str) -> dict:
        if name not in stats:
            stats[name] = {
                "total_matches": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "goals_scored": 0,
                "goals_conceded": 0,
            }
        return stats[name]

    for m in completed:
        p1, p2 = m["player1"], m["player2"]
        g1, g2 = int(m["goals1"]), int(m["goals2"])

        s1 = _ensure(p1)
        s2 = _ensure(p2)

        s1["total_matches"] += 1
        s2["total_matches"] += 1

        s1["goals_scored"] += g1
        s1["goals_conceded"] += g2
        s2["goals_scored"] += g2
        s2["goals_conceded"] += g1

        if g1 > g2:
            s1["wins"] += 1
            s2["losses"] += 1
        elif g2 > g1:
            s2["wins"] += 1
            s1["losses"] += 1
        else:
            s1["draws"] += 1
            s2["draws"] += 1

    # compute derived fields
    for name, s in stats.items():
        n = s["total_matches"] or 1
        s["avg_goals_scored"] = round(s["goals_scored"] / n, 3)
        s["avg_goals_conceded"] = round(s["goals_conceded"] / n, 3)
        s["winrate"] = round(s["wins"] / n, 3)

    save_players(stats)
    logger.info(f"rebuild_player_stats: {len(stats)} players updated")
    return stats

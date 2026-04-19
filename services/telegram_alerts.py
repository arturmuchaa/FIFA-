"""
Telegram alert dispatcher for value bets produced by the predictor.

Usage (production):
    from services.telegram_alerts import send_telegram_alert
    send_telegram_alert(bet_data)

`bet_data` is the dict produced downstream of `_best_bets` and enriched
with match context. See MOCK_BET at the bottom for the exact shape.

Configuration (environment variables):
    TELEGRAM_BOT_TOKEN   — from @BotFather
    TELEGRAM_CHAT_ID     — numeric chat/channel id (may start with `-100...`
                           for supergroups/channels)
    TELEGRAM_MATCH_URL   — optional override for the "open match" link.
                           Defaults to the shuffle.vip EFOOTBALL listing
                           because the scraper does not preserve per-match
                           URLs today.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_MATCH_URL = "https://shuffle.vip/pl/sports?section=upcoming&sport=EFOOTBALL"
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SECONDS = 8

# Filter rules — bety poza tym pasmem są ignorowane (return False, brak wysyłki)
_MIN_EV        = 0.05   # 5%
_LINE_MIN      = 5.0
_LINE_MAX      = 6.25

# Dedup store — plik JSON z listą kluczy `{match_id}:{line}:{side}`
_DEDUP_PATH = Path(__file__).resolve().parent.parent / "data" / "telegram_alerts.json"
_DEDUP_LOCK = threading.Lock()
_SENT_CACHE: set[str] | None = None


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def _load_sent() -> set[str]:
    """Load persisted set of already-sent alert keys (lazy, cached)."""
    global _SENT_CACHE
    if _SENT_CACHE is not None:
        return _SENT_CACHE
    try:
        if _DEDUP_PATH.exists():
            with _DEDUP_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _SENT_CACHE = set(data if isinstance(data, list) else [])
        else:
            _SENT_CACHE = set()
    except Exception as exc:
        logger.warning("telegram: could not read dedup store %s: %s", _DEDUP_PATH, exc)
        _SENT_CACHE = set()
    return _SENT_CACHE


def _persist_sent(sent: set[str]) -> None:
    try:
        _DEDUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DEDUP_PATH.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(sorted(sent), f, ensure_ascii=False)
        tmp.replace(_DEDUP_PATH)
    except Exception as exc:
        logger.warning("telegram: could not persist dedup store: %s", exc)


def _alert_key(bet: dict) -> str:
    return f"{bet.get('match_id', '?')}:{bet.get('line', '?')}:{bet.get('side', '?')}"


# ── Filter + format ───────────────────────────────────────────────────────────

def _passes_filter(bet: dict) -> bool:
    try:
        ev   = float(bet.get("ev",   bet.get("edge", 0.0)))
        line = float(bet.get("line", 0.0))
    except (TypeError, ValueError):
        return False
    return ev > _MIN_EV and _LINE_MIN <= line <= _LINE_MAX


def _format_message(bet: dict) -> str:
    team1 = bet.get("team1", bet.get("player1", "?"))
    team2 = bet.get("team2", bet.get("player2", "?"))
    side  = str(bet.get("side_pl") or bet.get("side", "")).upper() or "OVER"
    line  = bet.get("line", "?")
    odds  = bet.get("odds", bet.get("bookmaker_odds", "?"))
    ev    = bet.get("ev",   bet.get("edge", 0.0))
    url   = bet.get("match_url") or os.environ.get("TELEGRAM_MATCH_URL") or _DEFAULT_MATCH_URL

    try:
        ev_pct = f"{float(ev) * 100:.1f}%"
    except (TypeError, ValueError):
        ev_pct = str(ev)
    try:
        odds_str = f"{float(odds):.2f}"
    except (TypeError, ValueError):
        odds_str = str(odds)

    return (
        f"🔥 BET (VALUE)\n\n"
        f"{team1} vs {team2}\n"
        f"Typ: {side} {line}\n"
        f"Kurs: {odds_str}\n"
        f"EV: {ev_pct}\n\n"
        f"🎯 Otwórz mecz:\n"
        f"{url}\n\n"
        f"Instrukcja:\n"
        f"Liczba goli → {side} {line}"
    )


def _inline_keyboard(bet: dict) -> dict:
    url = bet.get("match_url") or os.environ.get("TELEGRAM_MATCH_URL") or _DEFAULT_MATCH_URL
    return {"inline_keyboard": [[{"text": "🎯 Otwórz mecz", "url": url}]]}


# ── Public API ────────────────────────────────────────────────────────────────

def send_telegram_alert(bet_data: dict) -> bool:
    """
    Send a Telegram alert for a value bet if it passes the configured
    filters and hasn't been sent before.

    Returns True when a message was delivered, False otherwise (filtered,
    duplicate, misconfigured, or network error).
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("telegram: skipped — missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
        return False

    if not _passes_filter(bet_data):
        logger.debug(
            "telegram: filtered out line=%s ev=%s",
            bet_data.get("line"), bet_data.get("ev", bet_data.get("edge")),
        )
        return False

    key = _alert_key(bet_data)
    with _DEDUP_LOCK:
        sent = _load_sent()
        if key in sent:
            logger.debug("telegram: duplicate skip %s", key)
            return False

    payload = {
        "chat_id":                  chat_id,
        "text":                     _format_message(bet_data),
        "disable_web_page_preview": True,
        "reply_markup":             json.dumps(_inline_keyboard(bet_data)),
    }
    try:
        resp = requests.post(
            _TELEGRAM_API.format(token=token),
            data=payload,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning("telegram: request failed for %s: %s", key, exc)
        return False

    if resp.status_code != 200:
        logger.warning(
            "telegram: API %s for %s — %s", resp.status_code, key, resp.text[:200],
        )
        return False

    with _DEDUP_LOCK:
        sent = _load_sent()
        sent.add(key)
        _persist_sent(sent)
    logger.info("telegram: sent %s", key)
    return True


# ── Mock example ──────────────────────────────────────────────────────────────

MOCK_BET = {
    "match_id":       "demo_match_001",
    "team1":          "PSG (player1)",
    "team2":          "MCI (player2)",
    "line":           6.0,
    "side":           "over",
    "side_pl":        "OVER",
    "odds":           1.95,
    "ev":             0.082,
    "match_url":      "https://shuffle.vip/pl/sports?section=upcoming&sport=EFOOTBALL",
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = send_telegram_alert(MOCK_BET)
    print("sent:" if ok else "not sent:", MOCK_BET["match_id"])

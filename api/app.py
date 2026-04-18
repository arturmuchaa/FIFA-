"""
FastAPI application — serves the REST API and the Web UI.

Endpoints:
  GET /          → HTML dashboard (table with over/under odds)
  GET /matches   → JSON list of upcoming matches with predictions
  GET /players   → JSON player stats
  GET /history   → JSON completed match history
  POST /refresh  → trigger manual scrape cycle (non-blocking)
"""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse

from core.database import load_matches, load_players, load_predictions, rebuild_player_stats
from core.model import predict_all_upcoming

logger = logging.getLogger(__name__)

app = FastAPI(title="Valhalla Cup Predictor", version="1.0.0")

# ─────────────────────────── helpers ────────────────────────────────────────

def _get_predictions() -> list[dict]:
    """
    Serve pre-computed predictions from data/predictions.json (written by the
    background predictor each cycle).  Fall back to the on-the-fly Poisson
    model if no persisted predictions exist yet.
    """
    preds = load_predictions()
    if preds:
        return preds
    # fallback: compute on-the-fly with the simple model
    matches = load_matches()
    players = load_players()
    upcoming = [m for m in matches if m.get("source") == "upcoming"]
    return predict_all_upcoming(upcoming, players)


# ─────────────────────────── REST API ───────────────────────────────────────

@app.get("/matches", response_class=JSONResponse)
async def get_matches():
    """Return upcoming matches with over/under predictions."""
    return _get_predictions()


@app.get("/predictions", response_class=JSONResponse)
async def get_predictions():
    """Return the latest pre-computed predictions (same as /matches)."""
    return _get_predictions()


@app.get("/players", response_class=JSONResponse)
async def get_players():
    """Return per-player statistics."""
    return load_players()


@app.get("/history", response_class=JSONResponse)
async def get_history():
    """Return completed match history."""
    matches = load_matches()
    return [m for m in matches if m.get("source") == "results"]


@app.post("/refresh", response_class=JSONResponse)
async def manual_refresh(background_tasks: BackgroundTasks):
    """Trigger a scrape cycle in the background."""
    from main import run_cycle
    background_tasks.add_task(run_cycle)
    return {"status": "refresh started"}


@app.post("/api/reset_history", response_class=JSONResponse)
async def reset_history():
    """
    Clear prediction history + bookmaker odds so /typy statistics restart
    from scratch. The historical `matches` table stays intact — the model
    keeps its training data.
    """
    try:
        from core.db_sqlite import reset_prediction_history, init_db
        from core.database import save_predictions
        init_db()
        reset_prediction_history()
        save_predictions([])  # empty the JSON cache used by /matches fallback
        return {"status": "ok"}
    except Exception as exc:
        logger.error("reset_history failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/wynik", response_class=JSONResponse)
async def submit_result(request: Request):
    """
    Record actual total goals for a match.
    Body: {"match_id": "...", "actual_goals": 7}
    Settles all prediction rows for this match in SQLite → feeds calibration.
    """
    try:
        body = await request.json()
        match_id     = str(body["match_id"])
        actual_goals = int(body["actual_goals"])
    except Exception as exc:
        return JSONResponse({"error": f"Invalid body: {exc}"}, status_code=400)

    try:
        from core.db_sqlite import settle_match
        updated = settle_match(match_id, actual_goals)
        return {"status": "ok", "match_id": match_id, "actual_goals": actual_goals, "rows_settled": updated}
    except Exception as exc:
        logger.error("settle_match failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ─────────────────────────── Web UI ─────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Valhalla Cup — Over/Under Predictor</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background: #0f1117;
      color: #e2e8f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      padding: 24px;
    }}

    h1 {{
      font-size: 1.8rem;
      color: #f59e0b;
      margin-bottom: 6px;
      letter-spacing: 1px;
    }}

    .subtitle {{
      color: #64748b;
      font-size: 0.85rem;
      margin-bottom: 28px;
    }}

    .refresh-btn {{
      display: inline-block;
      margin-bottom: 24px;
      padding: 8px 20px;
      background: #1e40af;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.9rem;
      transition: background 0.2s;
    }}
    .refresh-btn:hover {{ background: #2563eb; }}

    .match-card {{
      background: #1e2130;
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }}

    .match-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }}

    .player {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #e2e8f0;
    }}

    .vs {{
      color: #f59e0b;
      font-weight: 800;
      font-size: 0.95rem;
    }}

    .match-meta {{
      font-size: 0.78rem;
      color: #64748b;
      margin-left: auto;
    }}

    .lambda-row {{
      font-size: 0.8rem;
      color: #94a3b8;
      margin-bottom: 12px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    th {{
      text-align: left;
      font-size: 0.75rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 6px 10px;
      border-bottom: 1px solid #2d3748;
    }}

    td {{
      padding: 7px 10px;
      font-size: 0.9rem;
      border-bottom: 1px solid #1a2035;
    }}

    .line {{ color: #94a3b8; font-weight: 600; }}

    .over {{ color: #34d399; }}
    .under {{ color: #f87171; }}

    .book {{ color: #fbbf24; font-weight: 600; }}
    .book-na {{ color: #4a5568; }}

    .pct {{
      font-size: 0.72rem;
      color: #64748b;
      margin-left: 4px;
    }}

    .book-tag {{
      display: inline-block;
      margin-left: 8px;
      padding: 1px 8px;
      border-radius: 10px;
      background: #422006;
      color: #fbbf24;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}

    .no-book-tag {{
      display: inline-block;
      margin-left: 8px;
      padding: 1px 8px;
      border-radius: 10px;
      background: #1f2937;
      color: #64748b;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}

    .no-data {{
      color: #4a5568;
      text-align: center;
      padding: 40px;
      font-size: 1rem;
    }}

    .stats-row {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin-top: 8px;
      font-size: 0.78rem;
      color: #94a3b8;
    }}

    .last-updated {{
      font-size: 0.75rem;
      color: #4a5568;
      margin-top: 32px;
      text-align: right;
    }}

    .best-bet {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      padding: 10px 14px;
      background: #0b0f1a;
      border-radius: 8px;
      flex-wrap: wrap;
    }}

    .best-bet-label {{
      font-size: 0.68rem;
      color: #4a5568;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      white-space: nowrap;
    }}

    .best-bet-badge {{
      font-weight: 800;
      font-size: 0.95rem;
      white-space: nowrap;
    }}

    .best-bet-conf {{
      font-size: 0.75rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 10px;
      background: rgba(255,255,255,0.06);
    }}

    .best-bet-odds {{
      font-size: 0.8rem;
      color: #94a3b8;
      margin-left: auto;
    }}
  </style>
</head>
<body>

<h1>⚽ Valhalla Cup — Over/Under Predictor</h1>
<p class="subtitle">Model matematyczny · Poisson · Dane ze strony drafted.gg</p>

<button class="refresh-btn" onclick="triggerRefresh()">⟳ Odśwież dane</button>

{content}

<p class="last-updated">Ostatnia aktualizacja: {updated}</p>

<script>
  async function triggerRefresh() {{
    const btn = document.querySelector('.refresh-btn');
    btn.textContent = '⏳ Odświeżanie…';
    btn.disabled = true;
    try {{
      await fetch('/refresh', {{method: 'POST'}});
      setTimeout(() => location.reload(), 5000);
    }} catch(e) {{
      btn.textContent = '⚠️ Błąd';
      btn.disabled = false;
    }}
  }}

  // auto-reload every 60 s
  setTimeout(() => location.reload(), 60000);
</script>

</body>
</html>
"""

CARD_TEMPLATE = """\
<div class="match-card">
  <div class="match-header">
    <span class="player">{p1}</span>
    <span class="vs">VS</span>
    <span class="player">{p2}</span>
    {book_tag}
    <span class="match-meta">{date}</span>
  </div>
  <div class="lambda-row">
    λ₁={lam1} &nbsp;|&nbsp; λ₂={lam2} &nbsp;|&nbsp;
    <strong>λ={lam_total}</strong>
    &nbsp;|&nbsp; tempo={tempo} [{style}]
    &nbsp;|&nbsp; h2h={h2h}
    &nbsp;|&nbsp; <span style="color:#64748b;font-size:0.75rem">src:{src}</span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Linia</th>
        <th>OVER model</th>
        <th>UNDER model</th>
        <th>OVER bukm.</th>
        <th>UNDER bukm.</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {stats_section}
  {best_bet_section}
</div>
"""

ROW_TEMPLATE = """\
<tr>
  <td class="line">{line}</td>
  <td class="over">{over}<span class="pct">({p_over}%)</span></td>
  <td class="under">{under}<span class="pct">({p_under}%)</span></td>
  <td class="{book_over_cls}">{book_over}</td>
  <td class="{book_under_cls}">{book_under}</td>
</tr>
"""


def _render_stats(match: dict) -> str:
    stats = match.get("stats", {})
    p1s = stats.get("player1", {})
    p2s = stats.get("player2", {})
    h2h = match.get("h2h", {})
    if not (p1s or p2s or h2h):
        return ""

    items = []
    if h2h.get("wins_player1") is not None:
        items.append(
            f"H2H: {h2h['wins_player1']}–{h2h.get('wins_player2', '?')}"
        )
    if h2h.get("avg_goals_per_match"):
        items.append(f"Avg goals/match: {h2h['avg_goals_per_match']}")
    if p1s.get("wins_pct"):
        items.append(
            f"Win%: {p1s['wins_pct']}% vs {p2s.get('wins_pct', '?')}%"
        )
    if not items:
        return ""

    return (
        '<div class="stats-row">'
        + "".join(f"<span>{it}</span>" for it in items)
        + "</div>"
    )


def _render_best_bet(match: dict) -> str:
    """Return HTML for the TYP MODELU best-bet banner, or empty string."""
    bb = match.get("best_bet")
    if not bb:
        return ""
    color = bb.get("color", "#94a3b8")

    # Prefer the real bookmaker price when available
    book_odds = bb.get("bookmaker_odds")
    edge      = bb.get("edge")
    source    = bb.get("source", "legacy")

    if book_odds:
        odds_str = f'kurs bukm. <b style="color:#fbbf24">{book_odds}</b> · model {bb["model_odds"]}'
        if edge is not None:
            edge_pct = round(edge * 100, 1)
            edge_col = "#34d399" if edge > 0 else "#f87171"
            odds_str += (
                f' &nbsp;·&nbsp; <span style="color:{edge_col};font-weight:700">'
                f'EV {edge_pct:+.1f}%</span>'
            )
    else:
        odds_str = f'kurs modelu {bb["model_odds"]} · brak linii u bukmachera'

    tag = ""
    if source == "value":
        tag = '<span style="color:#fbbf24;font-weight:700;margin-left:6px">VALUE</span>'
    elif source == "fallback":
        tag = '<span style="color:#94a3b8;margin-left:6px">(fallback)</span>'

    return (
        f'<div class="best-bet">'
        f'<span class="best-bet-label">★ Typ modelu</span>'
        f'<span class="best-bet-badge" style="color:{color}">'
        f'{bb["side_pl"]} {bb["line"]}</span>'
        f'<span class="best-bet-conf" style="color:{color}">{bb["label"]}</span>'
        f'{tag}'
        f'<span class="best-bet-odds">'
        f'{round(bb["prob"]*100,1)}% &nbsp;·&nbsp; {odds_str}</span>'
        f'</div>'
    )


TYPY_TEMPLATE = """\
<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Valhalla — Historia typów</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background:#0f1117; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif; padding:20px; }}
    h1 {{ font-size:1.6rem; color:#f59e0b; margin-bottom:4px; }}
    .sub {{ color:#64748b; font-size:0.82rem; margin-bottom:24px; }}
    .nav {{ margin-bottom:20px; }}
    .nav a {{ color:#60a5fa; text-decoration:none; margin-right:16px; font-size:0.88rem; }}
    .nav a:hover {{ color:#93c5fd; }}
    .card {{ background:#1e2130; border:1px solid #2d3748; border-radius:10px; padding:16px 20px; margin-bottom:16px; }}
    .card.settled {{ border-color:#1f4035; }}
    .hdr {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap; }}
    .players {{ font-size:1.05rem; font-weight:700; }}
    .vs {{ color:#f59e0b; font-weight:800; }}
    .meta {{ font-size:0.75rem; color:#64748b; margin-left:auto; }}
    .badge-settled {{ background:#065f46; color:#6ee7b7; padding:2px 8px; border-radius:12px; font-size:0.72rem; }}
    .badge-open {{ background:#1e3a5f; color:#93c5fd; padding:2px 8px; border-radius:12px; font-size:0.72rem; }}
    .lam-row {{ font-size:0.75rem; color:#94a3b8; margin-bottom:10px; }}
    table {{ width:100%; border-collapse:collapse; margin-bottom:12px; }}
    th {{ font-size:0.7rem; color:#64748b; text-transform:uppercase; padding:5px 8px; border-bottom:1px solid #2d3748; text-align:left; }}
    td {{ padding:5px 8px; font-size:0.85rem; border-bottom:1px solid #1a2035; }}
    .line {{ color:#94a3b8; font-weight:600; }}
    .pct-over {{ color:#34d399; }}
    .pct-under {{ color:#f87171; }}
    .result-hit {{ color:#34d399; font-weight:700; }}
    .result-miss {{ color:#f87171; font-weight:700; }}
    .result-na {{ color:#4a5568; }}
    .settle-form {{ display:flex; align-items:center; gap:10px; margin-top:8px; flex-wrap:wrap; }}
    .settle-form label {{ font-size:0.82rem; color:#94a3b8; }}
    .settle-form input {{ background:#0f1117; border:1px solid #3d4a5c; border-radius:6px;
                          color:#e2e8f0; padding:5px 10px; width:80px; font-size:0.9rem; }}
    .settle-form input:focus {{ outline:none; border-color:#60a5fa; }}
    .settle-btn {{ background:#1e40af; color:#fff; border:none; border-radius:6px;
                   padding:6px 16px; cursor:pointer; font-size:0.85rem; }}
    .settle-btn:hover {{ background:#2563eb; }}
    .settle-btn:disabled {{ background:#374151; cursor:not-allowed; }}
    .actual-result {{ font-size:0.9rem; }}
    .no-data {{ color:#4a5568; text-align:center; padding:40px; }}
    .cal-section {{ background:#131927; border:1px solid #2d3748; border-radius:8px;
                    padding:14px 18px; margin-bottom:24px; }}
    .cal-section h3 {{ font-size:0.9rem; color:#94a3b8; margin-bottom:10px; }}
    .cal-table th {{ color:#4a5568; }}
    .cal-ok {{ color:#34d399; }}
    .cal-bad {{ color:#f87171; }}
    .updated {{ font-size:0.72rem; color:#374151; text-align:right; margin-top:24px; }}
    .stat-section {{ background:#131927; border:1px solid #2d3748; border-radius:8px;
                     padding:14px 18px; margin-bottom:24px; }}
    .stat-section h3 {{ font-size:0.9rem; color:#94a3b8; margin-bottom:10px; }}
    .stat-good {{ color:#34d399; font-weight:700; }}
    .stat-bad  {{ color:#f87171; font-weight:700; }}
    .stat-na   {{ color:#4a5568; }}
    .label-pewny {{ color:#34d399; font-weight:700; }}
    .label-dobry {{ color:#60a5fa; font-weight:700; }}
    .label-ok    {{ color:#94a3b8; font-weight:700; }}
  </style>
</head>
<body>
<h1>Historia typów</h1>
<p class="sub">Jeden typ na mecz · automatyczna weryfikacja po wpisaniu wyniku · statystyki flat-bet</p>
<div class="nav">
  <a href="/">Powrót do typów</a>
  <a href="/api/kalibracja">JSON kalibracji</a>
  <a href="#" onclick="resetHistory(event)" style="color:#f87171">Wyczyść historię</a>
</div>

{stats_section}

{cal_section}

{content}

<p class="updated">Wygenerowano: {updated}</p>

<script>
async function resetHistory(evt) {{
  evt.preventDefault();
  if (!confirm('Wyczyścić całą historię typów i odłączyć zapisane kursy bukmachera? Historia meczów do modelu zostaje.')) return;
  try {{
    const r = await fetch('/api/reset_history', {{method: 'POST'}});
    const d = await r.json();
    if (d.status === 'ok') {{
      location.reload();
    }} else {{
      alert('Błąd: ' + (d.error || 'nieznany'));
    }}
  }} catch(e) {{
    alert('Błąd sieci: ' + e);
  }}
}}

async function settle(matchId, btn) {{
  const form = btn.closest('.settle-form');
  const goalsInput = form.querySelector('input[type=number]');
  const goals = parseInt(goalsInput.value);
  if (isNaN(goals) || goals < 0 || goals > 30) {{
    alert('Podaj prawidłowy wynik (0-30)');
    return;
  }}
  btn.disabled = true;
  btn.textContent = '⏳';
  try {{
    const resp = await fetch('/api/wynik', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{match_id: matchId, actual_goals: goals}})
    }});
    const data = await resp.json();
    if (data.status === 'ok') {{
      btn.textContent = '✓ Zapisano';
      btn.style.background = '#065f46';
      setTimeout(() => location.reload(), 800);
    }} else {{
      btn.textContent = '⚠️ Błąd';
      btn.disabled = false;
    }}
  }} catch(e) {{
    btn.textContent = '⚠️ Błąd';
    btn.disabled = false;
  }}
}}
</script>
</body>
</html>
"""


@app.get("/typy", response_class=HTMLResponse)
async def typy_page():
    """Historia typów modelu z możliwością wpisania wyników."""
    from datetime import datetime, timezone
    from core.db_sqlite import (
        get_prediction_history, get_line_calibration,
        get_model_stats, init_db, backfill_match_info,
    )
    from core.database import load_predictions

    try:
        init_db()
        # Backfill player names for predictions stored before match_info existed
        backfill_match_info(load_predictions())
        history   = get_prediction_history(limit=60)
        cal       = get_line_calibration(min_samples=5)
        mstats    = get_model_stats()
    except Exception as exc:
        logger.error("typy_page error: %s", exc)
        history, cal, mstats = [], {}, {}

    # ── model accuracy stats section ────────────────────────────────────
    total_settled = mstats.get("total_settled", 0)
    flat_stake    = mstats.get("flat_stake", 100.0)
    if total_settled > 0:
        bb      = mstats.get("best_bets", {})
        by_line = mstats.get("by_line", {})

        n_bets   = bb.get("bets", 0)
        n_wins   = bb.get("wins", 0)
        n_losses = bb.get("losses", 0)
        profit   = bb.get("profit", 0.0)
        staked   = bb.get("staked", 0.0)
        win_rate = bb.get("win_rate", 0.0)
        yld      = bb.get("yield_pct", 0.0)

        profit_str  = f"+{profit:.0f} zł" if profit >= 0 else f"{profit:.0f} zł"
        profit_cls  = "stat-good" if profit > 0 else "stat-bad" if profit < 0 else ""
        yld_str     = f"{yld:+.1f}%"
        yld_cls     = "stat-good" if yld > 0 else "stat-bad" if yld < 0 else ""
        wr_cls      = "stat-good" if win_rate >= 0.60 else "stat-bad" if win_rate < 0.50 else ""

        # Summary banner
        summary_html = (
            f'<div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;padding:12px 16px;'
            f'background:#0b0f1a;border-radius:8px;align-items:center">'
            f'<div><div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase">Typy</div>'
            f'<div style="font-size:1.1rem;font-weight:700">{n_bets}</div></div>'
            f'<div><div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase">Trafione</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:#34d399">{n_wins}</div></div>'
            f'<div><div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase">Chybione</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:#f87171">{n_losses}</div></div>'
            f'<div><div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase">Skuteczność</div>'
            f'<div class="{wr_cls}" style="font-size:1.1rem;font-weight:700">{round(win_rate*100)}%</div></div>'
            f'<div><div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase">Zysk/Strata</div>'
            f'<div class="{profit_cls}" style="font-size:1.1rem;font-weight:700">{profit_str}</div></div>'
            f'<div><div style="font-size:0.68rem;color:#4a5568;text-transform:uppercase">Yield</div>'
            f'<div class="{yld_cls}" style="font-size:1.1rem;font-weight:700">{yld_str}</div></div>'
            f'<div style="margin-left:auto"><div style="font-size:0.68rem;color:#4a5568">Stawka</div>'
            f'<div style="font-size:0.85rem;color:#64748b">{flat_stake:.0f} zł / typ</div></div>'
            f'</div>'
        )

        # Per-label rows
        lbl_rows = ""
        for lbl, lbl_cls in [("PEWNY", "label-pewny"), ("DOBRY", "label-dobry"), ("OK", "label-ok")]:
            s  = bb.get("by_label", {}).get(lbl, {})
            nb = s.get("bets", 0)
            if nb == 0:
                lbl_rows += (
                    f"<tr><td class='{lbl_cls}'>{lbl}</td>"
                    f"<td class='stat-na'>—</td><td class='stat-na'>—</td>"
                    f"<td class='stat-na'>—</td><td class='stat-na'>brak danych</td></tr>"
                )
            else:
                nw = s["wins"]; pr = s["profit"]; yr = s["yield_pct"]
                wr2 = s["win_rate"]
                pr_s = f"+{pr:.0f} zł" if pr >= 0 else f"{pr:.0f} zł"
                pr_c = "stat-good" if pr > 0 else "stat-bad"
                wr2_c = "stat-good" if wr2 >= 0.60 else "stat-bad" if wr2 < 0.50 else ""
                yr_c = "stat-good" if yr > 0 else "stat-bad"
                lbl_rows += (
                    f"<tr><td class='{lbl_cls}'>{lbl}</td>"
                    f"<td>{nw}/{nb}</td>"
                    f"<td class='{wr2_c}'>{round(wr2*100)}%</td>"
                    f"<td class='{pr_c}'>{pr_s}</td>"
                    f"<td class='{yr_c}'>{yr:+.1f}%</td></tr>"
                )

        # Per-line rows
        line_rows = ""
        for line_str in sorted(by_line.keys(), key=float):
            s  = by_line[line_str]
            n  = s["total"]
            c2 = s["correct"]
            a  = s["accuracy"]
            a_cls = "stat-good" if a >= 0.60 else "stat-bad" if a < 0.50 else ""
            line_rows += f"<tr><td class='line'>{line_str}</td><td>{c2}/{n}</td><td class='{a_cls}'>{round(a*100)}%</td></tr>"

        stats_section = (
            '<div class="stat-section">'
            '<h3>Statystyki modelu — flat-bet 100 zł/typ</h3>'
            + summary_html
            + '<table><thead><tr><th>Etykieta</th><th>W/L</th><th>Skuteczność</th>'
            '<th>Zysk/Strata</th><th>Yield</th></tr></thead>'
            f'<tbody>{lbl_rows}</tbody></table>'
            '<details style="margin-top:10px">'
            '<summary style="font-size:0.78rem;color:#64748b;cursor:pointer">Szczegóły per linia</summary>'
            '<table style="margin-top:8px"><thead><tr><th>Linia</th><th>Trafione/Łącznie</th><th>Skuteczność</th></tr></thead>'
            f'<tbody>{line_rows}</tbody></table>'
            '</details>'
            '</div>'
        )
    else:
        stats_section = (
            '<div class="stat-section">'
            '<p style="color:#4a5568;font-size:0.82rem">'
            'Statystyki flat-bet dostępne po pierwszych rozegranych meczach.</p>'
            '</div>'
        )

    # ── calibration summary section ──────────────────────────────────────
    if cal:
        rows_cal = ""
        for line in sorted(cal.keys()):
            offset = cal[line]
            cls = "cal-ok" if abs(offset) < 0.03 else "cal-bad"
            direction = "przeszacowany" if offset > 0 else "niedoszacowany"
            rows_cal += (
                f"<tr><td class='line'>{line}</td>"
                f"<td class='{cls}'>{offset:+.3f}</td>"
                f"<td style='color:#64748b;font-size:0.75rem'>{direction}</td></tr>"
            )
        cal_section = (
            '<div class="cal-section">'
            '<h3>Kalibracja modelu (z rzeczywistych wyników)</h3>'
            '<table><thead><tr><th>Linia</th><th>Błąd średni</th><th>Kierunek</th></tr></thead>'
            f'<tbody>{rows_cal}</tbody></table>'
            '<p style="font-size:0.72rem;color:#4a5568;margin-top:4px">'
            'Błąd &gt; 0 → model przeszacowuje over → korekta odejmowana automatycznie</p>'
            '</div>'
        )
    else:
        cal_section = (
            '<div class="cal-section">'
            '<p style="color:#4a5568;font-size:0.82rem">Kalibracja dostępna po wpisaniu ≥5 wyników na linię.</p>'
            '</div>'
        )

    # ── match cards ───────────────────────────────────────────────────────
    if not history:
        content = '<p class="no-data">Brak historii typów. Poczekaj na pierwszy cykl predykcji.</p>'
    else:
        cards = []
        for m in history:
            settled  = m["is_settled"]
            ag       = m.get("actual_goals")
            bet      = m.get("bet") or {}
            won      = bet.get("won")

            if settled and won is True:
                badge = '<span class="badge-settled" style="background:#065f46;color:#6ee7b7">✓ Trafiony</span>'
            elif settled and won is False:
                badge = '<span class="badge-settled" style="background:#7f1d1d;color:#fca5a5">✗ Chybiony</span>'
            elif settled:
                badge = '<span class="badge-settled">✓ Rozegrany</span>'
            else:
                badge = '<span class="badge-open">⏳ Oczekuje</span>'

            lam_str   = f"λ={m['lambda_val']:.2f}" if m.get("lambda_val") else ""
            tempo_str = f" | tempo={m['tempo']:.2f}" if m.get("tempo") else ""
            h2h_str   = f" | h2h={m['h2h']:.1f}" if m.get("h2h") else ""

            # Single-bet banner (replaces the per-line table)
            side_pl    = bet.get("side_pl", "—")
            line_val   = bet.get("line")
            line_str   = f"{line_val:g}" if isinstance(line_val, (int, float)) else "—"
            prob_pct   = round(bet.get("prob", 0) * 100, 1)
            side_color = "#34d399" if bet.get("side") == "over" else "#f87171"
            bk_odds    = bet.get("book_odds")
            md_odds    = bet.get("model_odds")

            odds_parts = []
            if bk_odds:
                odds_parts.append(f'kurs bukm. <b style="color:#fbbf24">{bk_odds:.2f}</b>')
            if md_odds:
                odds_parts.append(f'kurs modelu {md_odds:.2f}')
            odds_str = " · ".join(odds_parts) if odds_parts else ""

            if settled and won is True:
                outcome_str = f'<span class="result-hit">✓ WYGRANY</span>'
            elif settled and won is False:
                outcome_str = f'<span class="result-miss">✗ PRZEGRANY</span>'
            else:
                outcome_str = ''

            bet_banner = (
                f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;'
                f'padding:10px 14px;background:#0b0f1a;border-radius:8px;margin-bottom:10px">'
                f'<span style="font-size:0.68rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.6px">★ Typ modelu</span>'
                f'<span style="font-weight:800;font-size:1.05rem;color:{side_color}">{side_pl} {line_str}</span>'
                f'<span style="font-size:0.78rem;color:#94a3b8">{prob_pct}% pewności</span>'
                f'<span style="font-size:0.78rem;color:#94a3b8">{odds_str}</span>'
                f'<span style="margin-left:auto;font-size:0.85rem">{outcome_str}</span>'
                f'</div>'
            )

            if settled:
                result_block = f'<div class="actual-result" style="color:#6ee7b7">Wynik końcowy: <b>{ag}</b> goli łącznie</div>'
            else:
                result_block = (
                    f'<div class="settle-form">'
                    f'<label>Łączna liczba goli:</label>'
                    f'<input type="number" min="0" max="30" placeholder="np. 7">'
                    f'<button class="settle-btn" onclick="settle(\'{m["match_id"]}\', this)">Zapisz wynik</button>'
                    f'</div>'
                )

            card = (
                f'<div class="card {"settled" if settled else ""}">'
                f'<div class="hdr">'
                f'<span class="players">{m["player1"]} <span class="vs">VS</span> {m["player2"]}</span>'
                f'{badge}'
                f'<span class="meta">{m.get("date") or m.get("created_at","")[:16]}</span>'
                f'</div>'
                f'<div class="lam-row">{lam_str}{tempo_str}{h2h_str}</div>'
                f'{bet_banner}'
                f'{result_block}'
                f'</div>'
            )
            cards.append(card)
        content = "\n".join(cards)

    html = TYPY_TEMPLATE.format(
        stats_section = stats_section,
        cal_section   = cal_section,
        content       = content,
        updated       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    return HTMLResponse(content=html)


@app.get("/api/kalibracja", response_class=JSONResponse)
async def api_kalibracja():
    """JSON z bieżącymi offsetami kalibracji per linia."""
    from core.db_sqlite import get_line_calibration, init_db
    init_db()
    return get_line_calibration(min_samples=5)


@app.get("/api/statystyki", response_class=JSONResponse)
async def api_statystyki():
    """JSON ze statystykami dokładności modelu (per linia i typy pewne)."""
    from core.db_sqlite import get_model_stats, init_db
    init_db()
    return get_model_stats()


@app.get("/", response_class=HTMLResponse)
async def root():
    from datetime import datetime, timezone

    predictions = _get_predictions()

    # Keep only future matches, sorted by date, max 30
    def _sort_key(m: dict) -> str:
        return m.get("date") or ""

    predictions = sorted(predictions, key=_sort_key)[:30]

    if not predictions:
        content = '<p class="no-data">Brak danych — poczekaj na pierwszy cykl scrapowania lub kliknij Odśwież.</p>'
    else:
        cards = []
        for m in predictions:
            try:
                rows_html = ""
                for line, vals in m.get("predictions", {}).items():
                    bk_o = vals.get("book_over")
                    bk_u = vals.get("book_under")
                    rows_html += ROW_TEMPLATE.format(
                        line=line,
                        over=vals["over"],
                        under=vals["under"],
                        p_over=round(vals["p_over"] * 100, 1),
                        p_under=round(vals["p_under"] * 100, 1),
                        book_over=bk_o if bk_o else "—",
                        book_under=bk_u if bk_u else "—",
                        book_over_cls="book" if bk_o else "book-na",
                        book_under_cls="book" if bk_u else "book-na",
                    )
                sa   = m.get("style_a", "?")
                sb   = m.get("style_b", "?")
                has_bm = m.get("has_bookmaker") or any(
                    (v.get("book_over") or v.get("book_under"))
                    for v in (m.get("predictions", {}) or {}).values()
                )
                book_tag = (
                    '<span class="book-tag">BUKM</span>'
                    if has_bm
                    else '<span class="no-book-tag">brak bukm.</span>'
                )
                card = CARD_TEMPLATE.format(
                    p1=m["player1"],
                    p2=m["player2"],
                    date=m.get("date", "—"),
                    lam1=m.get("lambda1", "—"),
                    lam2=m.get("lambda2", "—"),
                    lam_total=m.get("lambda_total", "—"),
                    tempo=m.get("tempo_avg", "—"),
                    style=f"{sa}v{sb}",
                    h2h=m.get("h2h_avg_goals", "—"),
                    src=f"{m.get('stat_src_a','?')}/{m.get('stat_src_b','?')}",
                    rows=rows_html,
                    book_tag=book_tag,
                    stats_section=_render_stats(m),
                    best_bet_section=_render_best_bet(m),
                )
                cards.append(card)
            except Exception as exc:
                logger.warning(f"Card render error for {m.get('player1')} vs {m.get('player2')}: {exc}")
        content = "\n".join(cards) if cards else '<p class="no-data">Brak nadchodzących meczów.</p>'

    html = HTML_TEMPLATE.format(
        content=content,
        updated=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    return HTMLResponse(content=html)

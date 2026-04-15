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

from fastapi import FastAPI, BackgroundTasks
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

    .pct {{
      font-size: 0.72rem;
      color: #64748b;
      margin-left: 4px;
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
        <th>OVER</th>
        <th>UNDER</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {stats_section}
</div>
"""

ROW_TEMPLATE = """\
<tr>
  <td class="line">{line}</td>
  <td class="over">{over}<span class="pct">({p_over}%)</span></td>
  <td class="under">{under}<span class="pct">({p_under}%)</span></td>
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
                    rows_html += ROW_TEMPLATE.format(
                        line=line,
                        over=vals["over"],
                        under=vals["under"],
                        p_over=round(vals["p_over"] * 100, 1),
                        p_under=round(vals["p_under"] * 100, 1),
                    )
                sa   = m.get("style_a", "?")
                sb   = m.get("style_b", "?")
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
                    stats_section=_render_stats(m),
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

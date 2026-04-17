"""
Bookmaker odds scraper — shuffle.vip (Polish eFootball / Valhalla Cup page).

Scrapes the tournament listing and, for every visible match, returns the
player pair plus the full totals (Over/Under) grid and 1X2 odds offered by
the bookmaker. Output is later matched with our model predictions so the
best-bet selector can restrict itself to *real* lines and look for value.

Returned structure per match:
    {
        "player1":     "LUCAS",
        "player2":     "HOLIS",
        "date":        "14/04/2026 20:30",
        "totals": {
            "3.5": {"over": 1.12, "under": 5.90},
            "4.5": {"over": 1.35, "under": 3.10},
            ...
        },
        "match_winner": {"1": 1.55, "2": 2.45},   # (X rarely offered in eFIFA)
    }

Notes:
 * shuffle.vip uses a React SPA.  Most of the DOM is rendered client-side, so
   we rely on Playwright to wait for hydration and for odd cells to appear.
 * The totals grid is typically hidden behind a per-match "Więcej rynków"
   (More markets) button.  We click every match card in turn to expand it
   and read its detail pane.
 * The site renames/updates selectors often.  The parser is deliberately
   text-based so it keeps working when classnames change.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

BOOKMAKER_BASE = "https://shuffle.vip/pl/sports/efootball/efootball-international/"
# Legacy default — kept so callers that pass no url still work. The scraper
# will attempt to discover the *current* Valhalla Cup week at runtime by
# crawling the parent listing page above; this constant is only the last
# resort when discovery fails.
BOOKMAKER_URL = BOOKMAKER_BASE + "13012-valhalla-cup-2026-week-16"

# Totals lines our model supports. Anything outside this set is ignored.
_ALLOWED_LINES = {3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5}

# Regex patterns used everywhere in the parser
_LINE_RE  = re.compile(r"(?<!\d)(\d{1,2}\.5)(?!\d)")          # "5.5" / "10.5"
_ODDS_RE  = re.compile(r"(?<!\d)(\d{1,2}\.\d{2})(?!\d)")      # "1.85"
_INT_RE   = re.compile(r"^\d+$")
_DATE_RE  = re.compile(
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s+\d{1,2}:\d{2})?"
    r"|\d{1,2}:\d{2}"
)

# Polish bookmaker labels we care about
_OVER_LABELS  = {"over", "pow", "pow.", "powyżej", "powyzej"}
_UNDER_LABELS = {"under", "pon", "pon.", "poniżej", "ponizej"}
_TOTAL_LABELS = {
    "łącznie", "lacznie", "łączna liczba goli", "totals", "total",
    "totale", "total goals", "powyżej/poniżej", "powyzej/ponizej",
    "over/under", "liczba goli", "suma goli",
}

_JUNK = {
    "vs", "v", "1", "x", "2",
    "pow.", "pon.", "over", "under",
}


# ═════════════════════════════════════════════════════════════════════════════
# text cleaning helpers
# ═════════════════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_player_name(token: str) -> bool:
    """Accept alphabetic names (ASCII or diacritics), 2-30 chars."""
    if not token or len(token) < 2 or len(token) > 30:
        return False
    if _INT_RE.match(token):
        return False
    if _ODDS_RE.fullmatch(token):
        return False
    if _LINE_RE.fullmatch(token):
        return False
    if token.lower() in _JUNK:
        return False
    # Allow letters, digits, spaces, dots, apostrophes, hyphens
    return bool(re.fullmatch(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9 .'\-_]*", token))


def _odd_to_float(tok: str) -> float | None:
    try:
        v = float(tok)
    except (TypeError, ValueError):
        return None
    # Decimal odds are always > 1.00 and realistically below 50
    if 1.01 <= v <= 50.0:
        return v
    return None


# ═════════════════════════════════════════════════════════════════════════════
# extraction JS
# ═════════════════════════════════════════════════════════════════════════════
#
# Strategy:
#   1. For every DOM element, collect the list of its leaf text nodes (in
#      visual order).  Keep elements whose leaf list contains a " VS " / " v "
#      separator OR two player-like tokens surrounding totals odds.  These
#      are our match cards.
#   2. After clicking "expand" on a card we re-run extraction to grab the
#      fully populated totals grid.
#
# We let Python do the structural parsing so the JS stays minimal.

_COLLECT_JS = r"""
() => {
    const out = [];
    const isText = n => n.nodeType === 3;

    function leafTexts(el) {
        const arr = [];
        const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
        let n;
        while ((n = w.nextNode())) {
            const t = n.textContent.replace(/\s+/g, ' ').trim();
            if (t) arr.push(t);
        }
        return arr;
    }

    // Find every element that has text "VS" or " v " as an exact leaf
    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    const vsNodes = [];
    let n;
    while ((n = tw.nextNode())) {
        const t = n.textContent.trim().toUpperCase();
        if (t === 'VS' || t === 'V' || t === '-') vsNodes.push(n);
    }

    const seen = new Set();
    for (const v of vsNodes) {
        let el = v.parentElement;
        for (let depth = 0; depth < 10 && el && el !== document.body; depth++) {
            const texts = leafTexts(el);
            // Heuristic: a match card contains 4-80 leaf tokens, at least two
            // decimal odds, and the "VS" separator.
            if (texts.length >= 4 && texts.length <= 120) {
                const odds = texts.filter(x => /^\d{1,2}\.\d{2}$/.test(x));
                if (odds.length >= 2) {
                    const key = texts.slice(0, 6).join('|') + '@' + texts.length;
                    if (!seen.has(key)) {
                        seen.add(key);
                        out.push(texts);
                    }
                    break;
                }
            }
            el = el.parentElement;
        }
    }
    return out;
}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Parsing a single card's leaf list
# ═════════════════════════════════════════════════════════════════════════════

def _split_players(tokens: list[str]) -> tuple[str | None, str | None]:
    """
    Find the VS / v separator and return (player1, player2).

    Typical layouts seen on shuffle-style books:
        ["LUCAS", "Barca", "VS", "HOLIS", "Madrid", "…odds…"]
        ["Lucas", "-", "Holis", …]
    """
    sep_idx = None
    for i, t in enumerate(tokens):
        if t.strip().upper() in ("VS", "V"):
            sep_idx = i
            break
    if sep_idx is None:
        # sometimes the layout is "P1 - P2"
        for i, t in enumerate(tokens):
            if t.strip() == "-" and 0 < i < len(tokens) - 1:
                if _is_player_name(tokens[i - 1]) and _is_player_name(tokens[i + 1]):
                    sep_idx = i
                    break
    if sep_idx is None:
        return None, None

    before = [t for t in tokens[:sep_idx] if _is_player_name(t)]
    after  = [t for t in tokens[sep_idx + 1:] if _is_player_name(t)]
    if not before or not after:
        return None, None

    # Pick the last before and first after.  Drop "team" labels by preferring
    # the longest alphabetic slug on each side.
    def _pick(cands: list[str], side: str) -> str:
        # Strip tokens that look like a date
        cands = [c for c in cands if not _DATE_RE.search(c)]
        if not cands:
            return ""
        # Names are usually the first (after VS) or the last (before VS) token
        return cands[-1] if side == "before" else cands[0]

    return _pick(before, "before"), _pick(after, "after")


def _extract_totals(tokens: list[str]) -> dict[str, dict[str, float]]:
    """
    Walk through the cleaned token list and pair totals lines with their
    over/under decimal odds.

    Accepts any of the common layouts:
        [..., "Over", "5.5", "1.85", "Under", "5.5", "1.95", ...]
        [..., "Pow. 5.5", "1.85", "Pon. 5.5", "1.95", ...]
        [..., "5.5", "1.85", "1.95", ...]        (line | over | under)
    """
    totals: dict[str, dict[str, float]] = {}

    # Normalise tokens; keep numeric tokens intact
    toks = [t.strip() for t in tokens]

    i = 0
    while i < len(toks):
        tok = toks[i]
        low = tok.lower()

        # ── Style A:  "Over 5.5"/"Under 5.5" followed by an odd ─────────────
        #  label + line (can be merged as "Pow. 5.5") + odd
        merged = re.match(
            r"^(pow\.?|pon\.?|over|under|powyżej|poniżej|powyzej|ponizej)\s*(\d{1,2}\.5)$",
            tok,
            re.IGNORECASE,
        )
        if merged:
            side_word = merged.group(1).lower()
            line      = merged.group(2)
            side      = "over" if side_word.startswith(("pow", "ov", "powy")) else "under"
            odd = _find_next_odd(toks, i + 1)
            if odd and float(line) in _ALLOWED_LINES:
                totals.setdefault(line, {})[side] = odd
            i += 1
            continue

        if low in _OVER_LABELS or low in _UNDER_LABELS:
            side = "over" if low in _OVER_LABELS else "under"
            # Next line token
            line = _find_next_line(toks, i + 1, max_ahead=3)
            odd  = _find_next_odd(toks, i + 1, max_ahead=5)
            if line and odd and float(line) in _ALLOWED_LINES:
                totals.setdefault(line, {})[side] = odd
            i += 1
            continue

        # ── Style B: line then two odds (over, under)
        if _LINE_RE.fullmatch(tok) and float(tok) in _ALLOWED_LINES:
            o1 = _find_next_odd(toks, i + 1, max_ahead=2)
            o2 = _find_next_odd(toks, i + 2, max_ahead=3) if o1 else None
            if o1 and o2 and o1 != o2:
                line_str = tok
                entry = totals.setdefault(line_str, {})
                # We cannot yet tell which is over vs under; infer from
                # magnitudes — when a pair offers line L, under<over is
                # impossible for low L (≤ μ) and over<under is impossible
                # for high L (≥ μ).  Default: first=over when line ≤ 6.5,
                # else first=under.  Corrected later by cross-checking.
                if float(line_str) <= 6.5:
                    entry.setdefault("over", o1)
                    entry.setdefault("under", o2)
                else:
                    entry.setdefault("under", o1)
                    entry.setdefault("over", o2)
            i += 1
            continue

        i += 1

    # Drop malformed entries (need both over & under)
    return {k: v for k, v in totals.items() if "over" in v and "under" in v}


def _find_next_line(tokens: list[str], start: int, max_ahead: int = 3) -> str | None:
    for j in range(start, min(start + max_ahead, len(tokens))):
        m = _LINE_RE.fullmatch(tokens[j])
        if m:
            return m.group(1)
    return None


def _find_next_odd(tokens: list[str], start: int, max_ahead: int = 3) -> float | None:
    for j in range(start, min(start + max_ahead, len(tokens))):
        val = _odd_to_float(tokens[j])
        if val is not None:
            return val
    return None


def _extract_1x2(tokens: list[str]) -> dict[str, float]:
    """
    Pick out the 1X2 odds that appear BEFORE any totals/line token. Shuffle.vip
    lists the match winner first and the totals grid below it, so we can stop
    scanning as soon as we hit a totals line (e.g. "5.5" or "Pow. 3.5") or an
    explicit totals header.
    """
    out: dict[str, float] = {}

    cutoff = len(tokens)
    for i, t in enumerate(tokens):
        low = t.lower()
        if _LINE_RE.fullmatch(t):
            cutoff = i
            break
        if any(label in low for label in _TOTAL_LABELS):
            cutoff = i
            break
        if low in _OVER_LABELS or low in _UNDER_LABELS:
            cutoff = i
            break

    head = tokens[:cutoff]
    odds = [v for v in (_odd_to_float(t) for t in head) if v is not None]
    if len(odds) < 2:
        return out

    out["1"] = odds[0]
    if len(odds) >= 3 and 2.5 <= odds[1] <= 10.0:
        # Three-way only if middle odd realistic for a draw
        out["X"] = odds[1]
        out["2"] = odds[2]
    else:
        out["2"] = odds[1]
    return out


def _parse_card(tokens: list[str]) -> dict[str, Any] | None:
    p1, p2 = _split_players(tokens)
    if not p1 or not p2:
        return None

    date = next((t for t in tokens if _DATE_RE.search(t)), "")

    return {
        "player1":      p1,
        "player2":      p2,
        "date":         date,
        "totals":       _extract_totals(tokens),
        "match_winner": _extract_1x2(tokens),
        "_raw":         tokens[:60],  # kept for diagnostics; truncated
    }


# ═════════════════════════════════════════════════════════════════════════════
# Playwright driver
# ═════════════════════════════════════════════════════════════════════════════

# JS: on the parent eFootball listing page, find every anchor whose href
# matches a Valhalla Cup week. Newer weeks usually appear first; we keep
# them all so we can try the most recent one first.
_DISCOVER_JS = r"""
() => {
    const out = [];
    const seen = new Set();
    const as = Array.from(document.querySelectorAll('a[href]'));
    for (const a of as) {
        const h = a.getAttribute('href') || '';
        // accept absolute or relative hrefs pointing at Valhalla Cup weeks
        if (/valhalla-cup(-\d{4})?-week-\d+/i.test(h)) {
            const full = h.startsWith('http') ? h : (location.origin + h);
            if (!seen.has(full)) { seen.add(full); out.push(full); }
        }
    }
    return out;
}
"""


async def _discover_valhalla_urls(page) -> list[str]:
    """
    Crawl the eFootball-international landing page and return every Valhalla
    Cup week URL we can find, newest week first. Falls back to an empty list
    on any failure so the caller can use the legacy constant.
    """
    try:
        logger.info("Bookmaker: discovering current week → %s", BOOKMAKER_BASE)
        await page.goto(BOOKMAKER_BASE, timeout=40_000)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2_000)
        # Trigger lazy listings
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1_000)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)
        urls = await page.evaluate(_DISCOVER_JS)
    except Exception as exc:
        logger.warning("Bookmaker: discovery failed: %s", exc)
        return []

    if not urls:
        return []

    # Sort newest week first (highest week-N number wins)
    def _week_num(u: str) -> int:
        m = re.search(r"week-(\d+)", u)
        return int(m.group(1)) if m else 0

    urls = sorted(set(urls), key=_week_num, reverse=True)
    logger.info("Bookmaker: discovered %d Valhalla URL(s): %s",
                len(urls), [u.rsplit("/", 1)[-1] for u in urls[:5]])
    return urls


async def _collect_cards(page, url: str) -> list[list[str]]:
    """Visit one bookmaker page and return the raw leaf-token arrays per card."""
    logger.info("Bookmaker: navigating to %s", url)
    await page.goto(url, timeout=40_000)
    try:
        await page.wait_for_load_state("networkidle")
    except Exception:
        pass

    try:
        await page.wait_for_selector("body *", timeout=15_000)
    except Exception:
        pass
    await page.wait_for_timeout(3_000)

    # Scroll to trigger lazy-loaded match cards
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(1_500)
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(1_000)

    # Expand "Więcej rynków" / More markets buttons so totals grid appears.
    try:
        expanded = await page.evaluate(
            r"""
            () => {
                let n = 0;
                const btns = Array.from(document.querySelectorAll(
                    'button, [role="button"], [class*="expand"], [class*="toggle"]'
                ));
                for (const b of btns) {
                    const t = (b.textContent || '').toLowerCase();
                    if (t.includes('więcej') || t.includes('wiecej')
                        || t.includes('more') || t.includes('markets')) {
                        try { b.click(); n++; } catch(e) {}
                    }
                }
                return n;
            }
            """
        )
        logger.info("Bookmaker: expanded %d 'more markets' buttons", expanded)
    except Exception as exc:
        logger.debug("Bookmaker: expand step skipped: %s", exc)

    await page.wait_for_timeout(1_500)

    try:
        html = await page.content()
        Path("debug_bookmaker.html").write_text(html, encoding="utf-8")
        logger.info("Bookmaker: saved debug_bookmaker.html (%d chars)", len(html))
    except Exception:
        pass

    try:
        cards = await page.evaluate(_COLLECT_JS)
    except Exception as exc:
        logger.error("Bookmaker: _COLLECT_JS failed: %s", exc)
        return []

    logger.info("Bookmaker: %d candidate cards collected on %s",
                len(cards), url.rsplit("/", 1)[-1])
    return cards


async def scrape_bookmaker_odds(
    url: str | None = None,
) -> list[dict[str, Any]]:
    """
    Main entry point — returns a list of match/odds dicts. Never raises;
    failures degrade to an empty list so callers can fall back to the
    model-only flow without breaking the cycle.

    Strategy:
      1. If `url` is supplied, try it first.
      2. Otherwise, crawl the parent eFootball-international page and try
         each discovered Valhalla-Cup-week link, newest week first, until
         we find one with enough match cards to be useful.
      3. As a last resort, fall back to the hard-coded `BOOKMAKER_URL`.
    """
    matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", "--disable-dev-shm-usage",
                "--disable-gpu", "--single-process",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="pl-PL",
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = await context.new_page()

        try:
            # Build ordered URL list: explicit > discovered > legacy constant
            candidates: list[str] = []
            if url:
                candidates.append(url)
            else:
                discovered = await _discover_valhalla_urls(page)
                candidates.extend(discovered)
                if BOOKMAKER_URL not in candidates:
                    candidates.append(BOOKMAKER_URL)

            cards: list[list[str]] = []
            chosen_url = ""
            for candidate in candidates:
                try:
                    cards = await _collect_cards(page, candidate)
                except Exception as exc:
                    logger.warning("Bookmaker: %s failed: %s", candidate, exc)
                    cards = []
                if cards:
                    chosen_url = candidate
                    break
                logger.info("Bookmaker: 0 cards on %s — trying next candidate", candidate)

            if not cards:
                logger.warning("Bookmaker: no candidates returned cards (%d tried)",
                               len(candidates))
                return matches

            logger.info("Bookmaker: using %s (%d candidate cards)",
                        chosen_url.rsplit("/", 1)[-1], len(cards))

            seen_pairs: set[tuple[str, str]] = set()

            for idx, tokens in enumerate(cards):
                try:
                    parsed = _parse_card(tokens)
                    if not parsed:
                        # Diagnostic: show a few sample token lists so we can
                        # see why the parser rejected them.
                        if idx < 3:
                            logger.info(
                                "Bookmaker: card[%d] NOT parsed — tokens[:15]=%s",
                                idx, tokens[:15],
                            )
                        continue
                    key = (parsed["player1"].upper(), parsed["player2"].upper())
                    rev = (parsed["player2"].upper(), parsed["player1"].upper())
                    if key in seen_pairs or rev in seen_pairs:
                        continue

                    # Reject cards with no usable totals and no 1x2 — these
                    # are navigation widgets that slipped through.
                    if not parsed["totals"] and len(parsed["match_winner"]) < 2:
                        if idx < 3:
                            logger.info(
                                "Bookmaker: card[%d] %s vs %s has no odds — tokens[:20]=%s",
                                idx, parsed["player1"], parsed["player2"], tokens[:20],
                            )
                        continue

                    seen_pairs.add(key)
                    matches.append(parsed)
                    logger.info(
                        "  + %s vs %s | totals=%s | 1x2=%s",
                        parsed["player1"], parsed["player2"],
                        sorted(parsed["totals"].keys()),
                        parsed["match_winner"],
                    )
                except Exception as exc:
                    logger.debug("Bookmaker: card %d parse error: %s", idx, exc)

        except Exception as exc:
            logger.error("Bookmaker scraper error: %s", exc)
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass

    logger.info("Bookmaker done: %d matches with odds", len(matches))
    return matches

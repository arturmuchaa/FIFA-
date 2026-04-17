"""
Bookmaker odds scraper — shuffle.vip (Polish eFootball listing).

Flow
────
1. Land on the eFootball "upcoming" listing:
       https://shuffle.vip/pl/sports?section=upcoming&sport=EFOOTBALL
2. Find every card whose header contains "Valhalla Cup". Each card shows:
     * player pair (may appear as "Team (Player)", e.g. "Palmeiras (Ronin)")
     * 1X2 odds (with Remis = draw)
     * a "+N" link to the per-match detail page
3. Open every match detail page in sequence. On the detail page the
   totals market has a **Powyżej** (Over) section followed by a
   **Poniżej** (Under) section, each listing (line, odds) rows. Lines
   may be .0 / .25 / .5 / .75 — we filter to the half-integer lines the
   model actually prices.
4. Return a list of structured dicts the matcher can align with our
   drafted.gg upcoming matches.

Polish quirks handled:
  - Decimal separator is a comma on shuffle.vip ("1,58" instead of "1.58").
  - "Powyżej" / "Poniżej" section headers (with/without diacritics).
  - Player name is usually embedded as "Team (Player)" on the listing but
    plain on the detail page — we extract the parenthesised player when it
    exists, otherwise we use the first non-league token.

Returned structure per match:
    {
        "player1":     "RONIN",
        "player2":     "HUNTER",
        "date":        "17/04/2026 04:08",
        "totals": {
            "3.5": {"over": 1.12, "under": 5.90},
            "4.5": {"over": 1.35, "under": 3.10},
            ...
        },
        "match_winner": {"1": 1.17, "X": 7.00, "2": 7.50},
    }
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

LISTING_URL = "https://shuffle.vip/pl/sports?section=upcoming&sport=EFOOTBALL"

# Legacy tournament URL — retained only as a last-ditch fallback when the
# listing page fails to render any Valhalla cards.
BOOKMAKER_BASE = "https://shuffle.vip/pl/sports/efootball/efootball-international/"
BOOKMAKER_URL  = BOOKMAKER_BASE + "13012-valhalla-cup-2026-week-16"

# Totals lines our model supports. Anything outside this set is ignored.
_ALLOWED_LINES = {3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5}

# Labels we treat as "over" or "under" section headers (accent-insensitive)
_OVER_LABELS  = {"over", "pow", "pow.", "powyzej"}
_UNDER_LABELS = {"under", "pon", "pon.", "ponizej"}


# ═════════════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════════════

def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _parse_num(tok: str) -> float | None:
    """Parse '1,58' / '1.58' / '5.25' / '10.5' → float, else None."""
    if not tok:
        return None
    try:
        return float(tok.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _parse_odds(tok: str) -> float | None:
    """Same as _parse_num but clamp to the realistic decimal-odds range."""
    v = _parse_num(tok)
    if v is None:
        return None
    if 1.01 <= v <= 200.0:
        return v
    return None


def _player_from_label(label: str) -> str | None:
    """
    Extract the player name from a shuffle.vip team label.

    'Palmeiras (Ronin)'   → 'Ronin'
    'Real Madrid (Lucas)' → 'Lucas'
    'Lucas'               → 'Lucas'
    'Remis'               → None  (draw row)
    """
    if not label:
        return None
    s = label.strip()
    if not s or s.lower() in ("remis", "draw", "x"):
        return None
    m = re.search(r"\(([^)]+)\)\s*$", s)
    if m:
        name = m.group(1).strip()
        if 1 < len(name) <= 30:
            return name
    # No parenthesised player — take the whole label if it's short-ish.
    if 1 < len(s) <= 30:
        return s
    return None


# ═════════════════════════════════════════════════════════════════════════════
# JS: collect Valhalla Cup cards from the upcoming eFootball listing
# ═════════════════════════════════════════════════════════════════════════════

_LIST_JS = r"""
() => {
    // Return one entry per Valhalla Cup listing card.
    //   {hrefs: [{href, text, depth}], tokens}
    //
    // We locate every text node mentioning "Valhalla" and walk up to the
    // smallest ancestor that also contains a clickable anchor. That element
    // is the card; we harvest its entire text tree in visual order AND
    // return *every* anchor in the card so Python can pick the match URL
    // (not the tournament chip, which shares the same DOM).

    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const vsNodes = [];
    let n;
    while ((n = walker.nextNode())) {
        const t = (n.textContent || '').trim();
        if (/Valhalla\s*Cup/i.test(t)) vsNodes.push(n);
    }

    const out = [];
    const seen = new Set();
    for (const node of vsNodes) {
        // Walk up to the first ancestor large enough to be a card and
        // containing at least one anchor.
        let el = node.parentElement;
        let cardEl = null;
        for (let i = 0; i < 15 && el && el !== document.body; i++) {
            const anchors = el.querySelectorAll('a[href]');
            const rect = el.getBoundingClientRect();
            if (anchors.length >= 1 && rect.height > 90 && rect.height < 900) {
                cardEl = el;
                break;
            }
            el = el.parentElement;
        }
        if (!cardEl) continue;

        // Collect every anchor inside the card
        const rawAnchors = Array.from(cardEl.querySelectorAll('a[href]'));
        const hrefs = [];
        for (const a of rawAnchors) {
            const href = a.getAttribute('href') || '';
            if (!href || href === '#' || href.startsWith('javascript:')) continue;
            const full = href.startsWith('http') ? href : (location.origin + href);
            // Path depth = number of non-empty segments
            let path = '';
            try { path = new URL(full).pathname; } catch(e) { path = href; }
            const depth = path.split('/').filter(s => s.length > 0).length;
            const text = (a.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 60);
            hrefs.push({href: full, text: text, depth: depth, path: path});
        }
        if (hrefs.length === 0) continue;

        // Walk the element tree harvesting text leaves in order.
        const w2 = document.createTreeWalker(cardEl, NodeFilter.SHOW_TEXT);
        const tokens = [];
        let lf;
        while ((lf = w2.nextNode())) {
            const t = lf.textContent.replace(/\s+/g, ' ').trim();
            if (t) tokens.push(t);
        }
        if (tokens.length < 4) continue;

        const key = hrefs[0].href + '|' + tokens.slice(0, 8).join('|');
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({hrefs: hrefs, tokens: tokens});
    }
    return out;
}
"""


def _pick_match_href(hrefs: list[dict[str, Any]]) -> str | None:
    """
    Choose the anchor most likely to open the per-match detail page.

    shuffle.vip cards contain at least two links:
      - the "Valhalla Cup" chip  → tournament page (short path)
      - the player-row / "+N"    → match detail    (longer path)

    We pick the anchor with the deepest path. If the top candidate looks
    like the tournament base (ends in "-valhalla-cup-YYYY-week-N" with no
    further segment), we fall back to the next deepest.
    """
    if not hrefs:
        return None
    # Normalize & rank by depth, then by path length as a tiebreaker
    ranked = sorted(
        hrefs,
        key=lambda a: (a.get("depth") or 0, len(a.get("path") or "")),
        reverse=True,
    )
    for cand in ranked:
        path = (cand.get("path") or "").rstrip("/")
        # Skip obvious tournament base URLs
        if re.search(r"valhalla-cup-\d{4}-week-\d+$", path, re.I):
            continue
        if cand.get("href"):
            return cand["href"]
    # Nothing non-tournament? Take the deepest URL we saw anyway.
    return ranked[0].get("href") if ranked else None


# ═════════════════════════════════════════════════════════════════════════════
# JS: extract Powyżej / Poniżej grid from a match detail page
# ═════════════════════════════════════════════════════════════════════════════
#
# The detail page stacks two columns side-by-side (Over / Under) with a
# header per side and one row per line. We return the full leaf text in
# order so Python can pair (line, odds) under the right section header.

_DETAIL_JS = r"""
() => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const tokens = [];
    let n;
    while ((n = walker.nextNode())) {
        const t = n.textContent.replace(/\s+/g, ' ').trim();
        if (t) tokens.push(t);
    }
    return tokens;
}
"""


# ═════════════════════════════════════════════════════════════════════════════
# parsing
# ═════════════════════════════════════════════════════════════════════════════

def _split_players_from_listing(tokens: list[str]) -> tuple[str | None, str | None]:
    """
    Find two 'team (player)' labels on the listing card.

    Typical ordering (top → bottom visually):
      ['W 2m (Valhalla Cup 3 2026 Week #16)',
       'Palmeiras (Ronin)',
       'River Plate (Hunter)',
       'Palmeiras (R...',  '1,17',
       'Remis',             '7,00',
       'River Plate (H...', '7,50',
       '+36']

    The first two tokens matching "X (Y)" are the full pair. The shorter
    truncated "X (Y..." entries in the odds-column repeat the team but
    are cut off — we treat them as secondary confirmation.
    """
    candidates: list[str] = []
    for t in tokens:
        # Must contain a parenthesised short name, OR be a plain short name
        # (no digits-only, no odds, no "+N" counters, no flags)
        if t in ("Remis", "Draw", "X"):
            continue
        if re.fullmatch(r"\+\d+", t):
            continue
        if _parse_odds(t) is not None:
            continue
        if re.search(r"Valhalla|Cup|Week|#\d+|W\s*\d+m|\d+\s*m\b", t, re.I):
            continue
        # A "Team (Player)" token is our primary target
        if re.search(r"\([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'\-]{1,28}\)\s*$", t):
            candidates.append(t)
            continue
        # Fallback: a plain short alphabetical token
        if 2 <= len(t) <= 30 and re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'\-_]*", t):
            # Avoid section labels
            if _strip_accents(t).lower() in _OVER_LABELS | _UNDER_LABELS:
                continue
            candidates.append(t)

    # Dedupe while preserving order, then pick the first two distinct entries
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    if len(uniq) < 2:
        return None, None

    p1 = _player_from_label(uniq[0])
    p2 = _player_from_label(uniq[1])
    return p1, p2


def _extract_1x2_from_listing(tokens: list[str]) -> dict[str, float]:
    """
    On listing cards the 1X2 odds appear directly after player labels, e.g.
      [..., 'Palmeiras (R...', '1,17', 'Remis', '7,00', 'River Plate (H...', '7,50']
    We find the 'Remis' marker and pull the odd before/after it.
    """
    out: dict[str, float] = {}
    remis_idx = next(
        (i for i, t in enumerate(tokens) if t.strip().lower() in ("remis", "draw")),
        None,
    )
    if remis_idx is None:
        return out

    odds: list[float] = []
    for t in tokens[max(0, remis_idx - 4): remis_idx + 6]:
        v = _parse_odds(t)
        if v is not None:
            odds.append(v)
    if len(odds) >= 3:
        out["1"] = odds[0]
        out["X"] = odds[1]
        out["2"] = odds[2]
    elif len(odds) == 2:
        out["1"] = odds[0]
        out["2"] = odds[1]
    return out


def _extract_date(tokens: list[str]) -> str:
    """Best-effort date string from a token list."""
    for t in tokens:
        m = re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", t)
        if m:
            return m.group(0)
        m = re.search(r"\d{1,2}:\d{2}", t)
        if m:
            return m.group(0)
    return ""


def _extract_totals_from_detail(tokens: list[str]) -> dict[str, dict[str, float]]:
    """
    Walk detail-page tokens collecting (line, odds) rows under Powyżej / Poniżej.

    We stay permissive: any section-header word ('powyzej', 'ponizej', 'over',
    'under') flips the mode, and the next `(numeric-line, decimal-odds)` pair
    we see is stored accordingly. The detail page often lists quarter-integer
    lines (5.0, 5.25, 5.5, 5.75, 6.0) — we only keep half-integer lines in
    `_ALLOWED_LINES` since that's what our model prices.
    """
    out: dict[str, dict[str, float]] = {}
    mode: str | None = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = _strip_accents(tok).strip().lower()

        # Section headers
        if low in _OVER_LABELS:
            mode = "over"
            i += 1
            continue
        if low in _UNDER_LABELS:
            mode = "under"
            i += 1
            continue

        # (line, odds) pair under the current section
        if mode is not None:
            line = _parse_num(tok)
            if line is not None and 1.0 <= line <= 20.0:
                # Skip quarter lines our model doesn't score
                if line in _ALLOWED_LINES:
                    odd = None
                    # The odds cell is usually the very next leaf, but some
                    # layouts inject a padding token — scan up to 3 ahead.
                    for j in range(i + 1, min(i + 4, len(tokens))):
                        cand = _parse_odds(tokens[j])
                        if cand is not None:
                            odd = cand
                            i = j  # advance past the consumed odds cell
                            break
                    if odd is not None:
                        key = f"{line}"
                        entry = out.setdefault(key, {})
                        entry[mode] = odd
        i += 1

    # Keep only lines with both sides captured
    return {k: v for k, v in out.items() if "over" in v and "under" in v}


# ═════════════════════════════════════════════════════════════════════════════
# Playwright driver
# ═════════════════════════════════════════════════════════════════════════════

async def _goto(page, url: str, timeout: int = 40_000) -> bool:
    try:
        await page.goto(url, timeout=timeout)
        try:
            await page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception:
            pass
        try:
            await page.wait_for_selector("body *", timeout=10_000)
        except Exception:
            pass
        await page.wait_for_timeout(2_500)
        # Trigger lazy content
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1_000)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)
        return True
    except Exception as exc:
        logger.warning("Bookmaker: navigation to %s failed: %s", url, exc)
        return False


async def _expand_more_markets(page) -> int:
    """Click every "Więcej rynków" / "More markets" button on the page."""
    try:
        return await page.evaluate(
            r"""
            () => {
                let n = 0;
                const btns = Array.from(document.querySelectorAll(
                    'button, [role="button"], a, [class*="expand"], [class*="toggle"]'
                ));
                for (const b of btns) {
                    const t = (b.textContent || '').toLowerCase();
                    if (t.includes('więcej') || t.includes('wiecej')
                        || t.includes('more') || t.includes('markets')
                        || t.includes('rynków') || t.includes('rynkow')
                        || t.includes('suma goli') || t.includes('łącznie')
                        || t.includes('total')) {
                        try { b.click(); n++; } catch(e) {}
                    }
                }
                return n;
            }
            """
        )
    except Exception:
        return 0


async def _fetch_detail_totals(
    page, url: str,
) -> dict[str, dict[str, float]]:
    """
    Visit a per-match detail URL and extract the totals grid. Silent failure
    returns an empty dict so the caller can still keep the listing-card data.
    """
    if not await _goto(page, url, timeout=35_000):
        return {}

    await _expand_more_markets(page)
    await page.wait_for_timeout(1_500)

    try:
        tokens: list[str] = await page.evaluate(_DETAIL_JS)
    except Exception as exc:
        logger.debug("Bookmaker: detail JS failed for %s: %s", url, exc)
        return {}

    totals = _extract_totals_from_detail(tokens)

    if not totals:
        # One more try after scrolling further down and re-expanding
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1_500)
            await _expand_more_markets(page)
            await page.wait_for_timeout(1_500)
            tokens = await page.evaluate(_DETAIL_JS)
            totals = _extract_totals_from_detail(tokens)
        except Exception:
            pass

    if not totals:
        logger.info(
            "Bookmaker: detail %s → no totals grid found (tokens[:30]=%s)",
            url.rsplit("/", 1)[-1], tokens[:30] if tokens else [],
        )
    return totals


async def scrape_bookmaker_odds(url: str | None = None) -> list[dict[str, Any]]:
    """
    Main entry point — returns a list of match/odds dicts. Never raises;
    failures degrade to an empty list so callers can fall back to the
    model-only flow without breaking the cycle.

    Strategy:
      1. Open the upcoming eFootball listing (or `url` when provided).
      2. Collect every Valhalla Cup card + its detail-page URL + 1X2.
      3. For each card: visit the detail URL and parse Powyżej/Poniżej.
      4. Return one entry per match with totals + 1X2.
    """
    matches: list[dict[str, Any]] = []
    listing_url = url or LISTING_URL

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
            logger.info("Bookmaker: opening listing %s", listing_url)
            if not await _goto(page, listing_url):
                return matches

            # Save debug HTML for post-run inspection
            try:
                html = await page.content()
                Path("debug_bookmaker.html").write_text(html, encoding="utf-8")
                logger.info("Bookmaker: saved debug_bookmaker.html (%d chars)", len(html))
            except Exception:
                pass

            cards: list[dict[str, Any]] = await page.evaluate(_LIST_JS)
            logger.info("Bookmaker: %d Valhalla card(s) on listing", len(cards))

            # Build intermediate list with listing-scoped data
            seen_pairs: set[tuple[str, str]] = set()
            prepared: list[dict[str, Any]] = []
            for idx, card in enumerate(cards):
                tokens  = card.get("tokens") or []
                hrefs   = card.get("hrefs")  or []
                # Back-compat: old cards may still return a scalar "href"
                if not hrefs and card.get("href"):
                    hrefs = [{"href": card["href"], "depth": 99, "path": ""}]
                href = _pick_match_href(hrefs)
                if not href:
                    continue

                if idx < 3:
                    logger.info(
                        "Bookmaker: listing card[%d] anchors=%s → picked %s",
                        idx,
                        [h.get("path") for h in hrefs[:5]],
                        href.rsplit("/", 2)[-2:] if href else None,
                    )

                p1, p2 = _split_players_from_listing(tokens)
                if not p1 or not p2:
                    if idx < 3:
                        logger.info(
                            "Bookmaker: listing card[%d] no players — tokens[:15]=%s",
                            idx, tokens[:15],
                        )
                    continue

                key = (p1.upper(), p2.upper())
                rev = (p2.upper(), p1.upper())
                if key in seen_pairs or rev in seen_pairs:
                    continue
                seen_pairs.add(key)

                prepared.append({
                    "player1":      p1,
                    "player2":      p2,
                    "date":         _extract_date(tokens),
                    "href":         href,
                    "match_winner": _extract_1x2_from_listing(tokens),
                    "totals":       {},
                })

            logger.info("Bookmaker: %d unique Valhalla matches queued for detail crawl", len(prepared))

            # Drill into each detail page to grab the totals grid
            for entry in prepared:
                try:
                    logger.info(
                        "Bookmaker: detail → %s vs %s (%s)",
                        entry["player1"], entry["player2"],
                        entry["href"].rsplit("/", 1)[-1],
                    )
                    totals = await _fetch_detail_totals(page, entry["href"])
                    entry["totals"] = totals
                    logger.info(
                        "  → %s vs %s | totals=%s | 1x2=%s",
                        entry["player1"], entry["player2"],
                        sorted(totals.keys()), entry["match_winner"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Bookmaker: detail crawl failed for %s vs %s: %s",
                        entry["player1"], entry["player2"], exc,
                    )

            # Keep only entries with either usable totals or 3-way winner odds
            matches = [
                e for e in prepared
                if e["totals"] or len(e["match_winner"]) >= 2
            ]

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

    logger.info(
        "Bookmaker done: %d matches with odds (out of %d cards seen)",
        len(matches), len(matches),
    )
    return matches

import hashlib
import logging
import re
from typing import Any

from playwright.async_api import async_playwright

UPCOMING_URL = "https://drafted.gg/valhalla-cup/upcoming-matches"
RESULTS_URL  = "https://drafted.gg/valhalla-cup/results"

logger = logging.getLogger(__name__)

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_DATE_RE = re.compile(
    rf"\d{{1,2}}\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}"
    rf"|\d{{2}}:\d{{2}}",
    re.IGNORECASE,
)
_JUNK = {"vs", "results", "upcoming matches", "upcoming", "contact",
         "valhalla cup", "head to head", "form", "stats", "home"}

# ── Overlay handling ──────────────────────────────────────────────────────────
#
# The site has a persistent widget overlay: div.fixed.inset-0 (z-60) containing
# an <iframe title="widget">.  Removing it causes React to re-render it.
# Solution: set pointer-events:none on existing overlays so clicks pass through.
# Re-apply before each card click in case React re-rendered the element.
#
# When a card IS clicked, React renders a NEW div.fixed.inset-0 (the stats modal).
# That new element is not affected by pointer-events:none (we didn't touch it yet),
# so its close button etc. are still interactive if needed.
# We select the stats modal specifically via :has-text('Head to head').

_PASSTHROUGH_OVERLAYS_JS = """
() => {
    let n = 0;
    document.querySelectorAll('.fixed.inset-0').forEach(el => {
        el.style.pointerEvents = 'none';
        n++;
    });
    return n;
}
"""

_ESCAPE_JS = """
() => {
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Escape', code: 'Escape', keyCode: 27,
        bubbles: true, cancelable: true
    }));
    document.dispatchEvent(new KeyboardEvent('keyup', {
        key: 'Escape', code: 'Escape', keyCode: 27,
        bubbles: true, cancelable: true
    }));
}
"""

# Stats modal selector (excludes the widget overlay which has no H2H text)
MODAL_SEL         = "div.fixed.inset-0:has-text('Head to head')"
MODAL_CONTENT_SEL = "text=Head to head"


# ── Modal text parser ─────────────────────────────────────────────────────────
#
# Modal layout (from screenshot):
#   Head to head (last 10 direct matches)
#   <player1>          <player2>
#   <player1> wins     <player2> wins
#   <n>    <draws>    <n>
#   Total average goals per match: X.X
#   Form (recent matches, any opponent)
#   L-W-W-...          D-W-W-...
#   (last 2 months, any opponent)
#   XX %    Wins       XX %
#   XX %    Draws      XX %
#   XX %    Losses     XX %
#   X.X     Goals for  X

def _parse_modal(texts: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "h2h":   {},
        "form":  {"player1": [], "player2": []},
        "stats": {"player1": {}, "player2": {}},
    }
    upper = [t.upper() for t in texts]

    # ── H2H wins + avg goals ──────────────────────────────────────────────────
    try:
        idx = next((i for i, t in enumerate(upper) if "HEAD TO HEAD" in t), None)
        if idx is not None:
            # Start from idx+1 to skip the header line itself
            # ("Head to head(last 10 direct matches)" contains "10" which
            # would otherwise always be parsed as wins_player1).
            chunk = " ".join(texts[idx + 1: idx + 20])
            # avg goals: "Total average goals per match: 6.3" or just a float
            avg = re.search(r"average goals per match[:\s]+(\d+\.\d+)", chunk, re.IGNORECASE)
            if avg:
                data["h2h"]["avg_goals_per_match"] = float(avg.group(1))
            else:
                floats = re.findall(r"\b\d+\.\d+\b", chunk)
                if floats:
                    data["h2h"]["avg_goals_per_match"] = float(floats[0])
            # wins: look for standalone integers around "wins" / "draws"
            nums = re.findall(r"\b(\d+)\b", chunk)
            nums = [int(n) for n in nums if int(n) <= 100]
            if len(nums) >= 3:
                data["h2h"]["wins_player1"] = nums[0]
                data["h2h"]["draws"]        = nums[1]
                data["h2h"]["wins_player2"] = nums[2]
            elif len(nums) == 2:
                data["h2h"]["wins_player1"] = nums[0]
                data["h2h"]["wins_player2"] = nums[1]
    except Exception as e:
        logger.debug(f"H2H parse: {e}")

    # ── Form: W/L/D sequences ─────────────────────────────────────────────────
    # Actual iframe layout: each result is on its own line ("W", "L", "D").
    # A percentage string (e.g. "83%") acts as the separator between player1
    # and player2 sequences.
    #
    #   Form
    #   (recent matches, any opponent)
    #   <player2 label>
    #   W                ← player1 result
    #   L
    #   …
    #   83%              ← player1 win-rate → separator
    #   L                ← player2 result
    #   …
    #   8%               ← player2 win-rate
    try:
        fidx = next((i for i, t in enumerate(upper) if t.strip() == "FORM"), None)
        if fidx is not None:
            window      = texts[fidx + 1: fidx + 30]
            window_up   = upper[fidx + 1: fidx + 30]
            pct_pos     = [i for i, t in enumerate(window_up)
                           if re.match(r"^\d+%$", t.strip())]

            def _wld(lines: list[str]) -> list[str]:
                return [t.strip().upper() for t in lines
                        if re.fullmatch(r"[WLD]", t.strip(), re.IGNORECASE)]

            if len(pct_pos) >= 1:
                data["form"]["player1"] = _wld(window[:pct_pos[0]])
            if len(pct_pos) >= 2:
                data["form"]["player2"] = _wld(window[pct_pos[0] + 1: pct_pos[1]])
            elif len(pct_pos) == 1:
                data["form"]["player2"] = _wld(window[pct_pos[0] + 1:])
    except Exception as e:
        logger.debug(f"Form parse: {e}")

    # ── Stats: Player comparison + W/D/L ratio ───────────────────────────────
    # Confirmed frame layout (oddin.gg iframe, see frame dump):
    #
    #   [p1_name]              ← pc_idx - 1
    #   Player comparison      ← pc_idx
    #   (last 2 months…)       ← pc_idx + 1
    #   [p2_name]              ← pc_idx + 2
    #   Goals for              ← pc_idx + 3   \
    #   <value>                ← pc_idx + 4    | player1 block
    #   Goals against                          | (10 label/value pairs
    #   <value>                                |  = 20 lines total)
    #   …                                     /
    #   Goals for              ← pc_idx + 23  \
    #   <value>                ← pc_idx + 24   | player2 block (same layout)
    #   …                                     /
    #
    #   W/D/L ratio            ← wdl_positions[0]
    #   Wins / <pct %> / Draws / <pct %> / Losses / <pct %>
    #   (chart header junk: GA GD GDH1 GDH2 PTS GF 20 40 60 80 100)
    #   W/D/L ratio            ← wdl_positions[1]
    #   Wins / <pct %> / Draws / <pct %> / Losses / <pct %>

    _COMP_LABELS = {
        "GOALS FOR":     "goals_for",
        "GOALS AGAINST": "goals_against",
    }
    _WDL_LABELS = {
        "WINS":   "wins_pct",
        "DRAWS":  "draws_pct",
        "LOSSES": "losses_pct",
    }

    try:
        pc_idx = next(
            (i for i, t in enumerate(upper) if t.strip() == "PLAYER COMPARISON"),
            None,
        )
        if pc_idx is not None:
            p1_block = texts[pc_idx + 3: pc_idx + 23]
            p2_block = texts[pc_idx + 23: pc_idx + 43]

            def _parse_comp(lines: list[str]) -> dict[str, float]:
                out: dict[str, float] = {}
                for j, line in enumerate(lines[:-1]):
                    key = _COMP_LABELS.get(line.upper().strip())
                    if key and key not in out:
                        try:
                            out[key] = float(lines[j + 1].strip())
                        except ValueError:
                            pass
                return out

            data["stats"]["player1"].update(_parse_comp(p1_block))
            data["stats"]["player2"].update(_parse_comp(p2_block))

        wdl_pos = [i for i, t in enumerate(upper) if t.strip() == "W/D/L RATIO"]

        def _parse_wdl(start: int) -> dict[str, float]:
            out: dict[str, float] = {}
            chunk_up  = upper[start: start + 10]
            chunk_raw = texts[start: start + 10]
            for j, t in enumerate(chunk_up[:-1]):
                key = _WDL_LABELS.get(t.strip())
                if key:
                    try:
                        out[key] = float(re.sub(r"[^\d.]", "", chunk_raw[j + 1]))
                    except (ValueError, IndexError):
                        pass
            return out

        if len(wdl_pos) >= 1:
            data["stats"]["player1"].update(_parse_wdl(wdl_pos[0]))
        if len(wdl_pos) >= 2:
            data["stats"]["player2"].update(_parse_wdl(wdl_pos[1]))

    except Exception as e:
        logger.debug(f"Stats parse: {e}")

    return data


# ── Main ──────────────────────────────────────────────────────────────────────

async def scrape_details(url: str = UPCOMING_URL) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-gpu", "--single-process",
                  "--disable-blink-features=AutomationControlled"],
        )
        # Desktop viewport (lg: breakpoint ≥1024px) + real browser user-agent
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        # Remove navigator.webdriver flag (bot detection bypass)
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = await context.new_page()

        try:
            logger.info(f"Details: navigating to {url}")
            await page.goto(url, timeout=30_000)
            await page.wait_for_load_state("networkidle")

            # ── wait for React hydration (SSR → interactive) ─────────────────
            # networkidle doesn't guarantee hydration; cursor-pointer appearing
            # means React has mounted the interactive components
            try:
                await page.wait_for_selector(
                    '[class*="cursor-pointer"]', timeout=10_000
                )
            except Exception:
                pass
            await page.wait_for_timeout(2_000)   # extra buffer for hydration

            # ── wait for skeleton loaders ────────────────────────────────────
            try:
                await page.wait_for_selector(
                    ".animate-skeleton-dark", state="hidden", timeout=10_000
                )
                logger.info("Details: skeleton done")
            except Exception:
                pass

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2_000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1_000)

            # ── make existing overlays click-transparent ─────────────────────
            n = await page.evaluate(_PASSTHROUGH_OVERLAYS_JS)
            logger.info(f"Details: passthrough applied to {n} overlay(s)")

            # ── collect match cards (dedup by player pair) ───────────────────
            all_divs = await page.query_selector_all("div")
            logger.info(f"Details: {len(all_divs)} divs total")

            pair_to_card: dict[tuple, Any] = {}
            for div in all_divs:
                try:
                    text = await div.inner_text()
                    if "VS" not in text.upper():
                        continue
                    if text.upper().count("VS") > 4:
                        continue
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    vs_list = [i for i, l in enumerate(lines) if l.upper() == "VS"]
                    if not vs_list:
                        continue
                    vi = vs_list[0]
                    if vi < 1 or vi >= len(lines) - 1:
                        continue
                    before = [l for l in lines[:vi]
                              if len(l) >= 2
                              and l.lower() not in _JUNK
                              and not _DATE_RE.search(l)
                              and not l.lower().startswith("match ")]
                    after  = [l for l in lines[vi + 1:]
                              if len(l) >= 2
                              and l.lower() not in _JUNK
                              and not _DATE_RE.search(l)
                              and not l.lower().startswith("match ")]
                    if not before or not after:
                        continue
                    # player name is second-to-last before VS (last is team name)
                    p1 = before[-2] if len(before) >= 2 else before[-1]
                    p2 = after[0]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    key = (p1, p2)
                    if key not in pair_to_card or len(lines) < pair_to_card[key][1]:
                        pair_to_card[key] = (div, len(lines))
                except Exception:
                    continue

            match_cards = [(div, p1, p2) for (p1, p2), (div, _) in pair_to_card.items()]
            logger.info(f"Details: {len(match_cards)} unique match cards")

            seen_ids: set[str] = set()

            for idx, (card_div, player1, player2) in enumerate(match_cards):
                match_id = hashlib.md5(f"{player1}_{player2}".encode()).hexdigest()[:12]
                if match_id in seen_ids:
                    continue

                try:
                    # ── re-apply passthrough (React may have re-rendered overlay)
                    await page.evaluate(_PASSTHROUGH_OVERLAYS_JS)

                    await card_div.scroll_into_view_if_needed()
                    await page.wait_for_timeout(300)

                    # ── find element with React onClick handler ──────────────
                    # Walking up by cursor:pointer finds the styled wrapper,
                    # but the actual React onClick may be on a different ancestor.
                    # Inspect __reactProps$ to find the exact element React
                    # registered the onClick on.
                    click_handle = await page.evaluate_handle("""
                        el => {
                            let cur = el;
                            while (cur && cur !== document.body) {
                                const propsKey = Object.keys(cur).find(k =>
                                    k.startsWith('__reactProps$') ||
                                    k.startsWith('__reactEventHandlers$')
                                );
                                if (propsKey) {
                                    const props = cur[propsKey];
                                    if (props && (props.onClick || props.onMouseDown || props.onPointerDown)) {
                                        return cur;
                                    }
                                }
                                cur = cur.parentElement;
                            }
                            // fallback: nearest cursor-pointer ancestor
                            cur = el;
                            while (cur && cur !== document.body) {
                                const cls = cur.getAttribute('class') || '';
                                const st  = window.getComputedStyle(cur);
                                if (cls.includes('cursor-pointer') || st.cursor === 'pointer') return cur;
                                cur = cur.parentElement;
                            }
                            return el;
                        }
                    """, card_div)
                    click_el = click_handle.as_element() or card_div
                    tag = await page.evaluate(
                        "el => el.tagName + ' ' + (el.getAttribute('class') || '').slice(0,80)",
                        click_el
                    )
                    logger.info(f"  [{idx}] click target: {tag}")

                    # Hover first — some React components only attach onClick
                    # after a mouseenter/mouseover event
                    try:
                        await click_el.hover(timeout=3_000)
                        await page.wait_for_timeout(150)
                    except Exception:
                        pass

                    # ── click the card ───────────────────────────────────────
                    # Strategy 1: native Playwright click
                    try:
                        await click_el.click(timeout=4_000)
                    except Exception as ce:
                        logger.debug(f"  [{idx}] native click failed: {ce}")

                    await page.wait_for_timeout(300)

                    # Strategy 2: React onClick — belt-and-suspenders because
                    # the H2H stats live in a cross-origin iframe (stats_in_dom
                    # is always NO), so we can't use DOM presence to gate this.
                    direct = await page.evaluate("""
                        el => {
                            let cur = el;
                            while (cur && cur !== document.body) {
                                const pk = Object.keys(cur).find(k =>
                                    k.startsWith('__reactProps$') ||
                                    k.startsWith('__reactEventHandlers$')
                                );
                                if (pk) {
                                    const p = cur[pk];
                                    if (p && p.onClick) {
                                        try {
                                            p.onClick({
                                                type: 'click', bubbles: true, cancelable: true,
                                                preventDefault: ()=>{}, stopPropagation: ()=>{},
                                                target: cur, currentTarget: cur,
                                                nativeEvent: new MouseEvent('click', {bubbles:true})
                                            });
                                            return 'ok:' + cur.tagName;
                                        } catch(e) { return 'err:' + e.message; }
                                    }
                                }
                                cur = cur.parentElement;
                            }
                            return 'no-handler';
                        }
                    """, card_div)
                    logger.info(f"  [{idx}] clicked {player1} vs {player2} (react={direct})")

                    # ── read H2H stats from iframe ────────────────────────────
                    # The stats widget is served by disir.oddin.gg in a cross-origin
                    # iframe embedded inside the stats modal.  page.wait_for_selector()
                    # only searches the main document, so we iterate page.frames and
                    # read the body text from each non-main frame.
                    await page.wait_for_timeout(4_000)  # allow iframe to update after click

                    modal_texts: list[str] = []
                    for frame in page.frames:
                        if (not frame.url
                                or frame.url == page.url
                                or frame.url.startswith("about:")):
                            continue
                        try:
                            frame_text = await frame.inner_text("body", timeout=5_000)
                            fu = frame_text.upper()
                            has_h2h = "HEAD TO HEAD" in fu
                            has_stats = has_h2h or ("WINS" in fu and "GOALS FOR" in fu)
                            logger.info(
                                f"  [{idx}] frame {frame.url[:70]}: "
                                f"{'HAS STATS' if has_stats else 'no stats'} "
                                f"({len(frame_text)} chars)"
                            )
                            if has_stats:
                                modal_texts = [l.strip() for l in frame_text.splitlines()
                                               if l.strip()]
                                break
                        except Exception as fe:
                            logger.info(f"  [{idx}] frame error {frame.url[:60]}: {fe}")

                    if not modal_texts:
                        logger.info(f"  [{idx}] no stats in any frame for {player1}, skip")
                        continue

                    logger.info(
                        f"  [{idx}] modal: {len(modal_texts)} lines — {modal_texts[:4]}"
                    )

                    if len(modal_texts) < 4:
                        logger.debug(f"  [{idx}] modal too short, skip")
                        continue

                    # Dump the first card's full frame to verify parser mapping
                    if idx == 0:
                        logger.info(
                            f"  [0] frame dump (all {len(modal_texts)} lines):\n"
                            + "\n".join(f"    {i:02d}: {l}" for i, l in enumerate(modal_texts))
                        )

                    modal_data = _parse_modal(modal_texts)
                    seen_ids.add(match_id)
                    results.append({
                        "match_id": match_id,
                        "player1":  player1,
                        "player2":  player2,
                        **modal_data,
                    })
                    logger.info(f"  ✓ {player1} vs {player2} | h2h={modal_data['h2h']}")

                except Exception as exc:
                    logger.warning(f"  [{idx}] error {player1}: {exc}")

                finally:
                    try:
                        await page.evaluate(_ESCAPE_JS)
                    except Exception:
                        pass
                    try:
                        await page.evaluate(_PASSTHROUGH_OVERLAYS_JS)
                    except Exception:
                        pass
                    await page.wait_for_timeout(300)

        except Exception as exc:
            logger.error(f"Details fatal: {exc}")
        finally:
            await context.close()
            await browser.close()

    logger.info(f"Details done: {len(results)} enriched")
    return results

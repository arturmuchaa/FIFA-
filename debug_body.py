"""
Narzędzie diagnostyczne — zapisuje surowy body text obu stron do pliku.

Uruchom na VPS:
  python debug_body.py

Wynik:
  debug_results.txt   ← każda linia ponumerowana
  debug_upcoming.txt

Patrząc na te pliki zobaczysz DOKŁADNĄ strukturę i będziesz wiedział
jakie indeksy zastosować w parserze.
"""

import asyncio
from playwright.async_api import async_playwright

URLS = {
    "debug_results.txt":  "https://drafted.gg/valhalla-cup/results",
    "debug_upcoming.txt": "https://drafted.gg/valhalla-cup/upcoming-matches",
}


async def dump(filename: str, url: str) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=30_000)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(5_000)
        body = await page.inner_text("body")
        await browser.close()

    lines = [l.strip() for l in body.split("\n") if l.strip()]
    with open(filename, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            f.write(f"[{i:03d}] {line}\n")
    print(f"✅ Saved {len(lines)} lines → {filename}")


async def main() -> None:
    for filename, url in URLS.items():
        print(f"Fetching {url}…")
        await dump(filename, url)


asyncio.run(main())

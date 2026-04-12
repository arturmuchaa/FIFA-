# Valhalla Cup — Over/Under Predictor

Kompletny system scrapingu + modelu Poissona + Web UI dla meczów FIFA na drafted.gg.

## Stack

| Komponent | Technologia |
|-----------|-------------|
| Scraping | Python + Playwright async (Chromium headless) |
| Backend | FastAPI + uvicorn |
| Model | Poisson distribution |
| Baza | JSON files (data/) |
| UI | Wbudowany HTML (serwowany przez FastAPI) |

## Struktura

```
FIFA-/
├── services/scraper/
│   ├── results.py     # historia meczów
│   ├── upcoming.py    # nadchodzące mecze
│   └── details.py     # modal click → H2H / Form / Stats
├── core/
│   ├── database.py    # odczyt/zapis JSON + statystyki graczy
│   └── model.py       # model Poissona → kursy over/under
├── api/
│   └── app.py         # FastAPI endpoints + Web UI
├── data/
│   ├── matches.json   # wszystkie mecze
│   └── players.json   # statystyki graczy
├── main.py            # główna pętla (60s) + start serwera
├── install.sh         # jednorazowa instalacja na VPS
└── valhalla.service   # systemd unit file
```

## Szybki start na VPS

```bash
# 1. Sklonuj repo
git clone <repo-url> /opt/valhalla
cd /opt/valhalla

# 2. Instalacja (jako root)
chmod +x install.sh
./install.sh

# 3. Uruchom
source venv/bin/activate
python main.py
```

Otwórz w przeglądarce: `http://VPS_IP:8000`

## Jako usługa systemd

```bash
cp valhalla.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable valhalla
systemctl start valhalla
systemctl status valhalla
```

## Endpointy API

| Endpoint | Opis |
|----------|------|
| `GET /` | Web UI z kursami |
| `GET /matches` | JSON — nadchodzące mecze + predykcje |
| `GET /players` | JSON — statystyki graczy |
| `GET /history` | JSON — historia wyników |
| `POST /refresh` | Ręczne uruchomienie cyklu scrapowania |

## Model matematyczny

```
λ1 = (avg_goals_scored_p1 + avg_goals_conceded_p2) / 2
λ2 = (avg_goals_scored_p2 + avg_goals_conceded_p1) / 2
λ_total = λ1 + λ2

P(over LINE) = 1 - Poisson_CDF(λ_total, floor(LINE))
odds = 1 / P(over LINE)
```

Linie: 3.5 / 4.5 / 5.5 / 6.5 / 7.5 / 8.5

## Cykl (co 60 sekund)

1. Scrape results → upsert do matches.json
2. Rebuild player stats → players.json
3. Scrape upcoming → upsert do matches.json
4. Scrape details (klik modal) → wzbogać rekordy
5. Final rebuild stats

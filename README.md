---
title: Aura Gold AI
emoji: 🪙
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
---

# Aura Gold AI — Hugging Face Space

The app serves the dashboard and FastAPI backend from the same public Space URL.
Live prices are fetched from Yahoo Finance (`GC=F`) when a prediction is requested.

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space and choose **Docker** as the SDK.
2. The Docker image installs `requirements-space.txt` and copies the runtime
   files `app.py`, `main.py`, `aura_price_engine.py`, `build_dataset.py`, `price_features.py`,
   `index.html`, `icon.png`, and `aura_price_models.pkl`.
3. Do not upload `.venv`, credentials, or the old standalone model files.
4. Wait for the Space to finish building, then open its public URL.

The app listens on port `7860`, which is the port expected by Hugging Face Spaces.
The browser automatically uses the same Space URL for `/predict`, so it does not
depend on the laptop's `127.0.0.1`.

## Deploy to Render

1. In Render, choose **New +** → **Blueprint** and select this GitHub repository.
2. Render reads `render.yaml`, builds the included `Dockerfile`, starts
   `app:app`, and exposes the web service on Render's assigned `PORT`. The
   compatibility entry point in `app.py` imports the FastAPI app from `main.py`;
   `/` serves the co-located `index.html`.
3. Use the generated `onrender.com` URL. Do not open `index.html` from the laptop;
   the dashboard uses the same public URL for `/predict`.
4. Render's health check is `/health`. It reports the market as closed during the
   weekend and serves the last known close instead of treating that as a server
   failure.

## Project structure

```text
C:\Gold/
├── build_dataset.py
├── train_price_forecaster.py
├── export_data_and_charts.py
├── app.py
├── main.py
├── aura_price_engine.py
├── price_features.py
├── gold_full_dataset.csv
├── gold_daily_dataset.csv
├── aura_price_models.pkl
└── charts/
```

## 1) Install dependencies

Use the project virtual environment on Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.venv` first if it does not exist. The training job also requires the
local PyTorch training dependency; it is not imported by the serving API.

## 2) Build the market snapshots

```powershell
.\.venv\Scripts\python.exe .\build_dataset.py
```

The builder and the live engine share ingestion for Yahoo Finance. Gold futures
(GC=F) supply completed hourly and daily candles. Crude oil (CL=F), Treasury
yield (^TNX), dollar index (DX-Y.NYB), gold implied volatility (^GVZ), equity
implied volatility (^VIX), and the iShares TIPS Bond ETF (TIP) supply daily
closes. Rolling macro signals use each symbol's own trading sessions. The
alignment shifts daily values by one calendar day before forward-filling, so a
Friday close first becomes eligible on Saturday and can carry through a market
closure. Hourly alignment uses UTC candle starts. The builder saves two years
of hourly history in gold_full_dataset.csv and five years of daily history in
gold_daily_dataset.csv.

## 3) Train the volatility-normalized forecasts

```powershell
.\.venv\Scripts\python.exe .\train_price_forecaster.py
```

Both frequencies use four economic and event layers alongside technical features.

1. Geopolitical and war risk adds a 20-session GVZ z-score to the absolute oil
   return shock measured against the prior 20 oil sessions.
2. Sentiment combines VIX relative to its 20-session EMA with RSI and gold's
   distance from its 20-bar EMA. A score at or above 0.25 activates the panic flag.
3. Real-rate and USD pressure adds the one-session yield change in decimal
   units to the DXY return and subtracts the TIP return. TIP is a market proxy
   affected by real yields and bond duration.
4. Calendar and liquidity features encode month, ISO week, and weekday as
   cycles. Flags cover Friday from 14:00 to 17:00 New York time (all Friday daily
   sessions), the first and last three scheduled US trading days of the month,
   recurring US and UK holidays and their preceding market sessions,
   October–November and January–February festival demand, and the week
   containing the first Friday of each month as an NFP proxy.

CatBoost and a 14-bar LSTM forecast 1H. CatBoost and Ridge forecast 7D and
30D. The models divide each target return by the trailing 20-bar standard
deviation of completed close returns times the square root of the horizon.
The blended prediction is converted back to dollars with the origin close and
that same volatility estimate.

Training, validation tuning, conformal calibration, and the locked holdout are
chronological, with a full horizon of excluded bars between adjacent partitions.
Daily validation has 200 usable rows per horizon across tuning and calibration,
plus the embargo between them. The holdout uses the last 200 eligible hourly
origins and the last 100 eligible daily origins per horizon. Blend weights
and shrinkage are chosen on validation with close MAE. The 80% likely and 95%
risk intervals use finite-sample quantiles of absolute normalized close errors
on the subsequent calibration slice. Interval widths scale with each origin's
gold price and realized volatility.

The feedback term applies 25% of the rolling mean of out-of-sample normalized
base forecast errors over the latest 20 matured bar positions. An error first
becomes eligible at its origin plus the forecast horizon. Holdout and live
forecasts use the same walk-forward rule and embargo exclusions. Model
parameters, shrinkage, and interval quantiles stay fixed throughout holdout
evaluation.

The Holdout Test benchmark compares close MAE against a no-change forecast
that carries the origin close forward. Positive skill means a lower MAE and is
computed as 100 × (1 − model MAE / baseline MAE). The report includes MAPE,
empirical coverage for both ranges, and the leading CatBoost feature
importances. The API and dashboard display the same holdout metrics.

## 4) Export audit data and charts

```powershell
.\.venv\Scripts\python.exe .\export_data_and_charts.py
.\.venv\Scripts\python.exe -m unittest -v test_forecast_integrity
```

The export verifies the two snapshot hashes against the trained bundle and uses
its stored holdout series. It writes cleaned_market_data.csv, aura_gold_dataset.csv
and four 300-DPI charts under charts/. The interval chart aligns the actual and
predicted closes by target date at all three horizons.

## 5) Run the local API and dashboard

```bash
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. Do not use the legacy standalone
model files (`gold_clf_model.pkl` and `gold_reg_model.pkl`).

## Important limitations

- Forecasts are estimates, not financial advice or guaranteed future prices.
- Horizon 1 is the next hourly candle (`1H`); horizons 7 and 30 are trading
  days, not calendar days.
- Displayed future dates are approximate weekdays and do not account for market holidays.
- Model performance must be judged against the baseline MAE and on newer unseen data.
- The dataset is expected to contain daily OHLC prices in a consistent currency/unit.
- Keep the original dataset source and license information.
- When the gold market is closed (Friday through Sunday, or before Monday 01:00 Iraq time),
  the app uses the last available model data instead of treating the closed market as a
  server connection error.

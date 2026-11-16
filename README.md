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
   files `app.py`, `main.py`, `aura_price_engine.py`, `price_features.py`,
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

## 2) Build the hourly dataset

```powershell
.\.venv\Scripts\python.exe .\build_dataset.py
```

The script fetches hourly gold, oil, Treasury-yield, and dollar data from
Yahoo Finance and writes `gold_full_dataset.csv`.

## 3) Train the price forecaster

```bash
.\.venv\Scripts\python.exe .\train_price_forecaster.py
```

This creates `aura_price_models.pkl` using:

- Hourly gold, crude oil, Treasury 10-year yield, and U.S. dollar data for the
  next 1-hour candle.
- Five years of daily data for the 7- and 30-trading-day forecasts.
- Engineered technical and relative market features; raw OHLC and future-target
  columns are excluded from model inputs.
- CatBoost + a 14-step LSTM for the hourly forecast, and Ridge + CatBoost for
  the daily forecasts.
- Chronological train, validation-tuning, independent conformal-calibration,
  and holdout partitions separated by horizon-sized embargoes.

The evaluation reports Close MAE, Close MAPE, the naive close-price baseline,
and empirical 95%/99% range coverage on the untouched holdout. Calibration
quantiles and ensemble weights are selected using validation data only.

Latest saved holdout audit:

- 1H: Close MAE $9.249 vs. $9.133 baseline; MAPE 0.230%; 95%/99% coverage
  94%/98%.
- 7 trading days: Close MAE $168.729 vs. $120.013 baseline; MAPE 4.022%;
  coverage 100%/100%.
- 30 trading days: Close MAE $423.514 vs. $339.876 baseline; MAPE 10.233%;
  coverage 98%/100%.

The current model did not beat the naive close-price baseline on any horizon.
The dashboard reports this explicitly; calibrated interval coverage does not
mean that point forecasts are more accurate or guarantee future performance.

## 4) Export audit data and charts

```powershell
.\.venv\Scripts\python.exe .\export_data_and_charts.py
```

This writes the cleaned and feature CSV files and four 300-DPI audit charts
under `charts/`.

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

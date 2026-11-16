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
aura_gold_ai_desktop/
├── archive/
│   └── XAU_1d_data.csv       # Put your Kaggle gold dataset here
├── models/                  # Trained model bundle is saved here
├── aura_gold_ai_dashboard.html
├── train_gold_pro.py
├── main.py
└── requirements.txt
```

## 1) Install dependencies

Open a terminal in this folder:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyWebView on Windows may need WebView2 Runtime. Install Microsoft's WebView2 Runtime if the app window does not open.

## 2) Add the dataset

Copy your CSV to:

`archive/XAU_1d_data.csv`

The trainer expects columns representing Date, Open, High, Low, Close.
Column names are case-insensitive. Volume is optional.

## 3) Train

```bash
python train_gold_pro.py
```

It creates:

`gold_bundle.pkl`

It trains separate models for 1, 2, 7, 30 and 252 trading-day returns. These represent approximately the next session, two sessions, one week, one month and one trading year. Training uses chronological walk-forward validation, a gap equal to the forecast horizon, and a final untouched holdout. Features are causal: each row uses only data available by that day's close, and hourly volatility is shifted to the previous day.

## 4) Run the desktop app

```bash
python main.py
```

`run_app.bat` starts this same safe API entry point (`main.py`). Do not use the legacy standalone model files (`gold_clf_model.pkl` and `gold_reg_model.pkl`).

The app loads the saved model and calls Python directly through PyWebView's JavaScript API. It does not need a separate web server.

## Important limitations

- Forecasts are estimates, not financial advice or guaranteed future prices.
- The 7/30 horizons mean trading rows in the dataset, not calendar days.
- Displayed future dates are approximate weekdays and do not account for market holidays.
- Model performance must be judged against the baseline MAE and on newer unseen data.
- The dataset is expected to contain daily OHLC prices in a consistent currency/unit.
- Keep the original dataset source and license information.
- When the gold market is closed (Friday through Sunday, or before Monday 01:00 Iraq time),
  the app uses the last available model data instead of treating the closed market as a
  server connection error.

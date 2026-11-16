from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch import nn

from build_dataset import (
    add_relative_features,
    add_technical_features,
    download_ticker,
)
from train_price_forecaster import daily_features, flatten, hourly_features


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "aura_price_models.pkl"


class PriceEngineError(Exception):
    """Expected error while loading data or producing a price forecast."""


class PriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True
        )
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.output(output[:, -1])


_lock = threading.Lock()
_bundle_cache: dict[str, object] = {"mtime": None, "bundle": None}


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise PriceEngineError(
            "حزمة النماذج غير موجودة. شغّل train_price_forecaster.py أولاً."
        )
    mtime = BUNDLE_PATH.stat().st_mtime
    with _lock:
        if _bundle_cache["mtime"] != mtime:
            _bundle_cache["bundle"] = joblib.load(BUNDLE_PATH)
            _bundle_cache["mtime"] = mtime
        return _bundle_cache["bundle"]  # type: ignore[return-value]


def _download_live_hourly() -> pd.DataFrame:
    gold = download_ticker("GC=F", ["Open", "High", "Low", "Close", "Volume"])
    oil = download_ticker("CL=F", ["Close"]).rename(columns={"Close": "Oil_Close"})
    treasury = download_ticker("^TNX", ["Close"]).rename(
        columns={"Close": "US10Y_Yield"}
    )
    dollar = download_ticker("DX-Y.NYB", ["Close"]).rename(columns={"Close": "DXY"})
    data = gold.join([oil, treasury, dollar], how="left").ffill().dropna()
    data = add_relative_features(add_technical_features(data))
    for lag in (1, 2, 3, 5, 10, 20):
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Oil_Lag_{lag}"] = data["Oil_Close"].shift(lag)
        data[f"US10Y_Lag_{lag}"] = data["US10Y_Yield"].shift(lag)
        data[f"DXY_Lag_{lag}"] = data["DXY"].shift(lag)
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def _download_live_daily() -> pd.DataFrame:
    frames = []
    for name, ticker in {
        "Gold": "GC=F",
        "Oil": "CL=F",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
    }.items():
        raw = flatten(
            yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        )
        if raw.empty:
            raise PriceEngineError(f"تعذر جلب بيانات {ticker}.")
        raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index)).tz_localize(None)
        if name == "Gold":
            part = raw[["Open", "High", "Low", "Close", "Volume"]].rename(
                columns={"High": "Gold_High", "Low": "Gold_Low"}
            )
        else:
            part = raw[["Close"]].rename(columns={"Close": f"{name}_Close"})
        frames.append(part)
    return frames[0].join(frames[1:], how="left").sort_index().ffill().dropna()


def _predict_lstm(meta: dict, features: pd.DataFrame) -> np.ndarray:
    model = PriceLSTM(meta["lstm_input_size"])
    model.load_state_dict(meta["lstm_state"])
    model.eval()
    scaled = meta["x_scaler"].transform(features[meta["features"]])
    window = meta["lstm_window"]
    if len(scaled) < window:
        raise PriceEngineError("بيانات السوق الحية غير كافية لتشغيل LSTM.")
    sequence = torch.from_numpy(scaled[-window:][None].astype(np.float32))
    with torch.no_grad():
        result = model(sequence).numpy()
    return meta["y_scaler"].inverse_transform(result)[0]


def predict(horizon: int) -> dict:
    horizon = int(horizon)
    bundle = load_bundle()
    if horizon not in bundle["models"]:
        raise PriceEngineError("الأفق المدعوم هو 1 أو 7 أو 30.")
    meta = bundle["models"][horizon]
    data = _download_live_hourly() if horizon == 1 else _download_live_daily()
    features = hourly_features(data) if horizon == 1 else daily_features(data)
    features = features.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    if any(column not in features for column in meta["features"]):
        missing = [c for c in meta["features"] if c not in features]
        raise PriceEngineError("ميزات التوقع ناقصة: " + ", ".join(missing[:5]))
    x = features[meta["features"]].iloc[[-1]]
    cat_return = np.asarray(meta["catboost"].predict(x)).reshape(-1)
    if horizon == 1:
        lstm_return = _predict_lstm(meta, features)
        returns = 0.60 * cat_return + 0.40 * lstm_return
    else:
        ridge_return = meta["ridge"].predict(
            meta["ridge_scaler"].transform(x)
        ).reshape(-1)
        returns = meta["damping_factor"] * (
            0.50 * ridge_return + 0.50 * cat_return
        )
    current = float(data["Close"].iloc[-1])
    prices = current * (1.0 + returns) + np.asarray(
        meta.get("bias_correction_usd", np.zeros(3))
    )
    metrics = meta["metrics"]
    interval = meta["conformal_intervals"]["99"]
    predicted_low = min(
        float(prices[1]) - interval["low_margin_usd"],
        float(prices[2]) - interval["close_margin_usd"],
    )
    predicted_high = max(
        float(prices[0]) + interval["high_margin_usd"],
        float(prices[2]) + interval["close_margin_usd"],
    )
    return {
        "status": "ok",
        "horizon": horizon,
        "current_price": round(current, 2),
        "predicted_price": round(float(prices[2]), 2),
        "predicted_high": round(predicted_high, 2),
        "predicted_low": round(predicted_low, 2),
        "volatility_usd": round(float(abs(predicted_high - predicted_low) / 2), 2),
        "drivers": {
            "dollar": round(float(data["DXY" if horizon == 1 else "DXY_Close"].iloc[-1]), 4),
            "treasury_10y": round(
                float(
                    data["US10Y_Yield" if horizon == 1 else "US10Y_Close"].iloc[-1]
                ),
                4,
            ),
            "oil": round(
                float(data["Oil_Close"].iloc[-1]),
                4,
            ),
        },
        "model_accuracy_pct": round(float(metrics["accuracy_pct"]), 4),
        "range_coverage_pct": round(
            float(meta.get("range_coverage_99_pct", meta["range_coverage_pct"])), 4
        ),
        "mae_usd": round(float(metrics["mae_usd"]), 4),
        "baseline_mae_usd": round(float(meta["baseline_metrics"]["mae_usd"]), 4),
        "beats_baseline": bool(meta["beats_baseline"]),
        "as_of": data.index[-1].strftime("%Y-%m-%d %H:%M"),
    }


def live_price() -> dict:
    data = _download_live_hourly()
    return {
        "status": "ok",
        "current_price": round(float(data["Close"].iloc[-1]), 3),
        "as_of": data.index[-1].strftime("%Y-%m-%d %H:%M"),
        "source": "Yahoo Finance GC=F",
    }

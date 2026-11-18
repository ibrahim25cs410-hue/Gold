from __future__ import annotations

import copy
import threading
import time
from _thread import LockType
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from build_dataset import (
    add_relative_features,
    add_technical_features,
    download_daily_indicators,
    download_ticker,
)
from price_features import (
    blend_returns,
    daily_features,
    flatten,
    hourly_features,
    sequence_window,
)


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "aura_price_models.pkl"


class PriceEngineError(Exception):
    """Expected error while loading data or producing a price forecast."""


_lock = threading.Lock()
_bundle_cache: dict[str, object] = {"mtime": None, "bundle": None}
_data_cache_lock = threading.Lock()
_data_cache_miss_locks: dict[str, LockType] = {}
_data_cache: dict[str, tuple[float, object]] = {}
LIVE_PRICE_TTL_SECONDS = 30
MARKET_DATA_TTL_SECONDS = 300
_CachedValue = TypeVar("_CachedValue")


def _copy_cached(value: object) -> object:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return copy.deepcopy(value)


def _cached_value(
    key: str, ttl_seconds: int, loader: Callable[[], _CachedValue]
) -> _CachedValue:
    now = time.monotonic()
    with _data_cache_lock:
        entry = _data_cache.get(key)
        if entry is not None and now < entry[0]:
            return cast(_CachedValue, _copy_cached(entry[1]))
        miss_lock = _data_cache_miss_locks.setdefault(key, threading.Lock())

    with miss_lock:
        with _data_cache_lock:
            entry = _data_cache.get(key)
            if entry is not None and time.monotonic() < entry[0]:
                return cast(_CachedValue, _copy_cached(entry[1]))
        value = loader()
        with _data_cache_lock:
            _data_cache[key] = (time.monotonic() + ttl_seconds, _copy_cached(value))
        return cast(_CachedValue, _copy_cached(value))


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise PriceEngineError(
            "حزمة النماذج غير موجودة. شغّل train_price_forecaster.py أولاً."
        )
    mtime = BUNDLE_PATH.stat().st_mtime
    with _lock:
        if _bundle_cache["mtime"] != mtime:
            try:
                bundle = joblib.load(BUNDLE_PATH)
            except Exception as exc:
                raise PriceEngineError(
                    f"تعذر تحميل حزمة النماذج {BUNDLE_PATH.name}: {exc}"
                ) from exc
            if not isinstance(bundle, dict) or not isinstance(bundle.get("models"), dict):
                raise PriceEngineError(
                    f"صيغة حزمة النماذج غير صالحة: {BUNDLE_PATH.name}."
                )
            _bundle_cache["bundle"] = bundle
            _bundle_cache["mtime"] = mtime
        return _bundle_cache["bundle"]  # type: ignore[return-value]


def _fetch_live_hourly() -> pd.DataFrame:
    gold = download_ticker("GC=F", ["Open", "High", "Low", "Close", "Volume"])
    oil = download_ticker("CL=F", ["Close"]).rename(columns={"Close": "Oil_Close"})
    treasury = download_ticker("^TNX", ["Close"]).rename(
        columns={"Close": "US10Y_Yield"}
    )
    dollar = download_ticker("DX-Y.NYB", ["Close"]).rename(columns={"Close": "DXY"})
    data = gold.join([oil, treasury, dollar], how="left")
    data = data.join(download_daily_indicators(data.index)).ffill().dropna()
    data = add_relative_features(add_technical_features(data))
    for lag in (1, 2, 3, 5, 10, 20):
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Oil_Lag_{lag}"] = data["Oil_Close"].shift(lag)
        data[f"US10Y_Lag_{lag}"] = data["US10Y_Yield"].shift(lag)
        data[f"DXY_Lag_{lag}"] = data["DXY"].shift(lag)
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def _download_live_hourly() -> pd.DataFrame:
    return _cached_value(
        "market_hourly", MARKET_DATA_TTL_SECONDS, _fetch_live_hourly
    )


def _fetch_live_daily() -> pd.DataFrame:
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
    data = frames[0].join(frames[1:], how="left").sort_index()
    data = data.join(download_daily_indicators(data.index)).ffill().dropna()
    return data


def _download_live_daily() -> pd.DataFrame:
    return _cached_value("market_daily", MARKET_DATA_TTL_SECONDS, _fetch_live_daily)


def _fetch_live_quote() -> dict:
    data = flatten(
        yf.download(
            "GC=F",
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )
    if data.empty:
        data = flatten(
            yf.download(
                "GC=F",
                period="5d",
                interval="1h",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        )
    if data.empty or "Close" not in data:
        raise PriceEngineError("تعذر جلب سعر الذهب الحي من Yahoo Finance.")
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    data.index = index
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        raise PriceEngineError("بيانات الذهب الحية لا تحتوي على أسعار إغلاق صالحة.")
    return {
        "status": "ok",
        "current_price": round(float(close.iloc[-1]), 3),
        "as_of": close.index[-1].strftime("%Y-%m-%d %H:%M"),
        "source": "Yahoo Finance GC=F",
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))


def _lstm_layer(sequence: np.ndarray, state: dict, layer: int) -> np.ndarray:
    weight_ih = np.asarray(state[f"lstm.weight_ih_l{layer}"], dtype=np.float32)
    weight_hh = np.asarray(state[f"lstm.weight_hh_l{layer}"], dtype=np.float32)
    bias_ih = np.asarray(state[f"lstm.bias_ih_l{layer}"], dtype=np.float32)
    bias_hh = np.asarray(state[f"lstm.bias_hh_l{layer}"], dtype=np.float32)
    hidden_size = weight_hh.shape[1]
    hidden = np.zeros(hidden_size, dtype=np.float32)
    cell = np.zeros(hidden_size, dtype=np.float32)
    output = np.empty((len(sequence), hidden_size), dtype=np.float32)

    for step, inputs in enumerate(sequence):
        gates = weight_ih @ inputs + bias_ih
        gates += weight_hh @ hidden + bias_hh
        input_gate, forget_gate, candidate, output_gate = np.split(gates, 4)
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        candidate = np.tanh(candidate)
        output_gate = _sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * np.tanh(cell)
        output[step] = hidden
    return output


def _predict_lstm(meta: dict, features: pd.DataFrame) -> np.ndarray:
    selected = features[meta["features"]]
    scaled = meta["x_scaler"].transform(selected)
    window = meta["lstm_window"]
    if len(scaled) < window:
        raise PriceEngineError("بيانات السوق الحية غير كافية لتشغيل LSTM.")
    sequence = sequence_window(scaled, len(scaled) - 1, window).astype(
        np.float32, copy=False
    )
    state = meta["lstm_state"]
    first_layer = _lstm_layer(sequence, state, 0)
    second_layer = _lstm_layer(first_layer, state, 1)
    output_weight = np.asarray(state["output.weight"], dtype=np.float32)
    output_bias = np.asarray(state["output.bias"], dtype=np.float32)
    scaled_prediction = (output_weight @ second_layer[-1] + output_bias)[None, :]
    return meta["y_scaler"].inverse_transform(scaled_prediction)[0]


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
        secondary_return = _predict_lstm(meta, features)
    else:
        secondary_return = meta["ridge"].predict(
            meta["ridge_scaler"].transform(x)
        ).reshape(-1)
    returns = blend_returns(
        cat_return,
        secondary_return,
        meta["ensemble_weights"],
        meta["damping_factor"],
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
        "range_coverage_95_pct": round(float(meta["range_coverage_95_pct"]), 4),
        "range_coverage_99_pct": round(float(meta["range_coverage_99_pct"]), 4),
        "range_coverage_pct": round(float(meta["range_coverage_99_pct"]), 4),
        "mae_close_usd": round(float(metrics["mae_close_usd"]), 4),
        "mae_high_usd": round(float(metrics["mae_high_usd"]), 4),
        "mae_low_usd": round(float(metrics["mae_low_usd"]), 4),
        "mape_close_pct": round(float(metrics["mape_close_pct"]), 4),
        "baseline_mae_close_usd": round(
            float(meta["baseline_metrics"]["mae_close_usd"]), 4
        ),
        "mae_usd": round(float(metrics["mae_usd"]), 4),
        "baseline_mae_usd": round(float(meta["baseline_metrics"]["mae_usd"]), 4),
        "beats_baseline": bool(meta["beats_baseline"]),
        "as_of": data.index[-1].strftime("%Y-%m-%d %H:%M"),
    }


def live_price() -> dict:
    return _cached_value("live_price", LIVE_PRICE_TTL_SECONDS, _fetch_live_quote)

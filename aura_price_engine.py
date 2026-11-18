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

from build_dataset import download_daily, download_hourly
from price_features import (
    BUNDLE_VERSION, blend_returns, daily_features, flatten, gold_target_prices,
    hourly_features, range_bounds, realized_volatility, residual_feedback,
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
            if (
                not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION
                or not isinstance(bundle.get("models"), dict)
            ):
                raise PriceEngineError(
                    f"Incompatible bundle {BUNDLE_PATH.name}. Run train_price_forecaster.py."
                )
            _bundle_cache["bundle"] = bundle
            _bundle_cache["mtime"] = mtime
        return _bundle_cache["bundle"]  # type: ignore[return-value]


def _download_live_hourly() -> pd.DataFrame:
    return _cached_value("market_hourly", MARKET_DATA_TTL_SECONDS, download_hourly)


def _download_live_daily() -> pd.DataFrame:
    return _cached_value("market_daily", MARKET_DATA_TTL_SECONDS, download_daily)


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


def _predict_lstm(
    meta: dict, features: pd.DataFrame, prediction_rows: np.ndarray,
) -> np.ndarray:
    window = meta["lstm_window"]
    context_start = int(prediction_rows[0]) - window + 1
    scaled = meta["x_scaler"].transform(features.iloc[context_start:]).astype(np.float32)
    state = meta["lstm_state"]
    output_weight = np.asarray(state["output.weight"], dtype=np.float32)
    output_bias = np.asarray(state["output.bias"], dtype=np.float32)
    predictions = []
    for row in prediction_rows:
        sequence = sequence_window(scaled, int(row) - context_start, window)
        first_layer = _lstm_layer(sequence, state, 0)
        second_layer = _lstm_layer(first_layer, state, 1)
        predictions.append(output_weight @ second_layer[-1] + output_bias)
    return meta["y_scaler"].inverse_transform(np.asarray(predictions))


def _forecast(meta: dict, data: pd.DataFrame, horizon: int) -> tuple[float, float]:
    features = hourly_features(data) if horizon == 1 else daily_features(data)
    features = features[meta["features"]].ffill()
    vol = realized_volatility(data["Close"], horizon)
    ready = features.notna().all(axis=1) & vol.notna()
    if features.empty or not ready.iloc[-1]:
        raise PriceEngineError("Market history is too short for the forecast.")
    last = len(data) - 1
    first_ready = ready.to_numpy().argmax()
    first_origin = first_ready + meta.get("lstm_window", 1) - 1
    if last < first_origin:
        raise PriceEngineError("Market history is too short for the forecast sequence.")
    window = meta["residual_window"]
    feedback_start = max(first_origin, last - horizon - window + 1)
    completed = np.arange(feedback_start, last - horizon + 1, dtype=int)
    dates = data.index[completed]
    completed = completed[(dates >= meta["feedback_start"]) & ~dates.isin(meta["feedback_embargo"])]
    rows = np.append(completed, last)
    selected = features.iloc[rows]
    cat = meta["catboost"].predict(selected, thread_count=2)
    if horizon == 1:
        secondary = _predict_lstm(meta, features, rows)
    else:
        secondary = meta["ridge"].predict(meta["ridge_scaler"].transform(selected))
    base = blend_returns(cat, secondary, meta["ensemble_weights"], meta["damping_factor"])
    close = data["Close"].to_numpy()
    dollar_vol = close * vol.to_numpy()
    trailing = min(len(data), horizon + window)
    residuals = np.full((trailing, 3), np.nan)
    offset = len(data) - trailing
    actual = gold_target_prices(data, horizon).iloc[completed].to_numpy()
    residuals[completed - offset] = (actual - close[completed, None]) / dollar_vol[completed, None] - base[:-1]
    correction = residual_feedback(residuals, horizon, window, meta["residual_gain"])[-1]
    close_forecast = close[last] + dollar_vol[last] * (base[-1, 2] + correction[2])
    return float(close_forecast), float(dollar_vol[last])


def predict(horizon: int) -> dict:
    horizon = int(horizon)
    bundle = load_bundle()
    if horizon not in bundle["models"]:
        raise PriceEngineError("الأفق المدعوم هو 1 أو 7 أو 30.")
    meta = bundle["models"][horizon]
    try:
        data = _download_live_hourly() if horizon == 1 else _download_live_daily()
    except RuntimeError as exc:
        raise PriceEngineError(str(exc)) from exc
    predicted, dollar_vol = _forecast(meta, data, horizon)
    current = float(data["Close"].iloc[-1])
    low_80, high_80 = range_bounds(predicted, dollar_vol, meta["conformal_intervals"]["80"])
    low_95, high_95 = range_bounds(predicted, dollar_vol, meta["conformal_intervals"]["95"])
    metrics = meta["metrics"]
    skill = meta["skill_score_pct"]
    return {
        "status": "ok",
        "horizon": horizon,
        "current_price": round(current, 2),
        "predicted_price": round(predicted, 2),
        "predicted_low_80": round(low_80, 2),
        "predicted_high_80": round(high_80, 2),
        "predicted_low": round(low_95, 2),
        "predicted_high": round(high_95, 2),
        "volatility_usd": round(dollar_vol, 2),
        "drivers": {
            "dollar": round(float(data["DXY" if horizon == 1 else "DXY_Close"].iloc[-1]), 4),
            "treasury_10y": round(float(data["US10Y_Yield" if horizon == 1 else "US10Y_Close"].iloc[-1]), 4),
            "oil": round(float(data["Oil_Close"].iloc[-1]), 4),
        },
        "mae_close_usd": round(metrics["mae_close_usd"], 4),
        "mae_high_usd": round(metrics["mae_high_usd"], 4),
        "mae_low_usd": round(metrics["mae_low_usd"], 4),
        "mape_close_pct": round(metrics["mape_close_pct"], 4),
        "baseline_mae_close_usd": round(meta["baseline_metrics"]["mae_close_usd"], 4),
        "mae_usd": round(metrics["mae_usd"], 4),
        "baseline_mae_usd": round(meta["baseline_metrics"]["mae_usd"], 4),
        "skill_score_pct": round(skill, 4) if skill is not None else None,
        "beats_baseline": bool(meta["beats_baseline"]),
        "range_coverage_80_pct": round(meta["range_coverage_80_pct"], 4),
        "range_coverage_95_pct": round(meta["range_coverage_95_pct"], 4),
        "as_of": data.index[-1].strftime("%Y-%m-%d %H:%M"),
    }


def live_price() -> dict:
    return _cached_value("live_price", LIVE_PRICE_TTL_SECONDS, _fetch_live_quote)

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["Target_High", "Target_Low", "Target_Close"]
RISK_FEATURE_COLUMNS = [
    "GVZ_level",
    "VIX_level",
    "GVZ_ret_1d",
    "VIX_ret_1d",
    "GVZ_zscore",
    "VIX_zscore",
    "GVZ_x_DXY",
]


def align_lagged_daily_close(
    daily_close: pd.Series, index: pd.Index
) -> pd.Series:
    daily = pd.to_numeric(daily_close, errors="coerce").copy()
    daily_index = pd.DatetimeIndex(pd.to_datetime(daily.index, errors="coerce"))
    if daily_index.isna().any():
        raise ValueError("Daily indicator data contains invalid dates.")
    if daily_index.tz is not None:
        daily_index = daily_index.tz_localize(None)
    daily.index = daily_index.normalize()
    daily = daily.loc[~daily.index.duplicated(keep="last")].sort_index().shift(1)

    target_index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if target_index.isna().any():
        raise ValueError("Target data contains invalid dates.")
    if target_index.tz is not None:
        target_index = target_index.tz_localize(None)
    aligned = daily.reindex(target_index.normalize(), method="ffill")
    aligned.index = index
    return aligned


def _fear_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    for name in ("GVZ", "VIX"):
        level = data[f"{name}_Close"]
        features[f"{name}_level"] = level
        features[f"{name}_ret_1d"] = level.pct_change(fill_method=None)
        rolling = level.rolling(20, min_periods=20)
        deviation = rolling.std(ddof=0).replace(0, np.nan)
        features[f"{name}_zscore"] = (level - rolling.mean()) / deviation
    return features


def completed_bar_trend(close: pd.Series, index: pd.Index, rule: str) -> pd.Series:
    completed_close = close.resample(rule, label="left", closed="left").last().shift(1)
    fast = completed_close.ewm(span=20, adjust=False).mean()
    slow = completed_close.ewm(span=50, adjust=False).mean()
    trend = (fast > slow).reindex(index, method="ffill").fillna(False)
    return trend.astype("int8")


def sequence_window(features: np.ndarray, current_index: int, window: int) -> np.ndarray:
    start = current_index - window + 1
    if window < 1 or start < 0 or current_index >= len(features):
        raise ValueError("Current row does not have a complete feature window.")
    return features[start : current_index + 1]


def blend_returns(
    catboost_returns: np.ndarray,
    secondary_returns: np.ndarray,
    weights: dict[str, float],
    damping_factor: float,
) -> np.ndarray:
    catboost_weight = weights["catboost"]
    return damping_factor * (
        catboost_weight * catboost_returns
        + (1.0 - catboost_weight) * secondary_returns
    )


def flatten(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


def daily_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"]
    features = pd.DataFrame(index=data.index)
    for name, column in {
        "Gold": "Close",
        "Oil": "Oil_Close",
        "US10Y": "US10Y_Close",
        "DXY": "DXY_Close",
    }.items():
        for period in (1, 3, 5, 10, 20, 60):
            features[f"{name}_Return_{period}d"] = data[column].pct_change(period)
    for span in (20, 50, 200):
        ema = close.ewm(span=span, adjust=False).mean()
        features[f"Gold_EMA{span}_Distance"] = (close - ema) / ema
    change = close.diff()
    gains = change.clip(lower=0).rolling(14).mean()
    losses = -change.clip(upper=0).rolling(14).mean()
    rs = gains / losses.where(losses != 0)
    features["RSI_14"] = (100 - 100 / (1 + rs)).fillna(50)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    features["MACD_Hist_Pct"] = (macd - signal) / close
    tr = pd.concat(
        [
            data["Gold_High"] - data["Gold_Low"],
            (close - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["ATR_Pct"] = tr.rolling(14).mean() / close
    features["Month"] = data.index.month
    features["DayOfWeek"] = data.index.dayofweek
    features["Is_Year_End"] = (data.index.month == 12).astype("int8")
    features = features.join(_fear_features(data))
    features["GVZ_x_DXY"] = features["GVZ_level"] * data["DXY_Close"]
    return features.replace([np.inf, -np.inf], np.nan)


def hourly_features(data: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "Open", "High", "Low", "Close", "Volume", "Oil_Close",
        "US10Y_Yield", "DXY", "GVZ_Close", "VIX_Close", *TARGET_COLUMNS,
        "Target_Close_8h", "Direction_Label",
    }
    columns = [
        column for column in data.columns
        if column in excluded or column.startswith("Target_")
    ]
    features = data.drop(columns=columns, errors="ignore").select_dtypes(
        include=np.number
    )
    day_index = pd.DatetimeIndex(data.index).normalize()
    daily_values = data[["GVZ_Close", "VIX_Close"]].copy()
    daily_values.index = day_index
    daily_values = daily_values.groupby(level=0, sort=True).first()
    daily_risk_features = _fear_features(daily_values).reindex(day_index)
    daily_risk_features.index = data.index
    features = features.join(daily_risk_features)
    features["GVZ_x_DXY"] = data["GVZ_Close"] * data["DXY"]
    return features.replace([np.inf, -np.inf], np.nan)

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["Target_High", "Target_Low", "Target_Close"]


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
    return features.replace([np.inf, -np.inf], np.nan)


def hourly_features(data: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "Open", "High", "Low", "Close", "Volume", "Oil_Close",
        "US10Y_Yield", "DXY", *TARGET_COLUMNS, "Target_Close_8h",
        "Direction_Label",
    }
    columns = [
        column for column in data.columns
        if column in excluded or column.startswith("Target_")
    ]
    features = data.drop(columns=columns, errors="ignore").select_dtypes(
        include=np.number
    )
    return features.replace([np.inf, -np.inf], np.nan)

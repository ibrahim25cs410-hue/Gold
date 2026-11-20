from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from price_features import MACRO_TICKERS, align_daily_macro, completed_bar_trend, flatten


ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "gold_full_dataset.csv"
DAILY_FILE = ROOT / "gold_daily_dataset.csv"
LAGS = (1, 2, 3, 5, 10, 20)
GOLD_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_market(ticker: str, period: str, interval: str, columns: list[str]) -> pd.DataFrame:
    # Yahoo occasionally rejects hourly requests at its rolling two-year boundary.
    periods=(period,"729d","60d","30d") if period=="730d" and interval=="1h" else (period,period)
    if period=="5y" and interval=="1d":
        periods=(period,"2y","6mo")
    for request_period in periods:
        data = flatten(yf.download(
            ticker, period=request_period, interval=interval, auto_adjust=False,
            progress=False, threads=False, timeout=15,
        ))
        if not data.empty and set(columns).issubset(data.columns):
            break
    if data.empty or not set(columns).issubset(data.columns):
        raise RuntimeError(f"Yahoo Finance returned no usable {interval} data for {ticker}.")
    data = data[columns].apply(pd.to_numeric, errors="coerce").dropna()
    dates = pd.DatetimeIndex(data.index)
    if interval == "1h":
        dates = dates.tz_convert("UTC") if dates.tz is not None else dates
        data.index = dates.tz_localize(None).floor("h")
    else:
        data.index = dates.tz_localize(None).normalize()
    if data.empty or data.index.hasnans:
        raise RuntimeError(f"Yahoo Finance returned invalid {interval} bars for {ticker}.")
    return data.loc[~data.index.duplicated(keep="last")].sort_index()


def download_macro_closes() -> dict[str, pd.Series]:
    return {name: download_market(ticker, "5y", "1d", ["Close"])["Close"]
            for name, ticker in MACRO_TICKERS.items()}


def download_hourly(
    macro_closes: dict[str, pd.Series] | None = None, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    now = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
    gold = download_market("GC=F", "730d", "1h", GOLD_COLUMNS)
    gold = gold.loc[gold.index + pd.Timedelta(hours=1) <= now.tz_convert("UTC").tz_localize(None)]
    closes = macro_closes if macro_closes is not None else download_macro_closes()
    macro = align_daily_macro(closes, gold.index).rename(
        columns={"US10Y_Close": "US10Y_Yield", "DXY_Close": "DXY"},
    )
    data = add_relative_features(add_technical_features(gold.join(macro)))
    for lag in LAGS:
        for name, column in (("Close", "Close"), ("Oil", "Oil_Close"),
                             ("US10Y", "US10Y_Yield"), ("DXY", "DXY")):
            data[f"{name}_Lag_{lag}"] = data[column].shift(lag)
    return data.replace([np.inf, -np.inf], np.nan)


def download_daily(
    macro_closes: dict[str, pd.Series] | None = None, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    now = (as_of if as_of is not None else pd.Timestamp.now(tz="UTC")).tz_convert("America/New_York")
    gold = download_market("GC=F", "5y", "1d", GOLD_COLUMNS)
    # Yahoo can publish an in-progress daily candle on the current session date.
    gold = gold.loc[gold.index + pd.Timedelta(hours=17) <= now.tz_localize(None)]
    gold = gold.rename(columns={"High": "Gold_High", "Low": "Gold_Low"})
    closes = macro_closes if macro_closes is not None else download_macro_closes()
    return gold.join(align_daily_macro(closes, gold.index))


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.rolling(14).mean()
    average_loss = losses.rolling(14).mean()
    relative_strength = average_gain / average_loss.where(average_loss != 0)
    data["RSI_14"] = (100 - (100 / (1 + relative_strength))).fillna(50)

    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    data["EMA_9"] = ema_9
    data["EMA_21"] = ema_21
    data["EMA_Spread_Pct"] = (ema_9 - ema_21) / ema_21

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR_14"] = true_range.rolling(14).mean()
    data["ATR_pct"] = data["ATR_14"] / close
    data["Volatility_10"] = close.pct_change().rolling(10).std()
    return data


def add_relative_features(data: pd.DataFrame) -> pd.DataFrame:
    series = {
        "Gold": data["Close"],
        "Oil": data["Oil_Close"],
        "US10Y": data["US10Y_Yield"],
        "DXY": data["DXY"],
    }
    for name, values in series.items():
        for period in (1, 4, 12, 24):
            data[f"{name}_PctChange_{period}h"] = values.pct_change(period)

    for period in (1, 4, 12, 24):
        data[f"Gold_DXY_Divergence_{period}h"] = (
            data[f"Gold_PctChange_{period}h"]
            - data[f"DXY_PctChange_{period}h"]
        )
        data[f"Gold_US10Y_Divergence_{period}h"] = (
            data[f"Gold_PctChange_{period}h"]
            - data[f"US10Y_PctChange_{period}h"]
        )

    atr = data["ATR_14"]
    for period in (24, 72):
        rolling_high = data["Close"].rolling(period).max().shift(1)
        rolling_low = data["Close"].rolling(period).min().shift(1)
        data[f"Distance_From_High_{period}h_ATR"] = (
            data["Close"] - rolling_high
        ) / atr
        data[f"Distance_From_Low_{period}h_ATR"] = (
            data["Close"] - rolling_low
        ) / atr

    data["Hour"] = data.index.hour
    data["DayOfWeek"] = data.index.day_of_week
    data["Is_London_NY"] = data["Hour"].between(12, 17).astype("int8")
    data["Is_Asian"] = data["Hour"].between(0, 7).astype("int8")
    data["Volume_Ratio"] = data["Volume"] / data["Volume"].rolling(20).mean().shift(1)
    data["Trend_4H"] = completed_bar_trend(data["Close"], data.index, "4h")
    data["Trend_1D"] = completed_bar_trend(data["Close"], data.index, "1D")
    return data


def triple_barrier_labels(data: pd.DataFrame) -> np.ndarray:
    close = data["Close"].to_numpy(dtype=np.float64)
    high = data["High"].to_numpy(dtype=np.float64)
    low = data["Low"].to_numpy(dtype=np.float64)
    atr = data["ATR_14"].to_numpy(dtype=np.float64)
    labels = np.full(len(data), -1, dtype=np.int8)

    for index in range(len(data) - 8):
        upper_target = close[index] + 1.5 * atr[index]
        lower_target = close[index] - 1.5 * atr[index]
        upper_stop = close[index] + 1.2 * atr[index]
        lower_stop = close[index] - 1.2 * atr[index]
        for future_index in range(index + 1, index + 9):
            up_target = high[future_index] >= upper_target
            down_target = low[future_index] <= lower_target
            up_stop = high[future_index] >= upper_stop
            down_stop = low[future_index] <= lower_stop
            if (up_target and down_stop) or (down_target and up_stop):
                break
            if up_target:
                labels[index] = 1
                break
            if down_target:
                labels[index] = 0
                break
            if up_stop or down_stop:
                break
    return labels


def build_dataset(
    macro_closes: dict[str, pd.Series] | None = None, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    data = download_hourly(macro_closes, as_of)
    for column in ("High", "Low", "Close"):
        data[f"Target_{column}"] = data[column].shift(-1)
    data["Target_Close_8h"] = data["Close"].shift(-8)
    data["Direction_Label"] = triple_barrier_labels(data)
    if len(data) <= 10_000:
        raise RuntimeError(f"Expected over 10,000 hourly bars, received {len(data)}.")
    return data


def main() -> None:
    as_of = pd.Timestamp.now(tz="UTC")
    macro_closes = download_macro_closes()
    hourly = build_dataset(macro_closes, as_of)
    daily = download_daily(macro_closes, as_of)
    if len(daily) < 500:
        raise RuntimeError(f"Expected at least 500 daily bars, received {len(daily)}.")
    for frame, destination in ((hourly, OUTPUT_FILE), (daily, DAILY_FILE)):
        temporary = destination.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index_label="Date")
        temporary.replace(destination)
        print(f"Saved {len(frame):,} completed bars to {destination.name}.")


if __name__ == "__main__":
    main()

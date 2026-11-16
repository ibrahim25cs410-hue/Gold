from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def ensure_yfinance() -> None:
    if importlib.util.find_spec("yfinance") is None:
        print("yfinance غير مثبتة؛ جاري تثبيتها في بيئة Python الحالية...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yfinance"],
            check=True,
        )


ensure_yfinance()

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "gold_full_dataset.csv"
LAGS = (1, 2, 3, 5, 10, 20)
TICKERS = {
    "الذهب (GC=F)": "GC=F",
    "النفط الخام (CL=F)": "CL=F",
    "السندات الأمريكية (^TNX)": "^TNX",
    "مؤشر الدولار (DX-Y.NYB)": "DX-Y.NYB",
}


def remove_timezone(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if dates.isna().any():
        raise ValueError("تعذر تحويل بعض التواريخ إلى صيغة تاريخ صالحة.")
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    return dates.floor("h")


def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


def download_ticker(ticker: str, columns: list[str]) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period="730d",
        interval="1h",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"لم تُرجع yfinance بيانات للرمز {ticker}.")

    data = flatten_yfinance_columns(data)
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(
            f"الأعمدة المطلوبة {missing_columns} غير متاحة للرمز {ticker}."
        )
    data = data.loc[:, columns].copy()
    data.index = remove_timezone(data.index)
    return data[~data.index.duplicated(keep="last")].sort_index()


def check_yfinance_connection() -> None:
    print("=== فحص اتصال yfinance ببيانات الساعة ===")
    for label, ticker in TICKERS.items():
        data = yf.download(
            ticker,
            period="5d",
            interval="1h",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise RuntimeError(
                f"فشل جلب البيانات للرمز {label} ({ticker}). تحقق من الاتصال."
            )
        data = flatten_yfinance_columns(data)
        if "Close" not in data.columns:
            raise RuntimeError(f"عمود Close غير متاح للرمز {ticker}.")
        print(
            f"[OK] {label}: {len(data)} شمعة، "
            f"آخر إغلاق = {float(data['Close'].iloc[-1]):.2f}"
        )


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
    close_4h = data["Close"].resample("4h").last()
    ema_20_4h = close_4h.ewm(span=20, adjust=False).mean()
    ema_50_4h = close_4h.ewm(span=50, adjust=False).mean()
    data["Trend_4H"] = (
        (ema_20_4h > ema_50_4h)
        .reindex(data.index, method="ffill")
        .fillna(0)
        .astype("int8")
    )
    close_1d = data["Close"].resample("1D").last()
    ema_20_1d = close_1d.ewm(span=20, adjust=False).mean()
    ema_50_1d = close_1d.ewm(span=50, adjust=False).mean()
    data["Trend_1D"] = (
        (ema_20_1d > ema_50_1d)
        .reindex(data.index, method="ffill")
        .fillna(0)
        .astype("int8")
    )
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


def build_dataset() -> pd.DataFrame:
    check_yfinance_connection()
    print("\nجاري سحب بيانات آخر سنتين بفاصل ساعة...")

    gold = download_ticker(
        "GC=F", ["Open", "High", "Low", "Close", "Volume"]
    )
    oil = download_ticker("CL=F", ["Close"]).rename(
        columns={"Close": "Oil_Close"}
    )
    treasury = download_ticker("^TNX", ["Close"]).rename(
        columns={"Close": "US10Y_Yield"}
    )
    dollar = download_ticker("DX-Y.NYB", ["Close"]).rename(
        columns={"Close": "DXY"}
    )

    data = gold.join([oil, treasury, dollar], how="left").ffill().dropna()
    data = add_technical_features(data)
    data = add_relative_features(data)

    for lag in LAGS:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Oil_Lag_{lag}"] = data["Oil_Close"].shift(lag)
        data[f"US10Y_Lag_{lag}"] = data["US10Y_Yield"].shift(lag)
        data[f"DXY_Lag_{lag}"] = data["DXY"].shift(lag)

    data["Target_High"] = data["High"].shift(-1)
    data["Target_Low"] = data["Low"].shift(-1)
    data["Target_Close"] = data["Close"].shift(-1)
    data["Target_Close_8h"] = data["Close"].shift(-8)
    data["Direction_Label"] = triple_barrier_labels(data)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= 10_000:
        raise RuntimeError(
            f"تم توليد {len(data)} شمعة فقط؛ المطلوب أكثر من 10,000 شمعة."
        )
    return data


def main() -> None:
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_FILE, index_label="Date")
    labels = dataset["Direction_Label"].value_counts().to_dict()
    print(
        f"\nتم بناء الداتا سيت الساعي بنجاح في {OUTPUT_FILE.name}!"
        f"\nعدد الشموع: {len(dataset)} | عدد الأعمدة: {len(dataset.columns)}"
        f"\nالتصنيف: صعود واضح={labels.get(1, 0)} | "
        f"هبوط واضح={labels.get(0, 0)} | محايد={labels.get(-1, 0)}"
    )


if __name__ == "__main__":
    main()

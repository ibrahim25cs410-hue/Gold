from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, EasterMonday, GoodFriday, Holiday,
    USLaborDay, USMemorialDay, USThanksgivingDay, nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset
from dateutil.relativedelta import MO


TARGET_COLUMNS = ["Target_High", "Target_Low", "Target_Close"]
BUNDLE_VERSION = 2
RESIDUAL_GAIN = 0.25
RESIDUAL_WINDOW = 20
MACRO_TICKERS = {
    "Oil_Close": "CL=F", "US10Y_Close": "^TNX", "DXY_Close": "DX-Y.NYB",
    "GVZ_Close": "^GVZ", "VIX_Close": "^VIX", "TIP_Close": "TIP",
}
ECONOMIC_FEATURES = ["geopolitical_war_risk", "psych_sentiment_index",
                     "panic_regime_flag", "real_rate_usd_pressure"]
MACRO_SIGNALS = ["gvz_zscore_20d", "oil_shock_20d", "vix_ema_ratio",
                 "yield_change_1d", "dxy_return_1d", "tip_return_1d"]


def align_lagged_daily_close(
    daily_close: pd.Series, index: pd.Index
) -> pd.Series:
    daily = pd.to_numeric(daily_close, errors="coerce").dropna()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily.loc[~daily.index.duplicated(keep="last")].sort_index()
    target_index = pd.DatetimeIndex(index).tz_localize(None).normalize()
    if daily.index.hasnans or target_index.hasnans:
        raise ValueError("Daily indicator or target data contains invalid dates.")
    if daily.empty or target_index.empty:
        return pd.Series(np.nan, index=index, name=daily.name)
    calendar = pd.date_range(daily.index[0], daily.index[-1] + pd.Timedelta(days=1))
    lagged = daily.reindex(calendar).shift(1).ffill()
    aligned = lagged.reindex(target_index, method="ffill")
    aligned.index = index
    return aligned


def align_daily_macro(closes: dict[str, pd.Series], index: pd.Index) -> pd.DataFrame:
    # Compute on each exchange's own sessions before lagging and expanding to bars.
    gvz, vix, oil, yield_, dollar, tip = (
        closes[name] for name in ("GVZ_Close", "VIX_Close", "Oil_Close",
                                  "US10Y_Close", "DXY_Close", "TIP_Close")
    )
    gvz_window = gvz.rolling(20)
    gvz_z = (gvz - gvz_window.mean()) / gvz_window.std(ddof=0).clip(lower=1e-8)
    oil_return = oil.diff() / oil.shift().abs().clip(lower=1e-8)
    oil_window = oil_return.shift().rolling(20)
    oil_shock = ((oil_return - oil_window.mean()) / oil_window.std(ddof=0).clip(lower=1e-6)).abs()
    signals = dict(zip(MACRO_SIGNALS, (
        gvz_z, oil_shock, vix / vix.ewm(span=20, adjust=False, min_periods=20).mean(),
        yield_.diff(), dollar.pct_change(fill_method=None), tip.pct_change(fill_method=None),
    )))
    return pd.DataFrame({name: align_lagged_daily_close(series, index)
                         for name, series in (closes | signals).items()}, index=index)


class USMarketCalendar(AbstractHolidayCalendar):
    rules = [Holiday("New Year", month=1, day=1,
                     observance=lambda date: date + pd.Timedelta(days=1) if date.weekday() == 6 else date),
             GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                     observance=nearest_workday),
             Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
             USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


class UKMarketCalendar(AbstractHolidayCalendar):
    rules = [GoodFriday, EasterMonday,
             Holiday("Early May", month=5, day=1, offset=DateOffset(weekday=MO(1))),
             Holiday("Spring", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
             Holiday("Summer", month=8, day=31, offset=DateOffset(weekday=MO(-1)))]


def event_seasonality_features(index: pd.DatetimeIndex, *, hourly: bool) -> pd.DataFrame:
    local = index.tz_localize("UTC").tz_convert("America/New_York") if hourly else index
    local = local.tz_localize(None)
    # Sunday evening Globex trading belongs to Monday's session.
    session = (local + pd.Timedelta(hours=6)).normalize() if hourly else local.normalize()
    if session.empty:
        raise ValueError("Calendar features require at least one bar.")
    start = session.min().to_period("M").start_time
    end = session.max().to_period("M").end_time.normalize()
    calendar_start, calendar_end = start - pd.Timedelta(days=10), end + pd.Timedelta(days=10)
    us_holidays = USMarketCalendar().holidays(calendar_start, calendar_end)
    uk_holidays = set(UKMarketCalendar().holidays(calendar_start, calendar_end))
    for year in range(calendar_start.year, calendar_end.year + 1):
        for month, day in ((1, 1), (12, 25), (12, 26)):
            actual = pd.Timestamp(year, month, day)
            observed = actual
            while observed.weekday() >= 5 or observed in uk_holidays:
                observed += pd.Timedelta(days=1)
            uk_holidays.add(observed)
    holiday_window = set()
    for market_holidays in (us_holidays, pd.DatetimeIndex(sorted(uk_holidays))):
        business_day = CustomBusinessDay(holidays=market_holidays)
        holiday_window.update(market_holidays)
        holiday_window.update(day - business_day for day in market_holidays)
    trading = pd.date_range(start, end, freq=CustomBusinessDay(holidays=us_holidays))
    counter = pd.Series(trading, index=trading).groupby(trading.to_period("M"))
    turn_of_month = trading[(counter.cumcount() < 3) | (counter.cumcount(ascending=False) < 3)]

    result = pd.DataFrame(index=index)
    for name, values, period in (("month", session.month - 1, 12),
                                 ("week", session.isocalendar().week.to_numpy(dtype=int) - 1, 53),
                                 ("dow", session.day_of_week, 7)):
        angle = 2 * np.pi * values / period
        result[f"{name}_sin"], result[f"{name}_cos"] = np.sin(angle), np.cos(angle)
    friday_close = (local.day_of_week == 4) & (local.hour >= 14) & (local.hour < 17)
    result["is_friday_close_window"] = (friday_close if hourly else session.day_of_week == 4).astype("int8")
    result["is_turn_of_month"] = session.isin(turn_of_month).astype("int8")
    result["is_major_holiday_window"] = session.isin(holiday_window).astype("int8")
    result["festival_demand_regime"] = np.select(
        [session.month.isin([10, 11]), session.month.isin([1, 2])], [1, 2], default=0
    ).astype("int8")
    monday = session - pd.to_timedelta(session.day_of_week, unit="D")
    friday = monday + pd.Timedelta(days=4)
    result["nfp_event_week"] = (friday.day <= 7).astype("int8")
    return result


def _economic_features(data: pd.DataFrame, close: pd.Series, *, hourly: bool) -> pd.DataFrame:
    change = close.diff()
    gains = change.clip(lower=0).rolling(14).mean()
    losses = -change.clip(upper=0).rolling(14).mean()
    rsi = (100 * gains / (gains + losses)).fillna(50)
    ema = close.ewm(span=20, adjust=False).mean()
    deviation = close / ema - 1
    result = event_seasonality_features(data.index, hourly=hourly)
    result["RSI_14"] = rsi
    result["Gold_EMA20_Distance"] = deviation
    result["geopolitical_war_risk"] = data["gvz_zscore_20d"] + data["oil_shock_20d"]
    sentiment = data["vix_ema_ratio"] - 1 - (rsi - 50) / 100 - 5 * deviation
    result["psych_sentiment_index"] = sentiment
    result["panic_regime_flag"] = (sentiment >= 0.25).astype("int8").where(sentiment.notna())
    # TNX is in yield percentage points. Positive TIP returns offset pressure.
    result["real_rate_usd_pressure"] = data["yield_change_1d"] / 100 + data["dxy_return_1d"] - data["tip_return_1d"]
    return result


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
            features[f"{name}_Return_{period}d"] = data[column].pct_change(period, fill_method=None)
    for span in (50, 200):
        ema = close.ewm(span=span, adjust=False).mean()
        features[f"Gold_EMA{span}_Distance"] = (close - ema) / ema
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
    features = features.join(_economic_features(data, close, hourly=False))
    return features.replace([np.inf, -np.inf], np.nan)


def hourly_features(data: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "Open", "High", "Low", "Close", "Volume", "Oil_Close",
        "US10Y_Yield", "DXY", *MACRO_TICKERS, *MACRO_SIGNALS, *TARGET_COLUMNS,
        "Target_Close_8h", "Direction_Label", "RSI_14", "EMA_9", "EMA_21",
        "DayOfWeek", "ATR_14", "MACD", "MACD_Signal",
    }
    columns = [
        column for column in data.columns
        if column in excluded or column.startswith("Target_") or "_Lag_" in column
    ]
    features = data.drop(columns=columns, errors="ignore").select_dtypes(
        include=np.number
    )
    features = features.join(_economic_features(data, data["Close"], hourly=True))
    return features.replace([np.inf, -np.inf], np.nan)


def realized_volatility(close: pd.Series, horizon: int) -> pd.Series:
    return close.pct_change(fill_method=None).rolling(20).std().clip(lower=1e-5) * np.sqrt(horizon)


def gold_target_prices(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    high, low = ("High", "Low") if horizon == 1 else ("Gold_High", "Gold_Low")
    return data[[high, low, "Close"]].shift(-horizon).set_axis(TARGET_COLUMNS, axis=1)


def residual_feedback(
    residuals: np.ndarray, horizon: int, window: int, gain: float
) -> np.ndarray:
    """Only origins at or before t-horizon can affect the forecast at t."""
    return (
        pd.DataFrame(residuals).shift(horizon).rolling(window, min_periods=1)
        .mean().fillna(0).to_numpy() * gain
    )


def range_bounds(close: np.ndarray, dollar_vol: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    margin = dollar_vol * quantile
    return close - margin, close + margin

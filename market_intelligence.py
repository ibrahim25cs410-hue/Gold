"""Causal gold features and explicit UTC availability for economic releases."""
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from forecast_evaluation import clock_for
from price_features import daily_features, hourly_features

FRED_SERIES = ("DGS2", "DGS10", "DFII10", "T10YIE", "DFF")


def fetch_fred(folder):
    """Save a fresh, separate source snapshot for repeatable offline research."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for series in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2021-01-01"
        with urlopen(url, timeout=25) as response:
            payload = response.read()
        if payload.splitlines()[0] != f"observation_date,{series}".encode():
            raise ValueError(f"Unexpected FRED schema for {series}")
        (folder / f"{series}.csv").write_bytes(payload)


def asof_releases(origins, releases):
    """Join by availability, preserving the archive's observation and revision dates."""
    releases = releases.copy()
    releases["available_at"] = pd.to_datetime(releases.available_at, utc=True)
    if releases.available_at.isna().any() or releases.available_at.duplicated().any():
        raise ValueError("Release availability must be unique and known")
    left = pd.DataFrame({"available_at": pd.to_datetime(origins, utc=True)})
    return pd.merge_asof(left, releases.sort_values("available_at"), on="available_at")


def rates(origins, folder):
    columns = {}
    for name in FRED_SERIES:
        path = Path(folder) / f"{name}.csv"
        if not path.exists():
            return None
        source = pd.read_csv(path)
        observation = pd.to_datetime(source.iloc[:, 0])
        values = pd.to_numeric(source[name], errors="coerce")
        # Latest-vintage market observations. Approximated eligibility is deliberately
        # conservative and is never represented as the exact publication timestamp.
        available = (observation + pd.offsets.BDay(2)).dt.tz_localize("America/New_York")
        # Holidays can map multiple source dates to the same business day. Missing
        # quotes never displace a valid reading.
        quotes = pd.DataFrame({"observation": observation, "available_at": available,
                               "rate": values}).dropna().sort_values("observation")
        quotes = quotes.drop_duplicates("available_at", keep="last")
        quotes["change20"] = quotes.rate.diff(20)
        joined = asof_releases(origins, quotes)
        fresh = (pd.to_datetime(origins, utc=True).tz_localize(None)
                 - pd.DatetimeIndex(joined.observation)).days <= 12
        columns[name] = joined.rate.where(fresh).to_numpy()
        columns[name + "_change20"] = joined.change20.where(fresh).to_numpy()
    out = pd.DataFrame(columns)
    out["curve_10y_2y"] = out.DGS10 - out.DGS2
    return out


def feature_groups(data, horizon, folder):
    close = data.Close
    high = data.High if horizon == 1 else data.Gold_High
    low = data.Low if horizon == 1 else data.Gold_Low
    f = hourly_features(data) if horizon == 1 else daily_features(data)
    windows = (3, 12, 24) if horizon == 1 else (5, 20, 60)
    returns = np.log(close).diff()
    span = (high - low).replace(0, np.nan)
    gold = pd.DataFrame({f"momentum_{n}": np.log(close).diff(n) for n in windows})
    gold["volatility"] = returns.rolling(windows[-1]).std()
    previous = close.shift()
    tr = pd.concat([high-low, (high-previous).abs(), (low-previous).abs()], axis=1).max(axis=1)
    up, down = high.diff(), -low.diff()
    pos = up.where((up > down) & (up > 0), 0).rolling(14).sum()
    neg = down.where((down > up) & (down > 0), 0).rolling(14).sum()
    technical = pd.DataFrame({
        "body": ((close-data.Open)/span).fillna(0),
        "upper_wick": ((high-np.maximum(close,data.Open))/span).fillna(0),
        "lower_wick": ((np.minimum(close,data.Open)-low)/span).fillna(0),
        "close_position": ((close-low)/span).fillna(.5),
        "atr": tr.rolling(14).mean()/close,
        "adx": ((pos-neg).abs()/(pos+neg).replace(0,np.nan)).fillna(0).rolling(14).mean(),
        "bollinger_z": ((close-close.rolling(20).mean())/close.rolling(20).std()).fillna(0),
        "resistance": close/high.shift().rolling(windows[-1]).max()-1,
        "support": close/low.shift().rolling(windows[-1]).min()-1,
        "macd": (close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean())/close,
        "rsi": f.RSI_14,
        "ema_distance": f.Gold_EMA20_Distance,
    })
    if horizon == 1:
        technical = technical.join(f[["Trend_4H", "Trend_1D"]])
    volume = np.log1p(data.Volume.where(data.Volume >= 0))
    futures = pd.DataFrame({"volume_anomaly": volume-volume.shift().rolling(20).median(),
                            "zero_volume": volume.eq(0).astype(float)})
    sessions = clock_for(data.index, horizon).reindex(data.index)
    event = f[["month_sin", "month_cos", "dow_sin", "dow_cos", "is_turn_of_month"]].copy()
    event["quarter_end"] = (data.index.month % 3 == 0) * event.is_turn_of_month
    if horizon == 1:
        event["hour_sin"] = np.sin(2*np.pi*data.index.hour/24)
        event["hour_cos"] = np.cos(2*np.pi*data.index.hour/24)
    macro = f[["real_rate_usd_pressure", "psych_sentiment_index"]].copy()
    dxy = data["DXY" if horizon == 1 else "DXY_Close"]
    yield10 = data["US10Y_Yield" if horizon == 1 else "US10Y_Close"]
    macro["dollar_20"] = dxy.pct_change(windows[-1], fill_method=None)
    macro["yield_20"] = yield10.diff(windows[-1])
    gov = rates(pd.DatetimeIndex(sessions.dropna()), folder)
    if gov is not None:
        gov.index = sessions.dropna().index
        macro = macro.join(gov)
    return {"gold": gold, "technical": technical, "macro": macro, "futures": futures,
            "calendar": event, "risk_proxy": f[["geopolitical_war_risk"]]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Archive FRED market rate CSVs")
    parser.add_argument("folder", type=Path)
    fetch_fred(parser.parse_args().folder)

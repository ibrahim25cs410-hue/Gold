# -*- coding: utf-8 -*-
"""
gold_engine.py  -  المحرك المشترك لمشروع Aura Gold AI
يُستخدم من التدريب (train_gold_pro.py) ومن السيرفر (main.py) بنفس الدالة بالضبط،
حتى لا يحصل اختلاف بين ميزات التدريب وميزات التوقع (سبب شائع للنتائج الثابتة).
"""
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("GOLD_DATA_DIR", BASE_DIR / "archive"))
BUNDLE_PATH = Path(os.environ.get("GOLD_BUNDLE", BASE_DIR / "gold_bundle.pkl"))
HORIZONS = (1, 2, 7, 30, 252)  # اليوم ، يومان، أسبوع، شهر، سنة تداول تقريباً
WARMUP = 252                    # صفوف البداية التي لا تكتمل فيها المؤشرات
OPTIONAL = ("rv_h", "rv_h_ratio")  # ميزات قد تكون ناقصة بشكل طبيعي


class EngineError(Exception):
    """خطأ مفهوم يُعرض للمستخدم بدل رقم وهمي."""


# ----------------------------------------------------------------- البيانات
def _read_csv(name, required=True):
    path = DATA_DIR / name
    if not path.exists():
        if required:
            raise EngineError(f"الملف غير موجود: {path}")
        return None
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:                       # ملف مفصول بفاصلة عادية
        df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def load_daily():
    return _read_csv("XAU_1d_data.csv")


def load_hourly():
    return _read_csv("XAU_1h_data.csv", required=False)


# ------------------------------------------------------------------ الميزات
def make_features(daily, hourly=None):
    """
    ميزات بلا اتجاه سعري (نسب ونسب مئوية وz-score فقط).
    السبب: القيم الخام (EMA, Bollinger بالدولار) تخرج عن نطاق التدريب عندما يصعد
    الذهب لمستويات جديدة، فتعطي أشجار القرار نفس الناتج دائماً (رقم ثابت).
    كل ميزة في يوم t تستخدم بيانات حتى إغلاق اليوم t فقط (بدون تسريب من المستقبل).
    """
    d = daily.set_index("Date")
    c, o, h, l = d["Close"], d["Open"], d["High"], d["Low"]
    f = pd.DataFrame(index=d.index)
    r1 = c.pct_change()

    for n in (1, 2, 3, 5, 10, 21, 63):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 10, 21, 63):
        f[f"vol_{n}"] = r1.rolling(n).std()
    f["vol_ratio"] = f["vol_5"] / (f["vol_21"] + 1e-12)

    for p in (5, 10, 21, 50, 100, 200):
        f[f"ema_dist_{p}"] = c / c.ewm(span=p, adjust=False).mean() - 1
    f["ema21_slope"] = c.ewm(span=21, adjust=False).mean().pct_change(5)

    delta = c.diff()
    for n in (7, 14):
        up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        f[f"rsi_{n}"] = 100 - 100 / (1 + up / (dn + 1e-12))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - sig) / c

    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_z"] = (c - m20) / (s20 + 1e-12)
    f["bb_width"] = 4 * s20 / m20

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c

    f["body"] = (c - o) / o
    f["upper_wick"] = (h - np.maximum(o, c)) / c
    f["lower_wick"] = (np.minimum(o, c) - l) / c
    f["range"] = (h - l) / c
    f["gap"] = o / c.shift() - 1

    hi, lo = c.rolling(252).max(), c.rolling(252).min()
    f["pos_252"] = (c - lo) / (hi - lo + 1e-12)
    f["drawdown_252"] = c / hi - 1
    f["z_63"] = (c - c.rolling(63).mean()) / (c.rolling(63).std() + 1e-12)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_14"] = (c - ll) / (hh - ll + 1e-12)

    f["dow"] = d.index.dayofweek
    f["month"] = d.index.month

    # تذبذب لحظي من الشموع الساعية (اختياري)
    f["rv_h"], f["rv_h_ratio"] = np.nan, np.nan
    if hourly is not None and len(hourly):
        hr = hourly.set_index("Date")["Close"].pct_change()
        rv = hr.groupby(hr.index.normalize()).std().reindex(d.index)
        # A daily row is available only after that day's close. Use the
        # previous calendar day's hourly volatility to stay conservative.
        rv = rv.shift(1)
        f["rv_h"] = rv
        f["rv_h_ratio"] = rv / (rv.rolling(21, min_periods=10).mean() + 1e-12)

    return f.replace([np.inf, -np.inf], np.nan)


def make_dataset(horizon, daily, hourly=None):
    """جدول التدريب: ميزات + العائد الفعلي بعد `horizon` جلسة (بالمئة)."""
    f = make_features(daily, hourly)
    c = daily.set_index("Date")["Close"]
    ds = f.iloc[WARMUP:].copy()
    future_date = pd.Series(c.index, index=c.index).shift(-horizon)
    ds["y_pct"] = ((c.shift(-horizon) / c - 1) * 100).reindex(ds.index)
    for column in OPTIONAL:
        if column in ds and not ds[column].notna().any():
            ds = ds.drop(columns=column)
    core = [x for x in ds.columns if x not in OPTIONAL and x != "y_pct"]
    ds = ds.dropna(subset=core + ["y_pct"])
    target_dates = future_date.reindex(ds.index)
    if not (target_dates > ds.index.to_series()).all():
        raise EngineError("تم رفض بيانات التدريب: الهدف ليس مستقبلياً لكل صف.")
    return ds


# -------------------------------------------------------------------- التوقع
_cache = {"key": None}


def _state():
    files = [DATA_DIR / "XAU_1d_data.csv", DATA_DIR / "XAU_1h_data.csv", BUNDLE_PATH]
    key = tuple(p.stat().st_mtime if p.exists() else 0 for p in files)
    if _cache["key"] != key:
        if not BUNDLE_PATH.exists():
            raise EngineError("النماذج غير موجودة. شغّل train_gold_pro.py أولاً.")
        daily, hourly = load_daily(), load_hourly()
        _cache.update(key=key, daily=daily, feats=make_features(daily, hourly),
                      bundle=joblib.load(BUNDLE_PATH))
    return _cache


def predict(horizon):
    st = _state()
    bundle = st["bundle"]
    m = bundle["models"].get(int(horizon))
    if m is None:
        raise EngineError(f"المدة {horizon} غير مدرّبة. المتاح: {sorted(bundle['models'])}")

    x = st["feats"].iloc[[-1]][bundle["features"]]
    missing = [k for k in x.columns if k not in OPTIONAL and x[k].isna().any()]
    if missing:
        raise EngineError(f"بيانات آخر يوم ناقصة ({', '.join(missing[:4])}…). تأكد من ملف البيانات.")

    p_up = 0.5 * (float(m["clf"].predict_proba(x)[0, 1]) + float(m["lr"].predict_proba(x)[0, 1]))
    pct = float(m["reg"].predict(x)[0])
    thr = m["thr"]

    # إشارة قوية فقط عندما يتفق المصنّف والانحدار ويتجاوز الاحتمال العتبة
    if p_up >= thr and pct > 0:
        direction = "BUY"
    elif p_up <= 1 - thr and pct < 0:
        direction = "SELL"
    else:
        direction = "HOLD"

    daily = st["daily"]
    cur = float(daily["Close"].iloc[-1])
    as_of = daily["Date"].iloc[-1]
    mt = m["metrics"]
    return {
        "status": "ok",
        "horizon": int(horizon),
        "current_price": round(cur, 2),
        "target_price": round(cur * (1 + pct / 100), 2),
        "predicted_pct_change": round(pct, 3),
        "predicted_direction": direction,
        "prob_up": round(p_up, 4),
        "confidence": round(max(p_up, 1 - p_up) * 100, 1),
        "accuracy": round(mt["holdout_acc"], 4),
        "baseline": round(mt["baseline"], 4),
        "edge": bool(mt["edge"]),
        "signal_accuracy": None if mt["signal_acc"] != mt["signal_acc"] else round(mt["signal_acc"], 4),
        "coverage": round(mt["coverage"], 3),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "data_age_days": int((pd.Timestamp.now().normalize() - as_of.normalize()).days),
        "trained_at": bundle["trained_at"],
        "history": [round(float(v), 2) for v in daily["Close"].iloc[-31:-1]],
    }


def predict_live(horizon):
    """Predict from fresh daily market data using the trained feature contract."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise EngineError(
            "التوقع الحي يحتاج مكتبة yfinance. ثبّتها من requirements.txt."
        ) from exc

    if int(horizon) not in HORIZONS:
        raise EngineError(f"المدة {horizon} غير مدرّبة. المتاح: {sorted(HORIZONS)}")

    live = yf.download(
        "GC=F",
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if live.empty:
        raise EngineError("تعذر جلب بيانات الذهب الحية حالياً.")

    if isinstance(live.columns, pd.MultiIndex):
        live.columns = live.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in live.columns]
    if missing:
        raise EngineError("بيانات السوق الحية ناقصة: " + ", ".join(missing))

    live = live.reset_index()
    date_column = "Date" if "Date" in live.columns else "Datetime"
    live = live.rename(columns={date_column: "Date"})
    live["Date"] = pd.to_datetime(live["Date"], utc=True).dt.tz_localize(None)
    live = live[required + ["Date"]].sort_values("Date").drop_duplicates("Date")
    for column in required:
        live[column] = pd.to_numeric(live[column], errors="coerce")
    live = live.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    if len(live) < WARMUP + max(HORIZONS):
        raise EngineError("بيانات السوق الحية غير كافية لحساب المؤشرات.")

    st = _state()
    model = st["bundle"]["models"][int(horizon)]
    features = make_features(live)
    x = features.iloc[[-1]][st["bundle"]["features"]]
    missing_features = [
        column for column in x.columns
        if column not in OPTIONAL and x[column].isna().any()
    ]
    if missing_features:
        raise EngineError(
            "بيانات السوق الحية غير كافية لبعض الميزات: "
            + ", ".join(missing_features[:4])
        )

    p_up = 0.5 * (
        float(model["clf"].predict_proba(x)[0, 1])
        + float(model["lr"].predict_proba(x)[0, 1])
    )
    pct = float(model["reg"].predict(x)[0])
    threshold = model["thr"]
    if p_up >= threshold and pct > 0:
        direction = "BUY"
    elif p_up <= 1 - threshold and pct < 0:
        direction = "SELL"
    else:
        direction = "HOLD"

    current = float(live["Close"].iloc[-1])
    metrics = model["metrics"]
    return {
        "status": "ok",
        "source": "Yahoo Finance GC=F",
        "live": True,
        "horizon": int(horizon),
        "current_price": round(current, 2),
        "target_price": round(current * (1 + pct / 100), 2),
        "predicted_pct_change": round(pct, 3),
        "predicted_direction": direction,
        "prob_up": round(p_up, 4),
        "confidence": round(max(p_up, 1 - p_up) * 100, 1),
        "accuracy": round(metrics["holdout_acc"], 4),
        "baseline": round(metrics["baseline"], 4),
        "edge": bool(metrics["edge"]),
        "as_of": live["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "trained_at": st["bundle"]["trained_at"],
        "history": [round(float(v), 2) for v in live["Close"].iloc[-31:-1]],
    }

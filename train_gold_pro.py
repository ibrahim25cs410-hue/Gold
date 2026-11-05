# -*- coding: utf-8 -*-
"""
train_gold_pro.py  -  تدريب احترافي لنماذج Aura Gold AI

ما الذي يميّزه عن النسخة القديمة؟
  1. لا تسريب من المستقبل: الميزات اليومية لا تُدمج بإغلاق اليوم نفسه.
  2. ميزات بلا مستوى سعري (نسب فقط) فلا تتجمّد النتائج عندما يتجاوز الذهب أعلى سعر شافه التدريب.
  3. تحقق Walk-Forward مع فاصل (gap) = مدة التوقع، ثم اختبار نهائي لم يُلمس أبداً.
  4. نماذج منتظمة بقوة (LightGBM + Logistic) مع اختيار عدد الأشجار على بيانات تحقق زمنية.
  5. تقرير صادق: المقارنة دائماً مع خط الأساس (أكثر اتجاه تكراراً) مع اختبار دلالة إحصائية.
  6. فحوصات "الرقم الثابت": تنبيه إذا النموذج يعطي نفس الإجابة تقريباً.
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gold_engine as ge

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GRID = [25, 50, 100, 150, 200, 300, 400, 600]      # أعداد الأشجار المرشّحة
LGB = dict(learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=60,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           random_state=42, verbose=-1)
THR_GRID = [0.52, 0.54, 0.56, 0.58, 0.60]
HOLDOUT = 0.15
N_SPLITS = 5


def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=1000))


def pick_iters(X, y_up, y_pct, gap):
    """يختار عدد الأشجار على آخر 20% زمنياً من بيانات التدريب (مع فاصل لمنع التداخل)."""
    cut = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:cut - gap], X.iloc[cut:]
    lo, hi = np.percentile(y_pct.iloc[:cut - gap], [1, 99])
    clf = LGBMClassifier(n_estimators=max(GRID), **LGB).fit(Xtr, y_up.iloc[:cut - gap])
    reg = LGBMRegressor(n_estimators=max(GRID), **LGB).fit(Xtr, np.clip(y_pct.iloc[:cut - gap], lo, hi))
    yu, yp = y_up.iloc[cut:], y_pct.iloc[cut:]
    ll = [log_loss(yu, np.clip(clf.predict_proba(Xva, num_iteration=k)[:, 1], 1e-6, 1 - 1e-6),
                   labels=[0, 1]) for k in GRID]
    mse = [float(np.mean((reg.predict(Xva, num_iteration=k) - yp) ** 2)) for k in GRID]
    return GRID[int(np.argmin(ll))], GRID[int(np.argmin(mse))]


def fit_all(X, y_up, y_pct, n_clf, n_reg):
    lo, hi = np.percentile(y_pct, [1, 99])
    clf = LGBMClassifier(n_estimators=n_clf, **LGB).fit(X, y_up)
    reg = LGBMRegressor(n_estimators=n_reg, **LGB).fit(X, np.clip(y_pct, lo, hi))
    lr = make_lr().fit(X, y_up)
    return clf, reg, lr


def predict_frame(models, X, y_up):
    clf, reg, lr = models
    return pd.DataFrame({"y": y_up.values,
                         "p": 0.5 * (clf.predict_proba(X)[:, 1] + lr.predict_proba(X)[:, 1]),
                         "pct": reg.predict(X)}, index=X.index)


def score(df, thr, horizon=1):
    y, p = df["y"].values, df["p"].values
    right = (p > 0.5) == (y == 1)
    base = max(y.mean(), 1 - y.mean())
    sig = ((p >= thr) & (df["pct"].values > 0)) | ((p <= 1 - thr) & (df["pct"].values < 0))
    n = len(df)
    acc = float(right.mean())
    n_eff = max(n / horizon, 1)   # العينات متداخلة زمنياً، فالعدد الفعلي المستقل = n / مدة التوقع
    return dict(acc=acc, baseline=float(base), n=n,
                z=float((acc - base) / np.sqrt(base * (1 - base) / n_eff)),
                coverage=float(sig.mean()), n_sig=int(sig.sum()),
                signal_acc=float(right[sig].mean()) if sig.sum() >= 20 else float("nan"),
                p_std=float(p.std()), frac_up=float((p > 0.5).mean()),
                pct_std=float(df["pct"].std()))


def choose_threshold(oof, horizon=1):
    best, best_acc = 0.55, -1
    for t in THR_GRID:
        s = score(oof, t, horizon)
        if s["coverage"] >= 0.20 and s["n_sig"] >= 50 and s["signal_acc"] > best_acc:
            best, best_acc = t, s["signal_acc"]
    return best


def train_horizon(h, daily, hourly):
    ds = ge.make_dataset(h, daily, hourly)
    features = [c for c in ds.columns if c != "y_pct"]
    X, y_pct = ds[features], ds["y_pct"]
    y_up = (y_pct > 0).astype(int)
    n = len(X)
    n_hold = int(n * HOLDOUT)
    dev = n - n_hold

    # 1) Walk-forward على بيانات التطوير فقط
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=h, test_size=max(60, dev // (N_SPLITS + 3)))
    folds, iters = [], []
    for tr, te in tscv.split(X.iloc[:dev]):
        kc, kr = pick_iters(X.iloc[tr], y_up.iloc[tr], y_pct.iloc[tr], h)
        models = fit_all(X.iloc[tr], y_up.iloc[tr], y_pct.iloc[tr], kc, kr)
        folds.append(predict_frame(models, X.iloc[te], y_up.iloc[te]))
        iters.append((kc, kr))
    oof = pd.concat(folds)
    thr = choose_threshold(oof, h)
    fold_acc = [float(((f["p"] > .5) == (f["y"] == 1)).mean()) for f in folds]

    # 2) اختبار نهائي لم يُستخدم في أي قرار (مع تنظيف آخر h صف من التدريب)
    kc, kr = int(np.median([i[0] for i in iters])), int(np.median([i[1] for i in iters]))
    tr_end = dev - h
    m_dev = fit_all(X.iloc[:tr_end], y_up.iloc[:tr_end], y_pct.iloc[:tr_end], kc, kr)
    hold = predict_frame(m_dev, X.iloc[dev:], y_up.iloc[dev:])
    hs = score(hold, thr, h)
    wf = score(oof, thr, h)

    # 3) النموذج النهائي على كل البيانات لاستخدامه فعلياً
    clf, reg, lr = fit_all(X, y_up, y_pct, kc, kr)

    top = sorted(zip(features, clf.feature_importances_), key=lambda t: -t[1])[:6]
    edge = hs["z"] >= 1.64 and np.mean(fold_acc) > wf["baseline"]

    print(f"\n── مدة {h} جلسة  |  عينات {n}  |  أشجار {kc}/{kr}  |  عتبة الثقة {thr:.2f}")
    print(f"   خط الأساس (الاتجاه الأكثر تكراراً): {hs['baseline']*100:.1f}%")
    print(f"   Walk-forward: {np.mean(fold_acc)*100:.1f}% ± {np.std(fold_acc)*100:.1f}  (folds: "
          + ", ".join(f"{a*100:.0f}" for a in fold_acc) + ")")
    print(f"   اختبار نهائي: {hs['acc']*100:.1f}%  (z = {hs['z']:+.2f})")
    sa = hs["signal_acc"]
    print(f"   الإشارات القوية: تغطية {hs['coverage']*100:.0f}% | دقة "
          + (f"{sa*100:.1f}%" if sa == sa else "عينات قليلة"))
    print(f"   فحص الثبات: تباين الاحتمال {hs['p_std']:.3f} | نسبة 'صعود' {hs['frac_up']*100:.0f}%")
    print("   أهم الميزات: " + ", ".join(f"{k}" for k, _ in top))
    if hs["p_std"] < 0.01 or hs["frac_up"] > 0.9 or hs["frac_up"] < 0.1:
        print("   ⚠ تحذير: النموذج يعطي تقريباً نفس الإجابة دائماً.")
    print("   الحكم: " + ("✅ يوجد أفضلية إحصائية على خط الأساس" if edge
                          else "⚠ لا توجد أفضلية مؤكدة على التخمين، تعامل مع الإشارة بحذر"))

    metrics = dict(holdout_acc=hs["acc"], baseline=hs["baseline"], z=hs["z"], wf_acc=float(np.mean(fold_acc)),
                   wf_std=float(np.std(fold_acc)), signal_acc=hs["signal_acc"], coverage=hs["coverage"],
                   edge=bool(edge), n=n)
    return dict(clf=clf, reg=reg, lr=lr, thr=thr, metrics=metrics), features


def main():
    t0 = time.time()
    print("⏳ تحميل البيانات…")
    daily, hourly = ge.load_daily(), ge.load_hourly()
    print(f"   يومي: {len(daily)} شمعة ({daily['Date'].iloc[0]:%Y-%m-%d} → {daily['Date'].iloc[-1]:%Y-%m-%d})"
          + (f" | ساعي: {len(hourly)}" if hourly is not None else " | الساعي غير موجود (اختياري)"))
    models, features = {}, None
    for h in ge.HORIZONS:
        models[h], features = train_horizon(h, daily, hourly)
    bundle = dict(version=1, trained_at=time.strftime("%Y-%m-%d %H:%M"),
                  data_end=str(daily["Date"].iloc[-1].date()), features=features, models=models)
    joblib.dump(bundle, ge.BUNDLE_PATH)
    print(f"\n✅ تم حفظ {ge.BUNDLE_PATH.name} خلال {time.time()-t0:.0f} ثانية")


if __name__ == "__main__":
    main()

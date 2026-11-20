"""Time-ordered forecasts, paired metrics and frozen experiment definitions."""
from functools import lru_cache

import numpy as np
import pandas as pd
import pandas_market_calendars as calendars
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from price_features import ECONOMIC_FEATURES, daily_features, hourly_features, realized_volatility

SEED = 42
FEATURES = [*ECONOMIC_FEATURES, "RSI_14", "Gold_EMA20_Distance", "month_sin",
            "month_cos", "dow_sin", "dow_cos", "is_turn_of_month", "nfp_event_week"]
ALPHA = 150.0


@lru_cache(maxsize=16)
def bar_clock(start, end, hourly):
    """Index = Yahoo bar label, values = scheduled close in UTC (naive UTC)."""
    schedule = calendars.get_calendar("GC").schedule(start, end)
    if not hourly:
        return schedule.market_close.dt.tz_localize(None)
    pieces = []
    for row in schedule.itertuples():
        labels = pd.date_range(row.market_open, row.market_close, freq="h", inclusive="left")
        ends = (labels + pd.Timedelta(hours=1)).to_series().clip(upper=row.market_close)
        pieces.append(pd.Series(ends.dt.tz_localize(None).to_numpy(), index=labels.tz_localize(None)))
    return pd.concat(pieces)


def clock_for(index, horizon):
    return bar_clock((index.min() - pd.Timedelta(days=4)).date(),
                     (index.max() + pd.Timedelta(days=70)).date(), horizon == 1)


def summarize(rows, horizon):
    if rows.empty:
        return {"n": 0}
    y, p, origin = (rows[k].to_numpy(float) for k in ("actual", "prediction", "origin_close"))
    e = p - y
    result = {"n": len(e), "mae": float(abs(e).mean()), "rmse": float(np.sqrt(np.mean(e**2))),
              "bias": float(e.mean()), "mape": float(np.mean(abs(e)/y)*100),
              "smape": float(np.mean(200*abs(e)/(abs(p)+abs(y)))),
              "direction_pct": float(np.mean(np.sign(p-origin)==np.sign(y-origin))*100),
              "baseline_mae": float(abs(y-origin).mean())}
    result["skill_pct"] = 100*(1-result["mae"]/result["baseline_mae"]) if result["baseline_mae"] else None
    for level in (80, 95):
        lo,hi=(rows[f"{side}{level}"].to_numpy(dtype=float,na_value=np.nan) for side in ("low","high"))
        available=np.isfinite(lo)&np.isfinite(hi)
        result[f"coverage{level}"] = float(((lo[available]<=y[available])&
                                             (y[available]<=hi[available])).mean()*100) if available.any() else None
        result[f"width{level}"] = float((hi[available]-lo[available]).mean()) if available.any() else None
    result["mae_delta_ci"] = paired_ci(abs(e)-abs(origin-y), horizon)
    return result


def paired_ci(delta, horizon):
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    block = max(horizon, int(np.ceil(n**(1/3))))
    if n < 3*block:
        return None
    rng = np.random.default_rng(SEED)
    starts = rng.integers(n, size=(2000, int(np.ceil(n/block))))
    indices = ((starts[:,:,None]+np.arange(block)) % n).reshape(len(starts), -1)[:,:n]
    return np.quantile(delta[indices].mean(axis=1), [.025,.975]).tolist()


def inputs(data,h):
    x = (hourly_features(data) if h==1 else daily_features(data))[FEATURES].ffill()
    scale = data.Close * realized_volatility(data.Close,h)
    return x, scale


def quantiles(errors):
    ordered = np.sort(np.abs(errors))
    if len(ordered)<20 or not np.isfinite(ordered).all():
        raise ValueError("Insufficient finite calibration observations")
    return {str(k):float(ordered[int(np.ceil((len(ordered)+1)*k/100))-1]) for k in (80,95)}


def simulate(data,h,outdir):
    x,scale = inputs(data,h)
    close = data.Close
    clock = clock_for(data.index,h)
    target = clock.index.to_series().shift(-h).reindex(data.index)
    eligible = np.flatnonzero((target.notna() & x.notna().all(axis=1) & scale.notna()).to_numpy())
    targets = pd.Series(close.reindex(pd.DatetimeIndex(target)).to_numpy(), index=data.index)
    normalized = (targets-close)/scale
    n = 200 if h==1 else 100
    end = len(data)-h
    hold = end-n
    cal = hold-h-100
    validation = cal-h-100
    folds = [validation-3*(n+h),validation-2*(n+h),validation-(n+h)]
    experiments = {}
    history = []
    def fit_at(start, strategy):
        train = eligible[(eligible < start-h) & (target.iloc[eligible] < data.index[start]).to_numpy()
                         & normalized.iloc[eligible].notna().to_numpy()]
        if strategy=="rolling":
            train = train[-(3000 if h==1 else 400):]
        if len(train)<100:
            raise ValueError("Insufficient training history")
        model = make_pipeline(StandardScaler(),Ridge(alpha=ALPHA))
        model.fit(x.iloc[train],normalized.iloc[train])
        return model,len(train)
    def issue(model,start,stop,q,tag):
        idx = eligible[(eligible>=start)&(eligible<stop)]
        if not len(idx):
            raise ValueError("No calendar-matched forecast origins")
        dollars = scale.iloc[idx]
        pred = close.iloc[idx]+dollars*model.predict(x.iloc[idx])
        issued = pd.DataFrame({"target":target.iloc[idx],"origin_close":close.iloc[idx],"prediction":pred})
        for k,v in q.items():
            issued[f"low{k}"],issued[f"high{k}"] = pred-v*dollars,pred+v*dollars
        issued["issued_asof_utc"]=clock.reindex(issued.index).to_numpy()
        issued["target_close_utc"]=clock.reindex(issued.target).to_numpy()
        issued.index.name="origin"
        issued.to_csv(outdir/f"{h}-{tag}.csv")  # immutable run directory, before outcome join
        # Advance the simulation clock over completed bars. A missing settlement stays unresolved.
        due = issued.reset_index().set_index("target")
        settlements = close.rename("actual").to_frame().join(due,how="inner")
        settled=settlements.reset_index(names="target").set_index("origin").sort_index()
        if not (settled.issued_asof_utc<settled.target_close_utc).all():
            raise ValueError("Invalid simulated forecast maturity")
        settled.to_csv(outdir/f"{h}-{tag}-settled.csv")
        return settled
    def calibrate(model,start,stop):
        rows=issue(model,start,stop,{},"cal-"+str(start)+"-"+strategy)
        rows=rows.loc[rows.target < data.index[stop+h]]
        return (quantiles((rows.actual-rows.prediction)/scale.loc[rows.index]),
                quantiles((rows.actual-rows.origin_close)/scale.loc[rows.index]))
    def persistence(rows,q):
        base=rows.assign(prediction=rows.origin_close)
        for k,v in q.items():
            base[f"low{k}"]=base.prediction-v*scale.loc[base.index]
            base[f"high{k}"]=base.prediction+v*scale.loc[base.index]
        return summarize(base,h)
    for strategy in ("expanding","rolling"):
        for fold,start in enumerate(folds):
            cutoff=start-h-100
            model,count=fit_at(cutoff,strategy)
            q,bq=calibrate(model,cutoff,start-h)
            rows=issue(model,start,start+n,q,f"{strategy}-fold{fold}")
            history.append(dict(strategy=strategy,fold=fold,train=count,first=str(rows.index[0]),
                                last=str(rows.index[-1]),baseline=persistence(rows,bq),**summarize(rows,h)))
        # Independent validation for model choice, followed by disjoint final calibration.
        vcal=validation-h-100
        model,count=fit_at(vcal,strategy)
        q,bq=calibrate(model,vcal,validation-h)
        rows=issue(model,validation,cal-h,q,strategy+"-validation")
        rows=rows.loc[rows.target < data.index[cal]]
        experiments[strategy]={**summarize(rows,h),"train":count,"baseline":persistence(rows,bq)}
    strategy=min(experiments,key=lambda s: experiments[s]["mae"])
    model,count=fit_at(validation,strategy)
    q,naiveq=calibrate(model,cal,hold-h)
    rows=issue(model,hold,end,q,"holdout")
    m=summarize(rows,h)
    calibration_count=len(pd.read_csv(outdir/f"{h}-cal-{cal}-{strategy}-settled.csv")
                          .query("target < @cut",local_dict={"cut":str(data.index[hold])}))
    meta={"kind":"normalized_ridge","model":model,"features":FEATURES,"conformal_intervals":q,
          "baseline_quantiles":naiveq,"metrics":{"mae_close_usd":m["mae"],"mape_close_pct":m["mape"]},
          "baseline_metrics":{"mae_close_usd":m["baseline_mae"]},"skill_score_pct":m["skill_pct"],
          "beats_baseline":m["mae"]<m["baseline_mae"],"range_coverage_80_pct":m["coverage80"],
          "range_coverage_95_pct":m["coverage95"]}
    return meta,dict(walk_forward=history,validation=experiments,selected=strategy,train=count,calibration=calibration_count,
                     holdout=m,baseline=persistence(rows,naiveq)),rows

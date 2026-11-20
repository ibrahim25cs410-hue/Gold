"""Strict market-only gold signal research and retrospective error diagnostics."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast_evaluation import clock_for, paired_ci, quantiles
from forecast_registry import ORIGINAL, database, matured, roles, sha
from market_intelligence import feature_groups
from research_phase2 import measure, split_rows

GROUPS = ("gold", "technical", "futures", "calendar")
CANDIDATES = [
    ("gold", "ridge", "normalized", 1, "short"),
    ("technical", "ridge", "normalized", 1, "short"),
    ("futures", "ridge", "normalized", 1, "short"),
    ("calendar", "ridge", "normalized", 1, "short"),
    ("full", "ridge", "normalized", 1, "short"),
    ("full", "ridge", "normalized", .25, "short"),
    ("full", "ridge", "normalized", .25, "long"),
    ("full", "ridge", "change", .25, "short"),
    ("full", "ridge", "return", .25, "short"),
    ("full", "ridge", "log", .25, "short"),
    ("full", "ridge", "level", 1, "short"),
    ("full", "elastic", "normalized", .25, "short"),
    ("full", "huber", "normalized", .25, "short"),
    ("full", "catboost", "normalized", .25, "short"),
    ("full", "hist_gradient", "normalized", .25, "short"),
    ("regime", "ridge", "normalized", .25, "short"),
]


def vintage_asof(origins, revisions):
    """Resolve latest observation and its known revision at each UTC origin."""
    rows = revisions.copy()
    for key in ("observed_at", "published_at"):
        rows[key] = pd.to_datetime(rows[key], utc=True)
    if (rows[["observed_at","published_at"]].isna().any().any()
        or (rows.published_at < rows.observed_at).any()
        or rows.duplicated(["observed_at","published_at"]).any()):
        raise ValueError("Invalid vintage observation/publication history")
    rows = rows.sort_values(["observed_at","published_at"])
    records = []
    for origin in pd.to_datetime(origins, utc=True):
        candidates = rows.loc[rows.published_at <= origin]
        records.append({"origin": origin, **(candidates.iloc[-1].to_dict() if len(candidates) else {})})
    return pd.DataFrame(records).set_index("origin")


def windows(design, target, actual, scale, cutoff, size):
    eligible = design.notna().all(axis=1) & actual.notna() & scale.notna() & target.lt(cutoff)
    previous = design.index[eligible]
    calibration = previous[-90:]
    train = previous[target.loc[previous] < calibration[0]][-size:]
    if len(calibration)<90 or len(train)<120:
        raise ValueError("Insufficient purged training or calibration")
    return train, calibration


def select(records):
    selection = [row for row in records if row["stage"] < 4 and row["status"] == "ok"]
    scores = pd.DataFrame(selection).groupby("model").mae.agg(["mean","count"])
    return scores.loc[scores["count"]==3,"mean"].idxmin()


def jump_flags(close):
    returns = np.log(close).diff()
    history = returns.shift().rolling(60).std()
    return returns.abs().gt(6*history) & history.gt(0)


def diagnose(frame, price, gold, train, horizon):
    origin = frame.index
    realized = frame.actual-frame.origin_close
    move = frame.prediction-frame.origin_close
    error = frame.prediction-frame.actual
    vol = gold.volatility
    low, high = vol.loc[train].quantile([1/3,2/3])
    trend = gold.filter(like="momentum").iloc[:,-1]
    down, up = trend.loc[train].quantile([1/3,2/3])
    move_cut = np.median(np.abs(move))
    tags = {
        "vol": np.select([vol.loc[origin]<low, vol.loc[origin]>high],["low","high"],default="middle"),
        "trend": np.select([trend.loc[origin]<down,trend.loc[origin]>up],["down","up"],default="sideways"),
        "month": origin.strftime("%Y-%m"),
        "session": np.select([origin.hour<7,origin.hour<13,origin.hour<21],
                             ["00–06 UTC","07–12 UTC","13–20 UTC"],default="21–23 UTC")
                   if horizon==1 else np.repeat("daily",len(origin)),
        "magnitude": np.where(move.abs()>move_cut,"above median","at or below median"),
        "sharp-target": np.where(realized.abs()>=realized.abs().quantile(.9),"top decile","other"),
        "origin-jump": np.where(jump_flags(price).loc[origin],"jump","other"),
    }
    breakdown = []
    for dimension, values in tags.items():
        block = frame.assign(group=values)
        for label, rows in block.groupby("group"):
            err = rows.prediction-rows.actual
            base = (rows.origin_close-rows.actual).abs().mean()
            breakdown.append(dict(dimension=dimension,label=label,n=len(rows),
                                  mae=float(err.abs().mean()),bias=float(err.mean()),
                                  baseline_mae=float(base)))
    worst = error.abs().nlargest(3).index
    return {"bins":breakdown,"worst":[{"origin":str(t),"target":str(frame.loc[t,"target"]),
                                       "error":float(error.loc[t])} for t in worst],
            "over_pct":float(error.gt(0).mean()*100),
            "move_correlation":float(move.corr(realized)) if move.std()>0 else None,
            "abs_error_price_correlation":float(error.abs().corr(frame.origin_close))}


def prediction(name, spec, design, train, origins, close, actual, scale, horizon):
    if spec is None:
        if name=="persistence":
            return close.loc[origins]
        if name=="mean20":
            return close.rolling(20).mean().loc[origins]
        return (close*np.exp(np.log(close).diff().rolling(20).mean()*horizon)).loc[origins]
    group, algorithm, transform, shrink, _ = spec
    included = GROUPS if group=="full" else (*GROUPS,"regime") if group=="regime" else ("gold",group)
    x = design.loc[:,design.columns.get_level_values(0).isin(included)]
    ys = {"normalized":(actual-close)/scale, "change":actual-close, "return":actual/close-1,
          "log":np.log(actual/close),"level":actual}
    if algorithm=="catboost":
        model = CatBoostRegressor(iterations=100, depth=3, loss_function="MAE",
                                  random_seed=42, thread_count=2, verbose=False,
                                  allow_writing_files=False)
    elif algorithm=="hist_gradient":
        model = HistGradientBoostingRegressor(max_iter=75,max_leaf_nodes=7,
                                             loss="absolute_error",random_state=42)
    else:
        estimator = {"ridge":Ridge(alpha=150), "elastic":ElasticNet(alpha=.035,l1_ratio=.25),
                     "huber":HuberRegressor(alpha=1,max_iter=500)}[algorithm]
        model = make_pipeline(StandardScaler(),estimator)
    model.fit(x.loc[train],ys[transform].loc[train])
    raw = pd.Series(model.predict(x.loc[origins]),index=origins)
    base = close.loc[origins]
    if transform=="log":
        converted = base*np.exp(raw)
    else:
        converted = {"normalized":base+raw*scale.loc[origins],"change":base+raw,
                     "return":base*(1+raw),"level":raw}[transform]
    if not np.isfinite(converted).all():
        raise ValueError("Non-finite forecast after target inversion")
    return base+shrink*(converted-base)


def experiment(data, horizon, output):
    groups = feature_groups(data,horizon,output)
    vetted = {key:groups[key] for key in GROUPS}
    gold = vetted["gold"]
    # Continuous interactions use only trailing realized volatility.
    vetted["regime"] = gold.filter(like="momentum").mul(gold.volatility,axis=0)
    design = pd.concat(vetted,axis=1).replace([np.inf,-np.inf],np.nan)
    target, actual, scale, stages = split_rows(data,design,horizon)
    target_clock = clock_for(data.index,horizon)
    origin_clock = target_clock.reindex(data.index)
    configs = {"persistence":None,"drift20":None,"mean20":None}
    configs.update({"_".join(map(str,s)):s for s in CANDIDATES})
    records, diagnostics, assessment, errors = [], {}, {}, []
    for stage, origins in enumerate(stages,1):
        names = configs if stage<4 else dict.fromkeys([select(records),"persistence","drift20","mean20"])
        for name in names:
            spec = configs[name]
            long = spec and spec[-1]=="long"
            size = (3000 if long else 750) if horizon==1 else (400 if long else 200)
            row = {"model":name,"stage":stage,"configuration":spec}
            try:
                train, cal = windows(design,target,actual,scale,origins[0],size)
                preds = prediction(name,spec,design,train,cal.append(origins),
                                   data.Close,actual,scale,horizon)
                bands = quantiles(((actual-preds)/scale).loc[cal])
                pred = preds.loc[origins]
                scores = measure(data.Close.loc[origins],actual.loc[origins],pred,
                                 scale.loc[origins],bands,horizon)
                row.update(scores,status="ok",train=[str(train[0]),str(train[-1]),len(train)],
                           calibration=[str(cal[0]),str(cal[-1]),len(cal)],
                           evaluation=[str(origins[0]),str(origins[-1]),len(origins)])
                error = (pred-actual.loc[origins]).abs()
                errors.append(pd.DataFrame({"stage":stage,"model":name,"error":error}))
                if stage==4:
                    frame = pd.DataFrame({"actual":actual.loc[origins],"prediction":pred,
                                          "origin_close":data.Close.loc[origins],"target":target.loc[origins],
                                          "issued_utc":origin_clock.loc[origins],
                                          "target_close_utc":target_clock.reindex(target.loc[origins]).to_numpy()})
                    assessment[name] = frame
                    diagnostics[name] = diagnose(frame,data.Close,gold,train,horizon)
            except Exception as exc:
                row.update(status="failed",error=repr(exc))
            records.append(row)
    winner = select(records)
    stats = pd.concat(errors).reset_index().pivot(index=["stage","Date"],columns="model",values="error")
    significance = {}
    for name in stats:
        baseline = ("gold_ridge_normalized_1_short" if name.split("_")[0] in GROUPS[1:] or
                    name=="full_ridge_normalized_1_short" else "persistence")
        deltas = (stats[name]-stats[baseline]).loc[1:3].dropna()
        significance[name] = {"reference":baseline,"mean_delta":float(deltas.mean()),
                              "ci":paired_ci(deltas,horizon)}
    pd.concat(assessment,names=["model","origin"]).to_csv(output/f"{horizon}-forecasts.csv")
    paired_assessment = {}
    if winner in assessment:
        for name, frame in assessment.items():
            delta = (assessment[winner].prediction-frame.actual).abs()-(frame.prediction-frame.actual).abs()
            paired_assessment[name] = paired_ci(delta,horizon)
    ready = design.notna().all(axis=1) & origin_clock.notna()
    return {"winner":winner,"records":records,"selection_delta":significance,
            "assessment_ci":paired_assessment,"diagnostics":diagnostics,
            "data_quality":{"rows":len(data),"unexpected_origins":int(origin_clock.isna().sum()),
                            "eligible_origins":int(ready.sum()),
                            "eligible_completed":int((ready & actual.notna()).sum()),
                            "jump_candidates":int(jump_flags(data.Close).sum()),
                            "zero_volume":int(data.Volume.eq(0).sum())}}


def previous_diagnostics(snapshot, phase2, output):
    original = joblib.load(ORIGINAL)
    report = {}
    for horizon in (1,7,30):
        data = read_market(snapshot,horizon)
        gold = feature_groups(data,horizon,output)["gold"]
        earlier = pd.read_csv(phase2/f"{horizon}-assessment.csv",index_col="origin",parse_dates=["origin"])
        model = json.loads((phase2/"experiments.json").read_text())["horizons"][str(horizon)]["winner"]
        earlier = earlier.assign(prediction=earlier[model],origin_close=data.Close.reindex(earlier.index))
        audit = original["models"][horizon]["holdout_audit"]
        prices = np.asarray(audit["actual_prices"])[:,2]
        legacy = np.asarray(audit["predicted_prices"])[:,2]
        base = np.asarray(audit["baseline_prices"])[:,2]
        report[str(horizon)] = {
            "original_archive":{"n":len(prices),"mae":float(np.abs(legacy-prices).mean()),
                                "baseline_mae":float(np.abs(base-prices).mean()),
                                "bias":float((legacy-prices).mean()),
                                "delta_ci":paired_ci(np.abs(legacy-prices)-np.abs(base-prices),horizon)},
            "phase2":diagnose(earlier,data.Close,gold,data.index[data.index<earlier.index[0]],horizon)}
        earlier[["prediction","actual","origin_close","target"]].to_csv(output/f"{horizon}-phase2-errors.csv")
    return report


def read_market(snapshot,horizon):
    name = "gold_full_dataset.csv" if horizon==1 else "gold_daily_dataset.csv"
    data = pd.read_csv(snapshot/name,index_col="Date",parse_dates=["Date"])
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Market candles must be sorted with unique keys")
    if not np.isfinite(data.Close).all() or not data.Close.gt(0).all():
        raise ValueError("Invalid gold closing prices")
    return data


def run(snapshot,phase2,output):
    output.mkdir(parents=True,exist_ok=False)
    report = {"historical_only":True,"inputs":{p.name:sha(p) for p in snapshot.glob("gold*.csv")},
              "phase2_hash":sha(phase2/"experiments.json"),"champion_archive_hash":sha(ORIGINAL),
              "code":{file:sha(Path(__file__).with_name(file)) for file in
                      ("research_phase3.py","research_phase2.py","market_intelligence.py","forecast_evaluation.py","price_features.py")},
              "previous":previous_diagnostics(snapshot,phase2,output),"horizons":{}}
    with database() as db:
        active,shadow = roles(db)
        report["registry"] = {"active":active,"shadow":shadow,
                              "matured_active":{h:len(matured(db,active,h)) for h in (1,7,30)}}
    for horizon in (1,7,30):
        result = experiment(read_market(snapshot,horizon),horizon,output)
        report["horizons"][horizon] = result
        print(horizon,result["winner"],flush=True)
    (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("snapshot","phase2","output"):
        parser.add_argument("--"+option,required=True,type=Path)
    run(**vars(parser.parse_args()))

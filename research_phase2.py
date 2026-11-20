"""Frozen Phase 2 research protocol. Production promotion uses the forward registry."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast_evaluation import clock_for, paired_ci, quantiles, summarize
from forecast_registry import sha
from market_intelligence import FRED_SERIES, feature_groups
from price_features import realized_volatility

OLD_HOLDOUT_START = {1: "2026-11-05 18:00", 7: "2026-06-17", 30: "2026-05-14"}
PROSPECTIVE_START = "2026-11-19T00:00:00Z"
UNAVAILABLE = {"maritime": "Unverified historical publication timestamps for local Baltic Dry file",
               "geopolitics": "No timestamped event archive; GVZ and oil shock tested as risk proxies",
               "scheduled_releases": "No archived as-of schedule or vintage release panel used",
               "contract_curve": "Continuous GC feed only; no verified contract history or open interest"}


def measure(close, actual, pred, scale, calibration, horizon):
    rows = pd.DataFrame({"actual": actual, "prediction": pred, "origin_close": close})
    for level, q in calibration.items():
        rows["low"+level], rows["high"+level] = pred-q*scale, pred+q*scale
    result = summarize(rows, horizon)
    errors = (actual-pred).abs()
    result.update(median_abs=float(errors.median()), worst_abs=float(errors.max()),
                  move_abs=float((pred-close).abs().mean()),
                  realized_move_abs=float((actual-close).abs().mean()))
    return result


def split_rows(data, design, horizon):
    clock = clock_for(data.index, horizon)
    target = clock.index.to_series().shift(-horizon).reindex(data.index)
    actual = data.Close.reindex(pd.DatetimeIndex(target)).set_axis(data.index)
    scale = data.Close * realized_volatility(data.Close, horizon)
    eligible = design.notna().all(axis=1) & actual.notna() & scale.notna()
    eligible &= target.lt(pd.Timestamp(OLD_HOLDOUT_START[horizon]))
    rows = data.index[eligible]
    size = 150 if horizon == 1 else 70
    stages = []
    for _ in range(4):
        stage = rows[-size:]
        if len(stage) != size:
            raise ValueError("Insufficient common chronological observations")
        stages.insert(0, stage)
        rows = rows[target.loc[rows] < stage[0]]
    return target, actual, scale, stages


def experiment(data, groups, horizon, output):
    close = data.Close
    design = pd.concat(groups, axis=1).replace([np.inf,-np.inf],np.nan)
    target, actual, scale, stages = split_rows(data, design, horizon)
    configs = [("persistence", (), "none", 0), ("drift20", (), "none", 1)]
    for group in groups:
        selected = ("gold",) if group == "gold" else ("gold",group)
        configs.append(("ridge_"+group, selected, "normalized", 1))
    full = tuple(groups)
    configs.extend((("ridge_full",full,"normalized",1),
                    ("ridge_log",full,"log",1),
                    ("ridge_change",full,"change",1),
                    ("ridge_shrunk",full,"normalized",.25),
                    ("elastic_full",full,"normalized",1),
                    ("catboost_full",full,"normalized",1)))
    records = []
    outcomes = {}
    for i, stage in enumerate(stages):
        # Last 90 eligible outcomes strictly before validation form calibration.
        # Purge both boundaries by actual scheduled target dates.
        earlier = design.index[(target < stage[0]) & actual.notna() &
                               design.notna().all(axis=1) & scale.notna()]
        calibration = earlier[-90:]
        training = earlier[target.loc[earlier] < calibration[0]]
        if len(training)<120:
            raise ValueError("Insufficient training before calibration")
        training = training[-(3000 if horizon==1 else 400):]
        if i==3:
            candidates = [cfg for cfg in configs if cfg[0] in (winner,"persistence","drift20")]
        else:
            candidates = configs
        for name, features, target_kind, shrink in candidates:
            row = dict(name=name, groups=features, target=target_kind, shrink=shrink,
                       stage=i+1, phase="historical_assessment" if i==3 else "selection",
                       train=[str(training[0]),str(training[-1]),len(training)],
                       calibration=[str(calibration[0]),str(calibration[-1]),len(calibration)],
                       evaluation=[str(stage[0]),str(stage[-1]),len(stage)])
            try:
                if name=="persistence":
                    prediction = close
                elif name=="drift20":
                    prediction = close * np.exp(np.log(close).diff().rolling(20).mean()*horizon)
                else:
                    x = design.loc[:, design.columns.get_level_values(0).isin(features)]
                    change = actual-close
                    y = {"normalized":change/scale, "log":np.log(actual/close),
                         "change":change}[target_kind]
                    if name.startswith("catboost"):
                        model = CatBoostRegressor(iterations=180, depth=4, learning_rate=.035,
                                                  random_seed=42, thread_count=2, verbose=False,
                                                  allow_writing_files=False)
                    else:
                        reg = (ElasticNet(alpha=.035,l1_ratio=.25,max_iter=5000)
                               if name.startswith("elastic") else Ridge(alpha=150))
                        model = make_pipeline(StandardScaler(),reg)
                    model.fit(x.loc[training],y.loc[training])
                    index = calibration.append(stage)
                    raw = shrink*pd.Series(model.predict(x.loc[index]),index=index)
                    prediction = {"normalized":close+raw*scale,
                                  "log":close*np.exp(raw), "change":close+raw}[target_kind]
                q = quantiles(((actual-prediction)/scale).loc[calibration])
                row.update(measure(close.loc[stage],actual.loc[stage],prediction.loc[stage],
                                   scale.loc[stage],q,horizon), status="ok")
                if i==3:
                    outcomes[name] = prediction.loc[stage]
            except Exception as exc:
                row.update(status="failed",error=repr(exc))
            records.append(row)
        if i==2:
            selection = pd.DataFrame(records)
            scores = selection.loc[selection.status=="ok"].groupby("name").agg(
                mae=("mae","mean"), folds=("mae","size"))
            winner = scores.loc[scores.folds==3].mae.sort_values().index[0]
    evidence = pd.DataFrame(outcomes).assign(actual=actual.loc[stages[-1]],
                                            target=target.loc[stages[-1]])
    evidence.to_csv(output/f"{horizon}-assessment.csv",index_label="origin")
    comparisons = {}
    for baseline in ("persistence","drift20"):
        if winner in outcomes and baseline in outcomes:
            delta = (outcomes[winner]-evidence.actual).abs()-(outcomes[baseline]-evidence.actual).abs()
            comparisons[baseline] = paired_ci(delta,horizon)
    return {"winner":winner,"experiment_log":records,"assessment_delta_ci":comparisons,
            "unmatched_origin_keys":int(clock_for(data.index,horizon).reindex(data.index).isna().sum())}


def run(snapshot, rates_folder, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    paths = [Path(snapshot)/name for name in ("gold_full_dataset.csv","gold_daily_dataset.csv")]
    paths += [Path(rates_folder)/f"{name}.csv" for name in FRED_SERIES]
    for path in paths:
        inputs[path.name] = sha(path) if path.exists() else None
    result = {"inputs":inputs,"prospective_start_utc":PROSPECTIVE_START,
              "prospective_status":"pending independent ledger evidence",
              "historical_boundaries":OLD_HOLDOUT_START,"excluded":UNAVAILABLE,"horizons":{},
              "code":{name:sha(Path(__file__).with_name(name)) for name in
                      ("research_phase2.py","market_intelligence.py","forecast_evaluation.py","price_features.py")}}
    for horizon in (1,7,30):
        path = paths[int(horizon!=1)]
        data = pd.read_csv(path,parse_dates=["Date"],index_col="Date")
        groups = feature_groups(data,horizon,rates_folder)
        result["horizons"][str(horizon)] = experiment(data,groups,horizon,output)
        (output/"experiments.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(horizon,result["horizons"][str(horizon)]["winner"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--rates",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.snapshot,args.rates,args.output)

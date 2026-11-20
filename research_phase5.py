"""Fixed-budget Federal Reserve signal ablations on common forecast targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_evaluation import clock_for, paired_ci, quantiles
from forecast_registry import sha
from market_intelligence import feature_groups
from policy_events import policy_features
from research_phase2 import measure, split_rows
from research_phase3 import prediction, read_market, select, windows

BASE = {"gold":("gold",), "technical":("gold","technical"), "futures":("gold","futures"),
        "market":("gold","technical","futures","calendar")}
POLICY = {"policy":("gold","policy"), "policy_vol":("gold","policy","policy_vol"),
          "full_policy":(*BASE["market"],"policy"), "full_policy_catboost":(*BASE["market"],"policy")}
CONTRASTS = {"policy":"gold","policy_vol":"gold","full_policy":"market",
             "full_policy_catboost":"market_catboost"}
HYPOTHESIS = ("After release, the observed FOMC decision and elapsed time add "
              "predictive information beyond causal gold features.")


def paired_bootstrap(folds,horizon,comparisons):
    """Independent resampling within folds; preserve serial dependence inside each."""
    rng = np.random.default_rng(42)
    draws = np.zeros(4000)
    n = sum(map(len,folds))
    for delta in folds:
        block = max(horizon,int(np.ceil(len(delta)**(1/3))))
        starts = rng.integers(len(delta),size=(4000,int(np.ceil(len(delta)/block))))
        ix = ((starts[:,:,None]+np.arange(block))%len(delta)).reshape(4000,-1)[:,:len(delta)]
        draws += np.asarray(delta)[ix].sum(axis=1)/n
    mean = np.concatenate(folds).mean()
    # Bonferroni simultaneous percentile intervals across the fixed hypothesis family.
    tail = .05/(2*comparisons)
    return {"mean":float(mean),"ci95":np.quantile(draws,[.025,.975]).tolist(),
            "family_ci":np.quantile(draws,[tail,1-tail]).tolist(),
            "fold_deltas":[float(np.mean(d)) for d in folds]}


def experiment(data,horizon,archive,output):
    market=feature_groups(data,horizon,output)
    clock=clock_for(data.index,horizon).reindex(data.index).dropna()
    policy=policy_features(pd.DatetimeIndex(clock),archive).set_axis(clock.index)
    groups={k:market[k] for k in BASE["market"]}
    groups["policy"]=policy
    groups["policy_vol"]=policy[["signed_impulse"]].mul(market["gold"].volatility,axis=0)
    design=pd.concat(groups,axis=1).replace([np.inf,-np.inf],np.nan)
    target,actual,scale,stages=split_rows(data,design,horizon)
    close=data.Close
    configs={**BASE,**POLICY,"market_catboost":BASE["market"],"persistence":(),"drift20":()}
    logs=[]
    errors={}
    assessments={}
    associations=[]
    for stage,origins in enumerate(stages,1):
        names=configs if stage<4 else dict.fromkeys([select(logs),"persistence","drift20"])
        train,cal=windows(design,target,actual,scale,origins[0],{1:3000,7:400,30:400}[horizon])
        if stage<4:
            outcome=np.log(actual.loc[train]/close.loc[train])
            associations.append({name:float(series.loc[train].corr(outcome,method="spearman"))
                                 if series.loc[train].nunique()>1 else None
                                 for name,series in policy.items()})
        for name in names:
            row={"model":name,"groups":configs[name],"stage":stage,
                 "training":[str(train.min()),str(train.max()),len(train)],
                 "calibration":[str(cal.min()),str(cal.max()),len(cal)],
                 "evaluation":[str(origins.min()),str(origins.max()),len(origins)]}
            try:
                alg="catboost" if name.endswith("catboost") else "ridge"
                spec=("full",alg,"normalized",.25,"long") if configs[name] else None
                # Remap the selected groups into one container for the shared regressor.
                features=design.loc[:,design.columns.get_level_values(0).isin(configs[name])]
                if spec:
                    features=pd.concat({"gold":features},axis=1)
                index=cal.append(origins)
                forecast=prediction(name,spec,features,train,index,close,actual,scale,horizon)
                q=quantiles(((actual-forecast)/scale).loc[cal])
                pred=forecast.loc[origins]
                row.update(measure(close.loc[origins],actual.loc[origins],pred,
                                   scale.loc[origins],q,horizon),status="ok",algorithm=alg if spec else name)
                errors[(stage,name)]=(pred-actual.loc[origins]).abs()
                if stage==4:
                    assessments[name]=pred
            except Exception as exc:
                row.update(status="failed",error=repr(exc))
            logs.append(row)
    comparisons={}
    for method,reference in CONTRASTS.items():
        keys=[((i,method),(i,reference)) for i in range(1,4)]
        if not all(a in errors and b in errors for a,b in keys):
            comparisons[method]={"decision":"defer: incomplete folds","reference":reference}
            continue
        delta=[(errors[a]-errors[b]).to_numpy() for a,b in keys]
        evidence=paired_bootstrap(delta,horizon,12)
        evidence.update(reference=reference,decision="retain for independent research" if
                        evidence["family_ci"][1]<0 and all(d<0 for d in evidence["fold_deltas"])
                        else "stop: incremental value not demonstrated")
        comparisons[method]=evidence
    winner=select(logs)
    final=stages[-1]
    frame=pd.DataFrame(assessments).assign(actual=actual.loc[final],target=target.loc[final])
    frame.to_csv(output/f"{horizon}-assessment.csv",index_label="origin")
    ci=paired_ci(errors[4,winner]-errors[4,"persistence"],horizon)
    market_ready=design.loc[:,design.columns.get_level_values(0).isin(BASE["market"])].notna().all(axis=1)
    return {"winner":winner,"records":logs,"incremental_policy_tests":comparisons,
            "training_rank_correlations":associations,"assessment_ci_vs_persistence":ci,
            "coverage":{"market_complete":int((market_ready & target.notna() & actual.notna()).sum()),
                        "common_complete":int((design.notna().all(axis=1)&target.notna()&actual.notna()).sum())}}


def run(snapshot,releases,output):
    output.mkdir(parents=True,exist_ok=False)
    archive=pd.read_csv(releases)
    for s in archive.loc[archive.status=="verified"].itertuples():
        file=releases.parent/s.source.rsplit("/",1)[1]
        if sha(file)!=s.sha256:
            raise ValueError(f"Statement hash mismatch: {file}")
    result={"hypothesis":HYPOTHESIS,"source_sha256":sha(releases),
            "inputs":{p.name:sha(p) for p in snapshot.glob("gold*.csv")},
            "code":{f:sha(Path(__file__).with_name(f)) for f in
                    ("research_phase5.py","policy_events.py","research_phase3.py","research_phase2.py",
                     "market_intelligence.py","forecast_evaluation.py","price_features.py")},
            "stopping_policy":{"selection_budget_per_horizon":33,"contrasts":12,
                               "retain":"upper simultaneous CI < 0 and lower MAE in all three folds",
                               "independent_evidence":"none; replay of examined history"},"horizons":{}}
    for horizon in (1,7,30):
        m=experiment(read_market(snapshot,horizon),horizon,archive,output)
        result["horizons"][horizon]=m
        print(horizon,m["winner"],flush=True)
    (output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot","releases","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    run(**vars(parser.parse_args()))

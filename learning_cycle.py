"""Run an expanding-history training cycle or the scheduled market observer."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

from build_dataset import DAILY_FILE, OUTPUT_FILE, download_daily, download_hourly, download_macro_closes
from forecast_evaluation import ALPHA, FEATURES, SEED, paired_ci, simulate, summarize
from forecast_registry import ROOT, artifact_path, database, event, home, load_version, now, paired, roles, sha
from price_features import BUNDLE_VERSION

HORIZONS=(1,7,30)


def validate(frame,h):
    high,low=("High","Low") if h==1 else ("Gold_High","Gold_Low")
    if frame.empty or frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Invalid/duplicate candle keys")
    prices=frame[["Open",high,low,"Close"]]
    if (not np.isfinite(prices).all().all() or (prices<=0).any().any()
        or (frame[high]<prices.max(axis=1)).any() or (frame[low]>prices.min(axis=1)).any()):
        raise ValueError("Invalid gold OHLC")
    macro=["GVZ_Close","VIX_Close","Oil_Close","TIP_Close",
           "DXY" if h==1 else "DXY_Close","US10Y_Yield" if h==1 else "US10Y_Close"]
    if not np.isfinite(frame[macro].iloc[-1]).all():
        raise ValueError("Missing latest macro inputs")


def merge_history(old,new,h):
    validate(old,h)
    validate(new,h)
    # Keep the first recorded vintage at overlapping timestamps, including vendor revisions.
    return pd.concat([old,new.loc[~new.index.isin(old.index)]]).sort_index()


def snapshot(folder,refresh):
    with database() as db:
        last=db.execute("SELECT path,hashes FROM snapshots ORDER BY at DESC LIMIT 1").fetchone()
        previous=db.execute("SELECT path FROM runs ORDER BY at DESC LIMIT 1").fetchone()
    macro=download_macro_closes() if refresh else None
    cutoff=pd.Timestamp.now(tz="UTC")
    frames=[]
    for h,base,get in ((1,OUTPUT_FILE,download_hourly),(7,DAILY_FILE,download_daily)):
        path=artifact_path(previous[0]).parent/base.name if previous else base
        if last:
            path=artifact_path(last["path"])/base.name
            if sha(path)!=json.loads(last["hashes"])[base.name]:
                raise ValueError("Archived dataset hash mismatch")
        frame=pd.read_csv(path,index_col="Date",parse_dates=["Date"],float_precision="round_trip")
        # Legacy historical files carry future labels for other tools. Never archive them here.
        frame=frame.drop(columns=[c for c in frame if c.startswith("Target_") or c=="Direction_Label"])
        validate(frame,h)
        if refresh:
            updates=get(macro,cutoff)
            if updates.empty or cutoff.tz_localize(None)-updates.index[-1]>pd.Timedelta(days=7):
                raise ValueError("Stale gold data returned during collection")
            frame=merge_history(frame,updates,h)
        frame.to_csv(folder/base.name,index_label="Date",float_format="%.17g")
        frames.append(pd.read_csv(folder/base.name,index_col="Date",parse_dates=["Date"],float_precision="round_trip"))
    hashes={p.name:sha(folder/p.name) for p in (DAILY_FILE,OUTPUT_FILE)}
    with database() as db:
        db.execute("INSERT INTO snapshots VALUES(?,?,?)",(str(now()),str(folder),json.dumps(hashes)))
    return frames


def compare_champion(meta,data,rows,h):
    from aura_price_engine import _forecast
    audit=meta.get("holdout_audit")
    lookup={t:i for i,t in enumerate(audit["dates"])} if audit else {}
    scored=[]
    for t,r in rows.iterrows():
        idx=lookup.get(t)
        if idx is not None and audit["target_dates"][idx]!=r.target:
            continue
        if idx is not None and audit["target_dates"][idx]==r.target and np.isclose(
            audit["baseline_prices"][idx,2],r.origin_close,rtol=0,atol=1e-7):
            price=audit["predicted_prices"][idx,2]
            bands=[audit[f"range_{level}_{side}"][idx] for level in (80,95) for side in ("low","high")]
        else:
            # The incumbent must already have finished independent calibration at this origin.
            permitted=meta["eligible_from"] if "eligible_from" in meta else audit["dates"][0]
            if t<pd.Timestamp(permitted):
                continue
            price,scale=_forecast(meta,data.loc[:t],h)
            bands=[price+sign*scale*meta["conformal_intervals"][str(k)] for k in (80,95) for sign in (-1,1)]
        scored.append([t,price,*bands])
    if not scored:
        return {"n":0}
    scored=pd.DataFrame(scored,columns=["origin","prediction","low80","high80","low95","high95"]).set_index("origin")
    matched=rows.loc[scored.index]
    champion=matched.assign(**{k:scored[k] for k in scored})
    return {"champion":summarize(champion,h),"candidate":summarize(matched,h),
            "mae_delta_ci":paired_ci(abs(matched.prediction-matched.actual)-abs(champion.prediction-matched.actual),h)}


def decide(db):
    active,shadow=roles(db)
    if not shadow:
        return
    evidence=[paired(db,active,shadow,h) for h in HORIZONS]
    if all(r["n"]>=max(100,10*h) for r,h in zip(evidence,HORIZONS)):
        action="promote" if all(r["pass"] for r in evidence) else "reject"
        if action=="promote":
            load_version(db,shadow)
        event(db,action,shadow,json.dumps(evidence))
        if action=="reject":
            event(db,"candidate",active,"Close rejected forward cohort")
        return action
    return "insufficient"


def train(refresh=True):
    with FileLock(str(home()/"training.lock"),timeout=0):
        version=now().strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
        folder=home()/version
        folder.mkdir()
        with database() as db:
            event(db,"training",version,"refresh="+str(refresh))
            active,_=roles(db)
        try:
            hourly,daily=snapshot(folder,refresh)
            hashes={f.name:sha(folder/f.name) for f in (DAILY_FILE,OUTPUT_FILE)}
            with database() as db:
                last=db.execute("SELECT report FROM runs ORDER BY at DESC LIMIT 1").fetchone()
                if refresh and last and json.loads(last["report"]).get("data_hashes")==hashes:
                    event(db,"skip",version,"No new eligible market rows since last successful training")
                    return None
                champion=load_version(db,active)
            report={"version":version,"compared_to":active,"horizons":{},"alpha":ALPHA,"seed":SEED,
                    "features":FEATURES,"trained":str(now()),"data_hashes":hashes}
            src=folder/"source"
            src.mkdir()
            for name in ("learning_cycle.py","forecast_registry.py","forecast_evaluation.py","aura_price_engine.py",
                         "price_features.py","build_dataset.py","main.py","train_price_forecaster.py"):
                shutil.copyfile(ROOT/name,src/name)
            report["code_sha256"]={p.name:sha(p) for p in src.iterdir()}
            git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True) if shutil.which("git") else None
            report["git_revision"]=git.stdout.strip() if git and git.returncode==0 else None
            report["python"]=sys.version
            (folder/"requirements.lock").write_text(subprocess.check_output(
                [sys.executable,"-m","pip","freeze"],text=True),encoding="utf-8")
            report["packages"]={n:importlib.metadata.version(n) for n in
                                ("numpy","pandas","catboost","scikit-learn","pandas_market_calendars","yfinance","joblib")}
            bundle={"version":BUNDLE_VERSION,"created_at":report["trained"],"models":{}}
            for h in HORIZONS:
                data=hourly if h==1 else daily
                meta,result,rows=simulate(data,h,folder)
                meta["eligible_from"]=rows.index[0]
                result["previous"]=compare_champion(champion["models"][h],data,rows,h)
                result["error_series"]={"dates":[str(v) for v in rows.target],
                                        "errors":(rows.prediction-rows.actual).tolist()}
                rows.to_csv(folder/f"{h}-holdout-scored.csv")
                bundle["models"][h]=meta
                report["horizons"][h]=result
                print(h,result["holdout"],flush=True)
            path=folder/"model.pkl"
            joblib.dump(bundle,path,compress=3)
            from aura_price_engine import _forecast
            for h,m in joblib.load(path)["models"].items():
                if not np.isfinite(_forecast(m,hourly if h==1 else daily,h)).all():
                    raise ValueError("Candidate inference failed")
            (folder/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            with database() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("INSERT INTO runs VALUES(?,?,?,?,?)",(version,str(now()),str(path),sha(path),json.dumps(report)))
                action=decide(db)
                current,shadow=roles(db)
                settled=db.execute("SELECT count(*) FROM outcomes JOIN forecasts USING(id) WHERE version=?",
                                   (shadow,)).fetchone()[0]
                if not shadow or not settled:
                    event(db,"candidate",version,"New prospective cohort; previous issues preserved")
                else:
                    event(db,"defer",version,"Existing prospective cohort still collecting paired outcomes")
                event(db,"retain",current,"Awaiting paired forward evidence" if action!="promote" else "Promotion recorded")
            return version
        except Exception:
            with database() as db:
                event(db,"failure",version,traceback.format_exc())
            raise


def observe():
    from aura_price_engine import predict
    failures=[]
    for h in HORIZONS:
        try:
            print(h,predict(h)["as_of"],flush=True)
        except Exception as exc:
            failures.append(f"{h}: {exc}")
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        decide(db)
    if failures:
        raise RuntimeError(" | ".join(failures))


def _worker_loop(stop):
    while not stop.is_set():
        try:
            observe()
        except Exception:
            with database() as db:
                event(db,"observation_failure","worker",traceback.format_exc())
        try:
            with database() as db:
                row=db.execute("SELECT max(at) FROM runs").fetchone()[0]
            from forecast_registry import performance
            age=now()-pd.Timestamp(row) if row else pd.Timedelta(days=8)
            drift=age>=pd.Timedelta(days=1) and any(
                r["drift"] for r in performance()["horizons"].values())
            if age>=pd.Timedelta(days=7) or drift:
                train()
        except Exception:
            with database() as db:
                event(db,"worker_failure","worker",traceback.format_exc())
        stop.wait(3600)


def worker(stop=None):
    from threading import Event
    from filelock import Timeout
    try:
        with FileLock(str(home()/"observer.lock"),timeout=0):
            _worker_loop(stop or Event())
    except Timeout:
        return


def rollback():
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        active,_=roles(db)
        older=db.execute("SELECT version FROM events WHERE action IN ('promote','rollback') "
                         "AND version!=? ORDER BY seq DESC LIMIT 1",(active,)).fetchone()
        version=older[0] if older else "original"
        load_version(db,version)
        event(db,"rollback",version,"Operator rollback from "+active)
        event(db,"candidate",version,"Reset shadow")


def replay(version):
    with database() as db:
        row=db.execute("SELECT path,report FROM runs WHERE id=?",(version,)).fetchone()
        if not row:
            raise ValueError("Unknown release")
    folder=artifact_path(row["path"]).parent
    report=json.loads(row["report"])
    for base,hashes in ((folder,report["data_hashes"]),(folder/"source",report["code_sha256"])):
        for name,digest in hashes.items():
            if sha(base/name)!=digest:
                raise ValueError("Replay source or data mismatch: "+name)
    program = """
from pathlib import Path
import sys,numpy as np,pandas as pd
from forecast_evaluation import simulate
folder=Path(sys.argv[1])
for h in (1,7,30):
    name="gold_full_dataset.csv" if h==1 else "gold_daily_dataset.csv"
    data=pd.read_csv(folder/name,index_col="Date",parse_dates=["Date"],float_precision="round_trip")
    _,_,rows=simulate(data,h,Path(sys.argv[2]))
    saved=pd.read_csv(folder/f"{h}-holdout-scored.csv",index_col="origin",parse_dates=["origin"])
    pd.testing.assert_index_equal(rows.index,saved.index)
    for key in ("prediction","actual","low80","high80","low95","high95"):
        np.testing.assert_allclose(rows[key],saved[key],rtol=0,atol=1e-7)
    print(h,len(rows),"replayed",flush=True)
"""
    with tempfile.TemporaryDirectory() as temp:
        subprocess.run([sys.executable,"-B","-c",program,str(folder.resolve()),temp],
                       cwd=folder/"source",check=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("command",choices=("train","observe","worker","rollback","replay"))
    cli.add_argument("--offline",action="store_true")
    cli.add_argument("--version")
    args=cli.parse_args()
    if args.command=="train":
        print(train(not args.offline))
    elif args.command=="replay":
        replay(args.version)
    else:
        {"observe":observe,"worker":worker,"rollback":rollback}[args.command]()

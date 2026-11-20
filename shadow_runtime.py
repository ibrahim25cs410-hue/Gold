"""Verify production checkpoints, activate candidates, and run a scheduled ledger tick."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from filelock import FileLock, Timeout
import numpy as np
import pandas as pd

from foundation_checkpoints import MODELS
from foundation_shadow import context_at_close,forecast_for,load_checkpoint
from forecast_evaluation import clock_for
from forecast_registry import database,event,foundation_forward,home,live_status,now,roles,sha


def state():
    output=home()/"foundation-live-r1"
    output.mkdir(exist_ok=True)
    return output


def put(path,payload):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf8")
    temporary.replace(path)


def smoke(snapshot, only=None):
    import torch
    import psutil

    torch.set_num_threads(2)
    measurements=[]
    entries={}
    for h,kind,name,key in ((1,"chronos","chronos2-zeroshot-hourly","gold_full_dataset.csv"),
                            (7,"timesfm","timesfm25-zeroshot-7session","gold_daily_dataset.csv"),
                            (30,"timesfm","timesfm25-zeroshot-30session","gold_daily_dataset.csv")):
        if only is not None and h!=only:
            continue
        repo,rev=MODELS[kind]
        path=home()/"foundation-checkpoints"/f"{kind}-{rev}"
        checkpoint=json.loads((path/"verified.json").read_text())
        assert (checkpoint["model_id"],checkpoint["revision"])==(repo,rev)
        data=pd.read_csv(snapshot/key,index_col="Date",parse_dates=["Date"])
        clock=clock_for(data.index,h)
        origin=data.index[-1]
        frame,clock=context_at_close(data,h,clock.loc[origin])
        target=clock.iloc[clock.index.get_loc(origin)+h]
        model=load_checkpoint(path,checkpoint["sha256"],kind)
        values=frame.Close.tail(256).to_numpy()
        start=time.perf_counter()
        left=forecast_for(model,kind,values,h)
        duration=time.perf_counter()-start
        right=forecast_for(model,kind,values,h)
        if not np.isfinite([v for v in left.values() if v is not None]).all() or left!=right:
            raise ValueError("Foundation inference is invalid or not reproducible")
        measurements.append({"name":name,"horizon":h,"model_id":repo,"revision":rev,
                             "origin":str(origin),"observation_close":str(clock.loc[origin]),
                             "target_close":str(target),"forecast":left,"inference_seconds":duration,
                             "rss_bytes":psutil.Process().memory_info().rss,
                             "evidence":"archived real input; engineering only, not forward evidence"})
        entries[str(h)]=[{"name":name,"kind":kind,"model_id":repo,"revision":rev,
                          "directory":str(path),"sha256":checkpoint["sha256"],"context":256,
                          "status":"SHADOW ONLY","registered_at":str(now()),
                          "audit_snapshot_sha256":sha(snapshot/key)}]
        print(name,left["prediction"],str(target),flush=True)
    target="smoke.json" if only is None else f"smoke-h{only}.json"
    put(state()/target,{"device":"cpu","versions":{n:version(n) for n in
        ("torch","transformers","chronos-forecasting","pandas_market_calendars","peft")},
        "measurements":measurements})
    put(state()/("candidates.json" if only is None else f"candidates-h{only}.json"),
        {"horizons":entries})


def audit_missed(db,previous,checked,entries):
    # The window closes at the next scheduled observation close. No late backfill.
    for h,models in entries.items():
        h=int(h)
        days=pd.DatetimeIndex([previous,checked]).normalize()
        clock=clock_for(days,h)
        for origin,deadline in clock.shift(-1).items():
            if pd.isna(deadline) or not previous<deadline<=checked:
                continue
            group=db.execute("""SELECT 1 FROM forecasts f JOIN forecast_members m ON
                               m.prediction_id=f.id WHERE origin=? AND h=? AND m.role=?""",
                               (str(origin),h,"foundation:"+models[0]["name"])).fetchone()
            detail=f"{h} origin {origin} closed at {deadline}"
            if not group and not db.execute(
                "SELECT 1 FROM events WHERE action='shadow_missed' AND version=? AND detail=?",
                (models[0]["name"],detail)).fetchone():
                event(db,"shadow_missed",models[0]["name"],detail)


def tick():
    from aura_price_engine import predict

    output=state()
    config=home()/"foundation-shadow.json"
    if not config.exists():
        raise RuntimeError("Foundation candidates are not activated")
    entries=json.loads(config.read_text())["horizons"]
    previous=output/"runtime.json"
    with FileLock(str(output/"tick.lock"),timeout=0):
        checked=now()
        attempt={}
        import torch
        torch.set_num_threads(2)
        for h in sorted(map(int,entries)):
            try:
                prediction=predict(h)
                attempt[str(h)]={k:prediction[k] for k in (
                    "forecast_id","forecast_group_id","as_of","target_close_utc","market_data_source")}
                origin=pd.to_datetime(prediction.get("as_of"),errors="coerce")
                if pd.notna(origin):
                    schedule=clock_for(pd.DatetimeIndex([origin]),h)
                    if origin not in schedule.index or checked>=schedule.iloc[schedule.index.get_loc(origin)+1]:
                        raise RuntimeError("Latest market observation is stale; no new group issued")
            except Exception as exc:
                attempt[str(h)]={"error":str(exc)}
                with database() as db:
                    event(db,"shadow_tick_failure",str(h),str(exc))
        with database() as db:
            if previous.exists():
                audit_missed(db,pd.Timestamp(json.loads(previous.read_text())["checked_utc"]),checked,entries)
            active,_=roles(db)
            counts={}
            for h in (1,7,30):
                counts[str(h)]=[dict(row) for row in db.execute(
                    """SELECT f.version, count(*) AS issued, count(o.actual) AS matured,
                    max(f.issued) AS last_issued_utc, max(f.target_close) AS latest_target_close_utc
                    FROM forecasts f JOIN forecast_members m ON m.prediction_id=f.id
                    LEFT JOIN outcomes o ON o.id=f.id
                    WHERE f.h=? AND m.role LIKE 'foundation:%' GROUP BY f.version""",(h,))]
            results={"checked_utc":str(checked),"champion":active,"attempt":attempt,
                     "candidates":counts,"paired":foundation_forward(db,active)}
        put(previous,results)


def activate(only=None):
    if only not in (None,1):
        raise ValueError("Partial activation is supported only for the independently checked hourly path")
    if only==1:
        cloud=json.loads((state()/"cloud-probe"/"probe.json").read_text())
        if cloud["status"]!="completed" or {r["horizon"] for r in cloud["metrics"]}!={1,7,30}:
            raise ValueError("Both exact models need to pass the CPU inference probe first")
    source=state()/("candidates.json" if only is None else "candidates-h1.json")
    proof=state()/("smoke.json" if only is None else "smoke-h1.json")
    if not proof.is_file():
        raise ValueError("Run the local inference smoke before activation")
    put(home()/"foundation-shadow.json",json.loads(source.read_text()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("smoke","activate","tick","status"))
    parser.add_argument("--snapshot",type=Path,default=Path("forecast_state/20261118T113211-48042bed"))
    parser.add_argument("--horizon",type=int,choices=(1,7,30))
    args=parser.parse_args()
    if args.command=="smoke":
        smoke(args.snapshot,args.horizon)
    elif args.command=="activate":
        activate(args.horizon)
    elif args.command=="status":
        print((state()/"runtime.json").read_text())
    else:
        try:
            tick()
        except Timeout:
            print("Skipping duplicate scheduler invocation")

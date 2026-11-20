"""Frozen 30-session policy research and an isolated forward-shadow bundle."""
import argparse
from fractions import Fraction
import html
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast_evaluation import clock_for, quantiles
from forecast_registry import home, sha
from market_intelligence import feature_groups
from policy_events import parse_statement, policy_features
from price_features import realized_volatility
from research_phase2 import measure, split_rows
from research_phase3 import prediction, read_market, windows
from research_phase5 import BASE, paired_bootstrap

PROTOCOL = Path(__file__).with_name("phase6_protocol.json")


def protocol():
    return json.loads(PROTOCOL.read_text())


def parse_range(payload):
    text = " ".join(html.unescape(re.sub(r"<[^>]+>"," ",payload.decode("utf8"))).split())
    text = re.sub("[‐‑–−]","-",text)
    number = r"(?:\d+-\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
    match = re.search(r"decided to (?:keep|maintain|raise|lower) the target range for the federal funds rate "
                      r"(?:at |(?:by [^.;]*? to )|to )("+number+r") to ("+number+r") percent",text,re.I)
    if not match:
        return None,None
    def fraction(value):
        return float(sum(Fraction(part) for part in value.split("-")))
    return tuple(fraction(value) for value in match.groups())


def audit(releases):
    manifest=pd.read_csv(releases)
    rows=[]
    for row in manifest.loc[manifest.status=="verified"].itertuples():
        path=releases.parent/row.source.rsplit("/",1)[1]
        payload=path.read_bytes()
        verified=parse_statement(row.source,payload)
        if any(verified[key]!=getattr(row,key) for key in ("sha256","published_at","action")):
            raise ValueError(f"Archived release mismatch: {path.name}")
        low,high=parse_range(payload)
        rows.append({**verified,"status":"verified","low":low,"high":high,
                     "retrieved_at":row.retrieved_at})
    frame=pd.DataFrame(rows).sort_values("published_at")
    frame["previous_low"]=frame.low.shift()
    frame["previous_high"]=frame.high.shift()
    frame["midpoint"]=(frame.low+frame.high)/2
    frame["change_bp"]=100*frame.midpoint.diff()
    frame["consistent"]=frame.change_bp.isna() | np.sign(frame.change_bp).eq(frame.action)
    return frame


def design_for(data, events, root):
    clock=clock_for(data.index,30).reindex(data.index).dropna()
    observed=pd.DatetimeIndex(clock)
    market=feature_groups(data,30,root)
    base={k:market[k] for k in BASE["market"]}
    policy=policy_features(observed,events).set_axis(clock.index)
    base["policy"]=policy
    base["policy_vol"]=policy[["signed_impulse"]].mul(market["gold"].volatility,axis=0)
    return pd.concat(base,axis=1).replace([np.inf,-np.inf],np.nan)


def for_groups(design, groups):
    return pd.concat({"gold":design.loc[:,design.columns.get_level_values(0).isin(groups)]},axis=1)


def event_sensitivity(delta, event_ids, moves):
    grouped=pd.DataFrame({"delta":delta,"event":event_ids,"move":moves}).dropna()
    retained=[float(grouped.loc[grouped.event!=event,"delta"].mean())
              for event in sorted(grouped.event.unique())]
    trimmed=grouped.loc[grouped.move<=grouped.move.quantile(.95),"delta"]
    return {"events":len(retained),"omitted_min":min(retained),"omitted_max":max(retained),
            "without_largest_moves":float(trimmed.mean())}


def run(snapshot,releases,output):
    output.mkdir(parents=True,exist_ok=False)
    old=json.loads((snapshot.parent/"phase5-r3"/"results.json").read_text())
    files={p.name:sha(p) for p in snapshot.glob("gold*.csv")}
    if old["inputs"]!=files or old["source_sha256"]!=sha(releases):
        raise ValueError("Phase 5 source hashes do not match")
    cfg=protocol()
    events=audit(releases)
    events.to_csv(output/"verified-events.csv",index=False)
    data=read_market(snapshot,30)
    design=design_for(data,events,output)
    clock=clock_for(data.index,30).reindex(data.index).dropna()
    delayed=events.assign(published_at=(pd.to_datetime(events.published_at,utc=True)
                                        +pd.Timedelta(days=7)).astype(str))
    delayed_design=design_for(data,delayed,output)
    event_keys=pd.Series(pd.to_datetime(events.published_at,utc=True).dt.tz_localize(None).array)
    eligible_event=np.searchsorted(event_keys,clock.to_numpy(),side="left")-1
    event_ids=pd.Series(eligible_event,index=clock.index)
    target,actual,scale,stages=split_rows(data,design,30)
    close=data.Close
    groups=cfg["candidates"]
    specifications={"market":(BASE["market"],"ridge",400),"gold":(("gold",),"ridge",400),
                    "A":(groups["A"],"ridge",400),"B":(groups["B"],"ridge",400),
                    "delayed":(groups["A"],"ridge",400),"short":(groups["A"],"ridge",250),
                    "rates":((*groups["A"],"rates"),"ridge",400),
                    "elastic":(groups["A"],"elastic",400),"catboost":(groups["A"],"catboost",400),
                    "persistence":((),"",400),"drift20":((),"",400)}
    rates=pd.merge_asof(pd.DataFrame({"clock":pd.to_datetime(clock.array,utc=True)}),
                         events.assign(published_at=pd.to_datetime(events.published_at,utc=True)),
                         left_on="clock",right_on="published_at",allow_exact_matches=False)
    rates=rates[["midpoint","change_bp"]].set_axis(clock.index)
    # Missing previous level excludes the row from this optional predictor; use a
    # conservative first-observation zero only when the statement left rates unchanged.
    first=events.iloc[0]
    if first.action==0:
        rates["change_bp"]=rates.change_bp.fillna(0)
    rate_design=pd.concat([design,pd.concat({"rates":rates},axis=1)],axis=1)
    records=[]
    errors={}
    frames=[]
    for stage,origins in enumerate(stages,1):
        names=specifications if stage<4 else ("A","B","market","gold","persistence","drift20")
        for name in names:
            gs,algorithm,size=specifications[name]
            features=delayed_design if name=="delayed" else rate_design if name=="rates" else design
            train,cal=windows(features,target,actual,scale,origins[0],size)
            spec=("full",algorithm,"normalized",cfg["shrink"],"long") if gs else None
            x=for_groups(features,gs) if gs else features
            idx=cal.append(origins)
            pred=prediction(name,spec,x,train,idx,close,actual,scale,30)
            q=quantiles(((actual-pred)/scale).loc[cal])
            point=pred.loc[origins]
            result=measure(close.loc[origins],actual.loc[origins],point,scale.loc[origins],q,30)
            records.append(dict(model=name,stage=stage,features=list(gs),algorithm=algorithm,
                                train=[str(train.min()),str(train.max()),len(train)],
                                period=[str(origins.min()),str(origins.max())],**result))
            errors[stage,name]=(point-actual.loc[origins]).abs()
            frames.append(pd.DataFrame({"stage":stage,"model":name,"prediction":point,
                                        "actual":actual.loc[origins],"target":target.loc[origins]}))
    pd.concat(frames).to_csv(output/"forecasts.csv",index_label="origin")
    checks={}
    for name,ref in (("A","market"),("B","gold"),("A","delayed"),("short","market"),
                     ("rates","A"),("elastic","market"),("catboost","market")):
        parts=[errors[i,name]-errors[i,ref] for i in (1,2,3)]
        result=paired_bootstrap([x.to_numpy() for x in parts],30,cfg["family_size"])
        delta=pd.concat(parts)
        moves=(actual-close).abs().reindex(delta.index)
        checks[name+" vs "+ref]={**result,**event_sensitivity(delta,event_ids.reindex(delta.index),moves)}
    payload={"config_sha256":sha(PROTOCOL),"inputs":files,"archive_sha256":sha(releases),
             "prior_results_sha256":sha(snapshot.parent/"phase5-r3"/"results.json"),
             "records":records,"robustness":checks,"independent":"unavailable"}
    (output/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf8")
    print("Audited decisions:",len(events),"parsed ranges:",int(events.low.notna().sum()),flush=True)


def freeze(snapshot,releases,output,expires):
    """Fit A and B once using purged, completed history. Never registers a promotion."""
    output.mkdir(parents=True,exist_ok=False)
    events=audit(releases)
    data=read_market(snapshot,30)
    design=design_for(data,events,output)
    clock=clock_for(data.index,30)
    target=clock.index.to_series().shift(-30).reindex(data.index)
    actual=data.Close.reindex(pd.DatetimeIndex(target)).set_axis(data.index)
    scale=data.Close*realized_volatility(data.Close,30)
    cfg=protocol()
    train,cal=windows(design,target,actual,scale,data.index[-1],cfg["window"])
    models={}
    y=(actual-data.Close)/scale
    for name,groups in cfg["candidates"].items():
        x=for_groups(design,groups)
        model=make_pipeline(StandardScaler(),Ridge(alpha=cfg["alpha"]))
        model.fit(x.loc[train],y.loc[train])
        pred=.25*model.predict(x.loc[cal])
        models[name]={"model":model,"groups":groups,"q":quantiles(y.loc[cal]-pred)}
    bundle=output/"models.pkl"
    joblib.dump({"models":models,"events":events,"shrink":cfg["shrink"]},bundle)
    manifest={"artifact":sha(bundle),"protocol":sha(PROTOCOL),"dataset":sha(snapshot/"gold_daily_dataset.csv"),
              "archive":sha(releases),"cutoff":str(data.index[-1]),"expires":expires,
              "train":str(train[-1]),"calibration":str(cal[-1])}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")


def refresh_events(folder,releases,expires):
    """Extend the audited event vintage without refitting either frozen model."""
    model_manifest=json.loads((folder/"manifest.json").read_text(encoding="utf8"))
    path=folder/"models.pkl"
    if sha(path)!=model_manifest["artifact"] or sha(PROTOCOL)!=model_manifest["protocol"]:
        raise ValueError("Research shadow artifact failed integrity check")
    latest=event_vintage(folder,model_manifest,joblib.load(path)["events"])
    current=audit(releases)
    keys=["published_at","action","low","high"]
    pd.testing.assert_frame_equal(latest[keys].reset_index(drop=True),
                                  current[keys].iloc[:len(latest)].reset_index(drop=True))
    if pd.Timestamp(expires)<=pd.Timestamp(model_manifest["expires"]):
        raise ValueError("Archive renewal must extend the eligibility boundary")
    payload=json.dumps({"expires":expires,"archive":sha(releases),
                        "events":current.to_dict("records")},default=str,sort_keys=True)
    import hashlib
    digest=hashlib.sha256(payload.encode()).hexdigest()
    with (folder/f"events-{digest}.json").open("x",encoding="utf8") as stream:
        stream.write(payload)


def event_vintage(folder,manifest,events):
    for path in folder.glob("events-*.json"):
        if path.stem!="events-"+sha(path):
            raise ValueError(f"Invalid event vintage: {path.name}")
        vintage=json.loads(path.read_text(encoding="utf8"))
        if pd.Timestamp(vintage["expires"])>pd.Timestamp(manifest["expires"]):
            manifest.update(expires=vintage["expires"],archive=vintage["archive"])
            events=pd.DataFrame(vintage["events"])
    return events


def shadow_members(data, asof):
    folder=home()/"phase6-shadow"
    manifest_path=folder/"manifest.json"
    if not manifest_path.is_file():
        return []
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    origin=data.index[-1]
    if origin<=pd.Timestamp(manifest["cutoff"]):
        return []
    if pd.Timestamp(asof)>=pd.Timestamp(manifest["expires"]) and not any(folder.glob("events-*.json")):
        return []
    path=folder/"models.pkl"
    if sha(path)!=manifest["artifact"] or sha(PROTOCOL)!=manifest["protocol"]:
        raise ValueError("Research shadow artifact failed integrity check")
    bundle=joblib.load(path)
    events=event_vintage(folder,manifest,bundle["events"])
    if pd.Timestamp(asof)>=pd.Timestamp(manifest["expires"]):
        return []
    design=design_for(data,events,folder)
    scale=float((data.Close*realized_volatility(data.Close,30)).iloc[-1])
    members=[]
    for name,config in bundle["models"].items():
        x=for_groups(design,config["groups"]).loc[[origin]]
        point=float(data.Close.iloc[-1]+scale*bundle["shrink"]*config["model"].predict(x)[0])
        if not np.isfinite([point,scale]).all():
            raise ValueError("Invalid policy shadow forecast")
        members.append((f"policy_{name}",f"phase6-{name}-{manifest['artifact'][:12]}",
                        manifest,point,scale,config["q"]))
    return members


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path)
    for name in ("releases","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--freeze-until")
    parser.add_argument("--refresh-events-until")
    args=vars(parser.parse_args())
    expiry=args.pop("freeze_until")
    refresh=args.pop("refresh_events_until")
    if refresh and not expiry:
        refresh_events(args["output"],args["releases"],refresh)
    elif args["snapshot"] is None or refresh:
        parser.error("Select a snapshot for research or freezing; event refresh is a separate operation")
    elif expiry:
        freeze(**args,expires=expiry)
    else:
        run(**args)

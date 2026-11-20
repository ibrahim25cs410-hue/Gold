"""Durable append-only issued forecasts, outcomes, run history and deployment events."""
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast_evaluation import clock_for, paired_ci, summarize

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/"aura_price_models.pkl"
INSUFFICIENT="Insufficient Evidence — بيانات الاختبار غير كافية لإثبات تحسن النموذج."


def home():
    folder=Path(os.environ.get("AURA_STATE_DIR",ROOT/"forecast_state")).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    return folder


def artifact_path(recorded_path):
    """Resolve migrated paths without modifying historical SQLite records."""
    folder=home()
    manifest=folder/"export-manifest.json"
    if not manifest.is_file():
        return Path(recorded_path)
    with manifest.open(encoding="utf8") as file:
        mapped=json.load(file).get("historical_path_map",{}).get(recorded_path)
    if mapped is None:
        return Path(recorded_path)
    destination=Path(mapped).resolve()
    if not destination.is_relative_to(folder):
        raise ValueError("Migrated audit artifact path escapes state directory")
    return destination


def now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


@contextmanager
def database():
    db=sqlite3.connect(home()/"audit.sqlite3",timeout=30)
    db.row_factory=sqlite3.Row
    try:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots(at TEXT PRIMARY KEY,path TEXT,hashes TEXT);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, at TEXT, path TEXT, digest TEXT, report TEXT);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, at TEXT, action TEXT, version TEXT, detail TEXT);
        CREATE TABLE IF NOT EXISTS forecasts(id TEXT PRIMARY KEY, issued TEXT, origin TEXT, h INTEGER,
          version TEXT, target TEXT, target_close TEXT, input_hash TEXT, config TEXT, origin_close REAL,
          prediction REAL, low80 REAL, high80 REAL, low95 REAL, high95 REAL, UNIQUE(origin,h,version));
        CREATE TABLE IF NOT EXISTS outcomes(id TEXT PRIMARY KEY REFERENCES forecasts(id), observed TEXT, actual REAL);
        CREATE TABLE IF NOT EXISTS forecast_groups(id TEXT PRIMARY KEY, instrument TEXT NOT NULL,
          observation_close TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS forecast_members(prediction_id TEXT PRIMARY KEY REFERENCES forecasts(id),
          group_id TEXT NOT NULL REFERENCES forecast_groups(id), role TEXT NOT NULL, UNIQUE(group_id,role));
        """)
        for table in ("forecasts","outcomes","runs","events","snapshots","forecast_groups","forecast_members"):
            for op in ("UPDATE","DELETE"):
                db.execute(f"CREATE TRIGGER IF NOT EXISTS lock_{table}_{op} BEFORE {op} ON {table} "
                           "BEGIN SELECT RAISE(ABORT,'immutable audit record'); END")
        with db:
            yield db
    finally:
        db.close()


def event(db, action, version, detail):
    db.execute("INSERT INTO events(at,action,version,detail) VALUES(?,?,?,?)",
               (str(now()),action,version,detail))


def roles(db):
    def latest(actions):
        return db.execute("SELECT version FROM events WHERE action IN ("+
                          ",".join("?" for _ in actions)+") ORDER BY seq DESC LIMIT 1",actions).fetchone()
    active=latest(("promote","rollback"))
    active=active[0] if active else "original"
    shadow=latest(("candidate",))
    return active,shadow[0] if shadow and shadow[0]!=active else None


def load_version(db, version):
    if version=="original":
        return joblib.load(ORIGINAL)
    row=db.execute("SELECT path,digest FROM runs WHERE id=?",(version,)).fetchone()
    if not row:
        raise ValueError("Model artifact integrity check failed")
    path=artifact_path(row["path"])
    if sha(path)!=row["digest"]:
        raise ValueError("Model artifact integrity check failed")
    return joblib.load(path)


def settle(db,data,h,asof):
    rows=db.execute("SELECT f.* FROM forecasts f LEFT JOIN outcomes o ON o.id=f.id "
                    "WHERE f.h=? AND o.id IS NULL AND f.target_close<=?",(h,str(asof))).fetchall()
    for row in rows:
        target=pd.Timestamp(row["target"])
        if target in data.index and pd.Timestamp(row["issued"])<pd.Timestamp(row["target_close"])<=asof:
            actual=float(data.loc[target,"Close"])
            if np.isfinite(actual) and actual>0:
                db.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?)",(row["id"],str(asof),actual))


def record(db,data,h,version,config,pred,scale,q,asof):
    origin=data.index[-1]
    clock=clock_for(pd.DatetimeIndex([origin]),h)
    if origin not in clock.index:
        return None
    pos=clock.index.get_loc(origin)
    target=clock.index[pos+h]
    deadline=clock.iloc[pos+h]
    existing=db.execute("SELECT * FROM forecasts WHERE origin=? AND h=? AND version=?",
                        (str(origin),h,version)).fetchone()
    if existing:
        return dict(existing)
    if not clock.iloc[pos]<=asof<min(deadline,clock.iloc[pos+1]):
        return None
    inputs=data.drop(columns=[c for c in data if c.startswith("Target_") or c=="Direction_Label"])
    digest=hashlib.sha256(pd.util.hash_pandas_object(inputs,index=True).values.tobytes()).hexdigest()
    fid=uuid.uuid4().hex
    bands=[pred+sign*scale*q[level] if q[level] is not None else None
           for level in ("80","95") for sign in (-1,1)]
    values=(fid,str(asof),str(origin),h,version,str(target),str(deadline),digest,json.dumps(config),
            float(data.Close.iloc[-1]),float(pred),*bands)
    db.execute("INSERT OR IGNORE INTO forecasts VALUES("+",".join("?" for _ in values)+")",values)
    return dict(db.execute("SELECT * FROM forecasts WHERE origin=? AND h=? AND version=?",
                           (str(origin),h,version)).fetchone())


def matured(db,version,h):
    return pd.read_sql_query("SELECT f.*,o.actual,m.group_id FROM forecasts f JOIN outcomes o USING(id) "
                             "LEFT JOIN forecast_members m ON m.prediction_id=f.id "
                             "WHERE version=? AND h=? ORDER BY origin",db,params=(version,h))


def issue_group(db,data,h,models,asof):
    """Persist one matched group or reuse the champion's immutable prior issuance."""
    origin = str(data.index[-1])
    active = models[0][1]
    existing = db.execute("SELECT f.*,m.group_id FROM forecasts f LEFT JOIN forecast_members m "
                          "ON m.prediction_id=f.id WHERE origin=? AND h=? AND version=?",
                          (origin,h,active)).fetchone()
    if existing:
        return dict(existing)
    group_id = uuid.uuid4().hex
    saved = None
    for role,version,config,point,scale,bands in models:
        entry = record(db,data,h,version,config,point,scale,bands,asof)
        if role=="champion":
            if not entry:
                return None
            saved = entry
            clock = clock_for(data.index[-1:],h)
            db.execute("INSERT INTO forecast_groups VALUES(?,?,?)",
                       (group_id,"GC=F",str(clock.loc[data.index[-1]])))
        if entry and entry["issued"]==str(asof):
            db.execute("INSERT INTO forecast_members VALUES(?,?,?)",(entry["id"],group_id,role))
    return {**saved,"group_id":group_id}


def live_status(db,h):
    rows = db.execute("""SELECT f.*,m.role,m.group_id,o.actual,o.observed,
                        f.prediction-o.actual AS error,g.instrument,g.observation_close,
                        CASE WHEN o.actual IS NULL THEN 'pending' ELSE 'completed' END AS status
                        FROM forecasts f JOIN forecast_members m ON m.prediction_id=f.id
                        JOIN forecast_groups g ON g.id=m.group_id LEFT JOIN outcomes o ON o.id=f.id
                        WHERE f.h=? ORDER BY f.issued DESC,f.id""",(h,)).fetchall()
    completed = [r for r in rows if r["actual"] is not None]
    metrics = {}
    if completed:
        frame = pd.DataFrame([dict(r) for r in completed]).sort_values("origin")
        metrics = {version:summarize(part,h) for version,part in frame.groupby("version")}
    return {"predictions":len(rows),"pending":len(rows)-len(completed),"completed":len(completed),
            "groups":len({r["group_id"] for r in rows}),
            "models":metrics,
            "latest":next((dict(r) for r in rows if r["role"]=="champion"),None),
            "latest_target":max((r["target_close"] for r in completed),default=None)}


def policy_forward(db,active):
    versions = db.execute("""SELECT DISTINCT m.role,f.version FROM forecast_members m
                            JOIN forecasts f ON f.id=m.prediction_id
                            WHERE f.h=30 AND m.role IN ('policy_A','policy_B')""").fetchall()
    result={}
    for row in versions:
        comparison=paired(db,active,row["version"],30)
        comparison.pop("pass",None)
        result[row["version"]]={"candidate":row["role"],"comparison":comparison,
                                "eligible_for_promotion":False}
    return result


def foundation_forward(db,active):
    versions=db.execute("""SELECT f.h,f.version,count(*) AS issued,count(o.actual) AS matured,
                          max(f.issued) AS last_issued_utc
                          FROM forecast_members m JOIN forecasts f ON f.id=m.prediction_id
                          LEFT JOIN outcomes o ON o.id=f.id
                          WHERE m.role LIKE 'foundation:%' GROUP BY f.h,f.version""").fetchall()
    result={}
    for row in versions:
        comparison=paired(db,active,row["version"],row["h"])
        comparison.pop("pass",None)
        comparison.update(issued=row["issued"],pending=row["issued"]-row["matured"],
                          matured=row["matured"],last_issued_utc=row["last_issued_utc"])
        result.setdefault(row["h"],{})[row["version"]]=comparison
    return result


def paired(db,active,shadow,h):
    a=matured(db,active,h)
    b=matured(db,shadow,h)
    keys=["issued","origin","target","target_close","input_hash","actual","origin_close"]
    if "group_id" in a and "group_id" in b:
        keys.append("group_id")
        a,b=a.dropna(subset=["group_id"]),b.dropna(subset=["group_id"])
    if a.empty or b.empty:
        return {"n":0,"pass":False}
    pairs=b.merge(a,on=keys,suffixes=("","_active"))
    if pairs.empty:
        return {"n":0,"pass":False}
    benchmarks = {}
    if "group_id" in keys:
        score_columns = ("prediction","low80","high80","low95","high95")
        for role,version in (("persistence","baseline:persistence:"),
                             ("drift","baseline:drift20:")):
            reference = matured(db,version+active,h)
            if reference.empty:
                return {"n":0,"pass":False}
            pairs = pairs.merge(reference[keys+list(score_columns)].rename(
                columns={k:f"{role}_{k}" for k in score_columns}),on=keys,validate="one_to_one")
        pairs = pairs.loc[pairs.persistence_prediction==pairs.origin_close]
        if pairs.empty:
            return {"n":0,"pass":False}
        pairs = pairs.sort_values("target_close")
        for role in ("persistence","drift"):
            benchmarks[role] = summarize(pairs.assign(**{k:pairs[f"{role}_{k}"]
                                                         for k in score_columns}),h)
    candidate=summarize(pairs,h)
    champion=summarize(pairs.assign(**{k:pairs[k+"_active"] for k in
                                      ("prediction","low80","high80","low95","high95")}),h)
    ci=paired_ci(abs(pairs.prediction-pairs.actual)-abs(pairs.prediction_active-pairs.actual),h)
    passed=(len(pairs)>=max(100,10*h) and ci and ci[1]<0 and candidate["mae_delta_ci"][1]<0
            and candidate["mae"]<=.99*champion["mae"] and candidate["rmse"]<=champion["rmse"]
            and abs(candidate["bias"])<=abs(champion["bias"])+.05*champion["mae"]
            and candidate["direction_pct"]>=champion["direction_pct"]-3
            and all(candidate[f"coverage{k}"] is not None and champion[f"width{k}"] is not None
                    and candidate[f"coverage{k}"]>=k-5 and candidate[f"width{k}"]<=1.1*champion[f"width{k}"]
                    for k in (80,95)))
    return {"n":len(pairs),"pass":bool(passed),"candidate":candidate,"champion":champion,
            "paired_ci":ci,**benchmarks}


def performance():
    with database() as db:
        active,shadow=roles(db)
        run=db.execute("SELECT id,at,report FROM runs ORDER BY at DESC LIMIT 1").fetchone()
        report=json.loads(run["report"]) if run else {}
        horizons={}
        for h in (1,7,30):
            rows=matured(db,active,h)
            recent=rows.tail(50)
            prior=rows.iloc[-100:-50]
            drift=None
            if len(prior)==50:
                drift=summarize(recent,h)["mae"]>1.25*summarize(prior,h)["mae"]
            horizons[h]={"forward":summarize(rows,h),"drift":drift,
                         "errors":(rows.prediction-rows.actual).tail(100).tolist(),
                         "dates":rows.target_close.tail(100).tolist(),
                         "baseline":summarize(matured(db,"baseline:persistence:"+active,h),h),
                         "live":live_status(db,h),
                         "paired":paired(db,active,shadow,h) if shadow else {"n":0,"pass":False},
                         "historical":report.get("horizons",{}).get(str(h))}
        if run:
            for h,summary in horizons.items():
                result=summary["historical"]
                if result and "error_series" not in result:
                    location=db.execute("SELECT path FROM runs WHERE id=?",(run["id"],)).fetchone()[0]
                    audit=pd.read_csv(Path(location).parent/f"{h}-holdout-scored.csv")
                    result["error_series"]={"dates":audit.target.tolist(),"errors":(audit.prediction-audit.actual).tolist()}
        shadow_path=home()/"foundation-shadow.json"
        configured={}
        if shadow_path.is_file():
            try:
                manifest=json.loads(shadow_path.read_text(encoding="utf8"))
                configured={h:[{k:e.get(k) for k in ("name","revision","status")} for e in entries]
                            for h,entries in manifest.get("horizons",{}).items()}
            except (OSError,ValueError,AttributeError,TypeError):
                configured={}
        runtime=home()/"foundation-live-r1"/"runtime.json"
        checked=None
        if runtime.is_file():
            try:
                checked=json.loads(runtime.read_text(encoding="utf8")).get("checked_utc")
            except (OSError,ValueError,AttributeError):
                pass
        return {"active":active,"challenger":shadow,"trained":load_version(db,active)["created_at"],
                "last_test":run["at"] if run else None,"tested_version":run["id"] if run else None,
                "horizons":horizons,"policy_forward":policy_forward(db,active),
                "foundation_forward":foundation_forward(db,active),
                "foundation_configured":configured,"foundation_last_tick_utc":checked,
                "events":[{**dict(r),"detail":r["detail"].splitlines()[-1][:250]} for r in db.execute(
                    "SELECT * FROM events ORDER BY seq DESC LIMIT 40")]}

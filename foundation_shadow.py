"""Opt-in offline foundation research forecasts. Production never fetches checkpoints."""
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from forecast_evaluation import clock_for
from forecast_registry import home, now, sha


@lru_cache(maxsize=3)
def load_checkpoint(folder, digest, kind="chronos", adapter=None, adapter_digest=None):
    if sha(folder/"model.safetensors")!=digest:
        raise ValueError("Foundation shadow weight hash mismatch")
    if kind=="chronos":
        from chronos import BaseChronosPipeline

        pipe=BaseChronosPipeline.from_pretrained(str(folder),device_map="cpu",local_files_only=True)
        if not adapter:
            return pipe
        base=pipe.model
    elif kind=="timesfm":
        from transformers import TimesFm2_5ModelForPrediction

        base=TimesFm2_5ModelForPrediction.from_pretrained(str(folder),local_files_only=True).eval()
        if not adapter:
            return base
    else:
        raise ValueError("Unknown foundation shadow model type")
    from peft import PeftModel

    if sha(adapter/"adapter_model.safetensors")!=adapter_digest:
        raise ValueError("Foundation shadow LoRA hash mismatch")
    tuned=PeftModel.from_pretrained(base,str(adapter),local_files_only=True).eval()
    return type(pipe)(model=tuned) if kind=="chronos" else tuned


def forecast_for(model, kind, context, horizon):
    import torch
    from kaggle_forecast_worker import chronos_close

    values=torch.tensor(context,dtype=torch.float32)
    with torch.inference_mode(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if kind=="chronos":
            q,mean=model.predict_quantiles([values],prediction_length=horizon,
                                           quantile_levels=[.025,.1,.5,.9,.975])
            return chronos_close(q,mean)
        output=model(past_values=[values])
    point=float(output.mean_predictions[0,horizon-1])
    levels=np.asarray(model.config.quantiles)
    quantiles=output.full_predictions[0,horizon-1,1:1+len(levels)].cpu().numpy()
    def endpoint(level):
        return float(np.interp(level,levels,quantiles)) if levels.min()<=level<=levels.max() else None
    return dict(prediction=point,low80=endpoint(.1),high80=endpoint(.9),
                low95=endpoint(.025),high95=endpoint(.975))


def context_at_close(data,horizon,asof):
    if data.empty or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Shadow context requires sorted, unique observations")
    calendar=clock_for(data.index,horizon)
    origin=data.index[-1]
    if origin not in calendar.index or calendar.loc[origin]>asof:
        raise ValueError("Latest shadow candle is off-calendar or incomplete")
    frame=data.loc[data.index.isin(calendar.index)]
    if not np.isfinite(frame.Close).all() or not frame.Close.gt(0).all():
        raise ValueError("Shadow close context contains invalid prices")
    return frame,calendar


def members(data,horizon,asof=None):
    manifest=home()/"foundation-shadow.json"
    if not manifest.is_file():
        return []
    settings=json.loads(manifest.read_text(encoding="utf8"))
    entries=settings.get("horizons",{}).get(str(horizon),[])
    if not entries:
        return []
    names=[entry["name"] for entry in entries]
    if len(names)!=len(set(names)) or "equal" in names:
        raise ValueError("Foundation shadow names must be unique and cannot use 'equal'")
    frame,calendar=context_at_close(data,horizon,now() if asof is None else asof)
    origin=data.index[-1]
    target=calendar.iloc[calendar.index.get_loc(origin)+horizon]
    values=[]
    result=[]
    for entry in entries:
        name=entry["name"]
        digest=entry["sha256"]
        revision=entry["revision"]
        if not re.fullmatch(r"[a-z0-9_-]+",name) or not re.fullmatch(r"[a-f0-9]{40}",revision):
            raise ValueError("Invalid foundation shadow name or revision")
        context=int(entry["context"])
        if not 64<=context<=512 or len(frame)<context:
            raise ValueError("Insufficient completed closes for the configured context")
        checkpoint=Path(entry["directory"]).resolve()
        kind=entry.get("kind","chronos")
        adapter=Path(entry["adapter"]).resolve() if entry.get("adapter") else None
        adapter_digest=entry.get("adapter_sha256")
        pipe=load_checkpoint(checkpoint,digest,kind,adapter,adapter_digest)
        forecast=forecast_for(pipe,kind,frame.Close.tail(context).to_numpy(),horizon)
        values.append(forecast)
        result.append(record(name,f"foundation:{name}:{revision[:12]}:{(adapter_digest or digest)[:10]}:{context}",
                             {"revision":revision,"sha256":digest,"adapter_sha256":adapter_digest,
                              "kind":kind,"context":context,"model_id":entry.get("model_id"),
                              "preprocessing":"observed-gc-close-v1","instrument":"GC=F",
                              "registered_at":entry.get("registered_at"),
                              "calendar":version("pandas_market_calendars"),
                              "target_close_utc":str(target)},forecast))
    if len(result)>1 and settings.get("equal_ensemble",False):
        import hashlib
        lineage=[item[1] for item in result]
        identifier=hashlib.sha256(json.dumps(lineage).encode()).hexdigest()[:20]
        point={name:float(np.mean([v[name] for v in values]))
               if all(v[name] is not None for v in values) else None for name in values[0]}
        result.append(record("equal",f"foundation:equal:{identifier}",{"members":lineage},point))
    return result


def record(name,version,config,values):
    # The shared ledger stores symmetric margins. Retain native endpoints in config.
    point=values["prediction"]
    if not np.isfinite(point) or not np.isfinite([x for x in values.values() if x is not None]).all():
        raise ValueError("Nonfinite foundation forecasts")
    for level in ("80","95"):
        low,high=values["low"+level],values["high"+level]
        if (low is None)!=(high is None) or (low is not None and low>high):
            raise ValueError("Invalid foundation interval endpoints")
    margins={k:max(abs(point-values["low"+k]),abs(values["high"+k]-point))
             if values["low"+k] is not None and values["high"+k] is not None else None
             for k in ("80","95")}
    return (f"foundation:{name}",version,{**config,"native_forecast":values,
                                         "classification":"research shadow"},point,1.,margins)

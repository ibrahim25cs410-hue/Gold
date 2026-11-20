"""Kaggle notebook workload. Local driver stages immutable inputs and this code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

MODELS = {
    "chronos2": "amazon/chronos-2",
    "chronos2-small": "autogluon/chronos-2-small",
    "chronos-t5-small": "amazon/chronos-t5-small",
    "timesfm": "google/timesfm-2.5-200m-transformers",
    "timestral": "mistralai/Timestral-209M",
    "kronos": "NeoQuasar/Kronos-small",
}


def gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle allocated no usable CUDA GPU; CPU fallback disabled.")
    return [{"name": torch.cuda.get_device_name(i),
             "free_bytes": torch.cuda.mem_get_info(i)[0],
             "total_bytes": torch.cuda.mem_get_info(i)[1]}
            for i in range(torch.cuda.device_count())]


def chronos_close(quantiles, mean):
    # Chronos-2 has a variate axis, while Chronos-T5 is already univariate.
    band=quantiles[0].reshape(-1,5)[-1]
    return {"prediction":float(mean[0].reshape(-1)[-1]),
            "low80":float(band[1]),"high80":float(band[3]),
            "low95":float(band[0]),"high95":float(band[4])}


def fit_adapter(config, report):
    from huggingface_hub import HfApi, snapshot_download

    key = config["model"]
    model_id = MODELS[key]
    revision = config.get("revision") or HfApi().model_info(model_id).sha
    checkpoint = snapshot_download(model_id,revision=revision,
                                   allow_patterns=["*.json","*.safetensors"])
    report.update(model_id=model_id, revision=revision)
    if key.startswith("chronos"):
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(checkpoint,device_map="cuda:0")
        def predict(context, future):
            values = torch.tensor(context.Close.to_numpy(),dtype=torch.float32)
            quantiles,mean = pipeline.predict_quantiles([values],prediction_length=len(future),
                                                quantile_levels=[.025,.10,.5,.90,.975])
            return chronos_close(quantiles,mean)
        return predict
    if key=="timesfm":
        from transformers import TimesFm2_5ModelForPrediction
        model = TimesFm2_5ModelForPrediction.from_pretrained(checkpoint).to("cuda:0").eval()
        def predict(context, future):
            values = torch.tensor(context.Close.to_numpy(),device="cuda:0",dtype=torch.float32)
            return float(model(past_values=[values]).mean_predictions.reshape(-1)[len(future)-1])
        return predict
    if key=="timestral":
        from timestral import TimestralForecaster
        forecaster=TimestralForecaster.from_pretrained(checkpoint,device="cuda")
        def predict(context,future):
            values=context.Close.to_numpy(dtype=np.float32)[None,None,:]
            forecast=forecaster.forecast(values,prediction_length=len(future))
            return float(forecast.median[0,0,-1])
        return predict
    from model import Kronos, KronosPredictor, KronosTokenizer
    tokenizer_id = "NeoQuasar/Kronos-Tokenizer-base"
    token_revision = config.get("tokenizer_revision") or HfApi().model_info(tokenizer_id).sha
    tokenizer_dir = snapshot_download(tokenizer_id,revision=token_revision)
    report["tokenizer_revision"] = token_revision
    predictor = KronosPredictor(Kronos.from_pretrained(checkpoint),
                                KronosTokenizer.from_pretrained(tokenizer_dir),
                                device="cuda:0",max_context=config["context"])
    def predict(context,future):
        columns = context[["Open","High","Low","Close","Volume"]].copy()
        columns.columns = ["open","high","low","close","volume"]
        return float(predictor.predict(df=columns,x_timestamp=pd.Series(context.index),
                                       y_timestamp=pd.Series(future),pred_len=len(future),
                                       T=.6,top_p=.9,sample_count=4).close.iloc[-1])
    return predict


def install(config):
    def pip(*packages):
        subprocess.run([sys.executable,"-m","pip","install","--quiet",*packages],check=True)
    if config["model"].startswith("chronos"):
        pip("chronos-forecasting==2.3.2")
    if config["model"]=="timesfm":
        pip("transformers==5.23.0","accelerate")
    if config["model"]=="timestral":
        pip("git+https://github.com/mistralai/timestral.git@"+config["timestral_commit"])
    if config["model"]=="kronos":
        source=Path("/tmp/aura-kronos-source")
        subprocess.run(["git","clone","https://github.com/shiyu-coder/Kronos.git",str(source)],check=True)
        subprocess.run(["git","-C",str(source),"checkout","--detach",config["kronos_commit"]],check=True)
        # The upstream pandas pin requires a source build on Kaggle's Python 3.13.
        # Preserve the working Kaggle NumPy, pandas and CUDA-enabled PyTorch stack.
        pip("--only-binary=:all:","einops==0.8.1")
        sys.path.insert(0,str(source))


def write_json(path,data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf8")
    temporary.replace(path)


def run(config,inputs,output):
    output.mkdir(parents=True,exist_ok=True)
    report={"status":"running","evidence":"historical diagnostic; dates previously examined",
            "config":config,"torch":torch.__version__,"cuda_build":torch.version.cuda}
    start=time.monotonic()
    try:
        report["gpu"]=gpu()
        report["cuda_available"]=True
        write_json(output/"status.json",report)
        if report["gpu"][0]["free_bytes"]<6*1024**3:
            raise RuntimeError("Less than 6 GiB free on GPU 0; preflight stopped.")
        torch.manual_seed(42)
        install(config)
        predictor=fit_adapter(config,report)
        torch.cuda.synchronize()
        report["model_loaded_allocated_bytes"]=torch.cuda.memory_allocated()
        write_json(output/"status.json",report)
        index=pd.read_csv(inputs/"origins.csv")
        prices={h:pd.read_csv(inputs/("hourly.csv" if h==1 else "daily.csv"),
                               index_col="Date",parse_dates=["Date"]) for h in (1,7,30)}
        result_path=output/"forecasts.csv"
        levels = index.sort_values(["horizon","origin"])
        checkpoint=inputs/"checkpoint.csv"
        completed=[]
        if checkpoint.exists():
            saved=pd.read_csv(checkpoint)
            if not saved.row_id.is_unique or not np.isfinite(saved.prediction).all():
                raise ValueError("Invalid forecast checkpoint")
            expected=index.set_index("row_id").loc[saved.row_id].reset_index()
            pd.testing.assert_frame_equal(saved[index.columns].reset_index(drop=True),expected,
                                          check_exact=False,rtol=1e-10,atol=1e-10)
            completed=saved.to_dict("records")
        previous={row["row_id"] for row in completed}
        for row in levels.itertuples(index=False):
            if row.row_id in previous:
                continue
            if time.monotonic()-start>config["budget_seconds"]:
                raise TimeoutError("Forecast budget exhausted; completed rows retained")
            origin=pd.Timestamp(row.origin)
            prefix=prices[row.horizon].loc[:origin].tail(config["context"])
            future=pd.DatetimeIndex(json.loads(row.future))
            started=time.monotonic()
            with torch.inference_mode():
                values=predictor(prefix,future)
            torch.cuda.synchronize()
            if not isinstance(values,dict):
                values={"prediction":values}
            if not np.isfinite(list(values.values())).all():
                raise ValueError("Nonfinite forecast")
            completed.append({**row._asdict(),**values,"inference_seconds":time.monotonic()-started})
            temporary=result_path.with_suffix(".tmp")
            pd.DataFrame(completed).to_csv(temporary,index=False)
            temporary.replace(result_path)
        if not result_path.exists():
            pd.DataFrame(completed).to_csv(result_path,index=False)
        result=pd.read_csv(result_path)
        report["metrics"]={}
        for h,rows in result.groupby("horizon"):
            scores={}
            for name in ("prediction","champion","persistence"):
                err=rows[name]-rows.actual
                scores[name]={"n":len(err),"mae":float(err.abs().mean()),
                              "rmse":float(np.sqrt(err.pow(2).mean())),"bias":float(err.mean()),
                              "direction_pct":float(100*(np.sign(rows[name]-rows.persistence)==
                                                          np.sign(rows.actual-rows.persistence)).mean())}
            base=scores["persistence"]["mae"]
            scores["skill_pct"]=100*(1-scores["prediction"]["mae"]/base) if base else None
            for level in (80,95):
                low,high=f"low{level}",f"high{level}"
                if low in rows and rows[[low,high]].notna().all().all():
                    scores["prediction"][f"coverage{level}"]=float(
                        100*rows.actual.between(rows[low],rows[high]).mean())
            report["metrics"][str(h)]=scores
        report["status"]="completed"
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        print(report["error"],flush=True)
    finally:
        if torch.cuda.is_available():
            report["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        report["elapsed_seconds"]=time.monotonic()-start
        write_json(output/"status.json",report)

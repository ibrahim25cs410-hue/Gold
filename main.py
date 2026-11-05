# -*- coding: utf-8 -*-
"""FastAPI server for local use and Hugging Face Spaces."""
import traceback
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import gold_engine as ge

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Aura Gold AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/icon.png")
def icon():
    p = BASE_DIR / "icon.png"
    return FileResponse(p) if p.exists() else JSONResponse({"status": "error"}, status_code=404)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": ge.BUNDLE_PATH.exists(),
        "live_market": True,
    }


@app.get("/predict")
def predict(horizon: int = 1, live: bool = False):
    try:
        return ge.predict_live(horizon) if live else ge.predict(horizon)
    except ge.EngineError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:  # خطأ غير متوقع: نُظهره بوضوح بدل رقم وهمي
        traceback.print_exc()
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}

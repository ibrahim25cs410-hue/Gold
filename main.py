# -*- coding: utf-8 -*-
"""FastAPI server for local use and Hugging Face Spaces."""
import logging
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import aura_price_engine as pe

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
app = FastAPI(title="Aura Gold AI")
API_VERSION = "price-forecast-v2"
DEPLOYMENT_REVISION = os.getenv("RENDER_GIT_COMMIT", "unknown")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Gold trading hours used by the app (Iraq time, UTC+3):
# open Monday at 01:00 and close Friday at 00:00.
MARKET_TZ = timezone(timedelta(hours=3), name="Asia/Baghdad")
MARKET_OPEN_TIME = time(1, 0)
MARKET_CLOSED_WEEKDAYS = (4, 5, 6)  # Friday through Sunday


def is_market_open(now: datetime | None = None) -> bool:
    """Return whether the gold market is open at the given Iraq-local time."""
    local_now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if local_now.weekday() in MARKET_CLOSED_WEEKDAYS:
        return False
    if local_now.weekday() == 0 and local_now.time() < MARKET_OPEN_TIME:
        return False
    return True


@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/icon.png")
def icon():
    p = BASE_DIR / "icon.png"
    return FileResponse(p) if p.exists() else JSONResponse({"status": "error"}, status_code=404)


@app.get("/health")
def health():
    try:
        bundle = pe.load_bundle()
        models_ready = all(
            horizon in bundle.get("models", {}) for horizon in (1, 7, 30)
        )
        if not models_ready:
            raise pe.PriceEngineError("حزمة النماذج لا تحتوي نماذج الآفاق 1 و7 و30.")
    except Exception as e:
        logger.exception("Model bundle readiness check failed")
        return JSONResponse(
            {
                "status": "error",
                "api_version": API_VERSION,
                "revision": DEPLOYMENT_REVISION,
                "models_ready": False,
                "detail": str(e),
            },
            status_code=503,
        )
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "revision": DEPLOYMENT_REVISION,
        "models_ready": True,
        "live_market": is_market_open(),
    }


@app.get("/live-price")
def live_price():
    try:
        result = pe.live_price()
        result["market_open"] = is_market_open()
        if not result["market_open"]:
            result["source"] = "آخر إغلاق معروف"
        return result
    except pe.PriceEngineError as e:
        return {"status": "error", "message": str(e)}
    except Exception:
        logger.exception("Unexpected live price failure")
        return {
            "status": "error",
            "message": "تعذر جلب السعر الحي حالياً. حاول مرة أخرى لاحقاً.",
        }


@app.get("/predict")
def predict(horizon: int = 1):
    if horizon not in (1, 7, 30):
        raise HTTPException(
            status_code=422,
            detail="الأفق المدعوم هو 1 أو 7 أو 30.",
        )
    try:
        market_open = is_market_open()
        result = pe.predict(horizon)
        result["market_open"] = market_open
        result["api_version"] = API_VERSION
        result["revision"] = DEPLOYMENT_REVISION
        return result
    except pe.PriceEngineError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected prediction failure")
        raise HTTPException(status_code=500, detail=str(e)) from e

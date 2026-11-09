# -*- coding: utf-8 -*-
"""FastAPI server for local use and Hugging Face Spaces."""
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import gold_engine as ge

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
app = FastAPI(title="Aura Gold AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Gold trading hours used by the app (Iraq time, UTC+3):
# open Monday at 01:00 and close Friday at 00:00.
MARKET_TZ = timezone(timedelta(hours=3), name="Asia/Baghdad")
MARKET_OPEN_TIME = time(1, 0)
MARKET_CLOSED_WEEKDAYS = (4, 5, 6)  # Friday through Sunday
_LAST_LIVE_RESULTS: dict[int, dict[str, Any]] = {}


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
        "live_market": is_market_open(),
    }


@app.get("/live-price")
def live_price():
    try:
        result = ge.get_live_price()
        result["market_open"] = is_market_open()
        if not result["market_open"]:
            result["source"] = "آخر إغلاق معروف"
        return result
    except ge.EngineError as e:
        return {"status": "error", "message": str(e)}
    except Exception:
        logger.exception("Unexpected live price failure")
        return {
            "status": "error",
            "message": "تعذر جلب السعر الحي حالياً. حاول مرة أخرى لاحقاً.",
        }


@app.get("/predict")
def predict(horizon: int = 1, live: bool = False):
    try:
        market_open = is_market_open()
        if live and not market_open:
            cached = _LAST_LIVE_RESULTS.get(int(horizon))
            if cached is None:
                cached = ge.predict(horizon)
                _LAST_LIVE_RESULTS[int(horizon)] = dict(cached)
            else:
                cached = dict(cached)
            cached["live"] = True
            cached["source"] = "آخر إغلاق معروف"
            cached["market_open"] = False
            return cached
        result = ge.predict_live(horizon) if live else ge.predict(horizon)
        result["market_open"] = market_open
        if live:
            _LAST_LIVE_RESULTS[int(horizon)] = dict(result)
        return result
    except ge.EngineError as e:
        return {"status": "error", "message": str(e)}
    except Exception:
        logger.exception("Unexpected prediction failure")
        return {
            "status": "error",
            "message": "تعذر تنفيذ التوقع حالياً. حاول مرة أخرى لاحقاً.",
        }

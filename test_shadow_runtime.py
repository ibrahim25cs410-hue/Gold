import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from foundation_shadow import context_at_close,load_checkpoint,record
from forecast_evaluation import summarize
from forecast_registry import database
from shadow_runtime import audit_missed, tick


class ShadowRuntimeTests(unittest.TestCase):
    def test_unavailable_native_interval_is_unscored(self):
        forecast=dict(prediction=101.,low80=99.,high80=103.,low95=None,high95=None)
        self.assertIsNone(record("probe","test",{},forecast)[-1]["95"])
        metrics=summarize(pd.DataFrame([{**forecast,"actual":100.,"origin_close":99.}]),7)
        self.assertIsNone(metrics["coverage95"])
        self.assertIsNone(metrics["width95"])
        self.assertEqual(metrics["coverage80"],100.)

    def test_incomplete_bar_rejected_and_exact_target_preserved(self):
        origin=pd.Timestamp("2026-11-18 08:00")
        data=pd.DataFrame({"Close":[1900.]},index=pd.DatetimeIndex([origin]))
        with self.assertRaisesRegex(ValueError,"incomplete"):
            context_at_close(data,1,pd.Timestamp("2026-11-18 08:59:59"))
        frame,clock=context_at_close(data,1,pd.Timestamp("2026-11-18 09:00"))
        self.assertEqual(frame.Close.iloc[-1],1900)
        self.assertEqual(clock.iloc[clock.index.get_loc(origin)+1],pd.Timestamp("2026-11-18 10:00"))

    def test_corrupt_and_absent_checkpoint_fail_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(folder,"untrusted")
            (folder/"model.safetensors").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                load_checkpoint(folder,"untrusted")

    def test_missed_origin_is_not_backfilled_or_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,AURA_STATE_DIR=tmp),database() as db:
            config={"1":[{"name":"test"}]}
            old=pd.Timestamp("2026-11-18 09:59")
            now=pd.Timestamp("2026-11-18 10:01")
            audit_missed(db,old,now,config)
            audit_missed(db,old,now,config)
            self.assertEqual(db.execute("SELECT count(*) FROM events WHERE action='shadow_missed'").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT count(*) FROM forecasts").fetchone()[0],0)

    def test_ticker_logs_one_market_failure_and_checks_other_horizons(self):
        import json
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,AURA_STATE_DIR=tmp):
            (Path(tmp)/"foundation-shadow.json").write_text(
                json.dumps({"horizons":{str(h):[{"name":"test"}] for h in (1,7,30)}}))
            fake=dict(forecast_id="test",forecast_group_id=None,as_of="test",
                      target_close_utc="test",market_data_source="fixture")
            with patch("aura_price_engine.predict",side_effect=[RuntimeError("offline"),fake,fake]):
                tick()
            state=json.loads((Path(tmp)/"foundation-live-r1/runtime.json").read_text())
            self.assertEqual(state["attempt"]["1"]["error"],"offline")
            self.assertEqual(state["attempt"]["30"]["forecast_id"],"test")

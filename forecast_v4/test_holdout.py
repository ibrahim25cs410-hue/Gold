"""Regression checks for the frozen hourly evaluation window."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo

from .holdout import in_holdout, ledger_utc, load_bounds
from .prepare_demo import prepare
from .verify_hourly import analyze


class HourlyHoldoutTests(unittest.TestCase):
    bounds = load_bounds(Path(__file__).with_name("hourly_holdout.json"))

    def test_baghdad_boundary_conversion(self):
        baghdad = ZoneInfo("Asia/Baghdad")
        expected = (datetime(2026, 11, 23, 3, tzinfo=baghdad),
                    datetime(2027, 1, 4, 3, tzinfo=baghdad))
        for utc, local in zip(self.bounds, expected):
            self.assertEqual(utc.replace(tzinfo=timezone.utc).astimezone(baghdad), local)
            self.assertEqual(ledger_utc(local.isoformat()), utc)

    def test_inclusive_start_exclusive_end(self):
        for issue, expected in (
            ("2026-11-22T23:59:59.999999Z", False),
            ("2026-11-23 00:00:00", True),
            ("2026-11-23T03:00:00+03:00", True),
            ("2027-01-04T02:59:59.999999+03:00", True),
            ("2027-01-04T03:00:00+03:00", False),
            ("2027-01-04 00:00:00", False),
        ):
            with self.subTest(issue=issue):
                self.assertEqual(in_holdout(issue, self.bounds), expected)

    def test_mixed_offset_inputs_on_both_read_paths(self):
        group = {
            "group_id": "fixture",
            "observation_close": "2026-11-23 00:00:00",
            "issued": "2026-11-23T03:01:00+03:00",
            "target_close": "2026-11-23T01:00:00Z",
            "observed": "2026-11-23T04:03:00+03:00",
            "origin_close": 99,
            "actual": 101,
        }
        rows = [
            dict(group, role=role, version=version, prediction=prediction)
            for role, version, prediction in (
                ("champion", "original", 100),
                ("persistence", "reference", 99),
                ("drift", "drift20", 101),
            )
        ]
        snapshot = prepare(rows, "fixture", self.bounds)
        self.assertEqual(snapshot["cohorts"]["original"]["scores"]["original"]["n"], 1)
        self.assertEqual(snapshot["cohorts"]["original"]["rows"][0]["issue"], group["issued"])
        analysis = analyze([row for row in rows if in_holdout(row["issued"], self.bounds)])
        self.assertEqual(analysis["counts"]["fixed_hour"], 1)
        self.assertEqual(analysis["observed_delay_bounds_seconds"], [180, 180])

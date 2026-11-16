import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import aura_price_engine as engine
from build_dataset import add_relative_features
from price_features import sequence_window
from train_price_forecaster import (
    chronological_split,
    metrics,
    range_coverages,
    sequence_data,
)


class ForecastIntegrityTests(unittest.TestCase):
    def test_resampled_trends_do_not_change_when_future_prices_change(self) -> None:
        index = pd.date_range("2025-01-01", periods=1500, freq="h")
        close = pd.Series(2000 + np.arange(len(index)) * 0.1, index=index)
        source = pd.DataFrame(
            {
                "Close": close,
                "Oil_Close": 70 + np.arange(len(index)) * 0.001,
                "US10Y_Yield": 4 + np.arange(len(index)) * 0.00001,
                "DXY": 100 + np.arange(len(index)) * 0.001,
                "ATR_14": 2.0,
                "Volume": 1000.0,
            },
            index=index,
        )
        cutoff = index[1000]
        changed = source.copy()
        changed.loc[changed.index > cutoff, "Close"] *= 1.5

        original_features = add_relative_features(source)
        changed_features = add_relative_features(changed)
        for column in ("Trend_4H", "Trend_1D"):
            pd.testing.assert_series_equal(
                original_features.loc[:cutoff, column],
                changed_features.loc[:cutoff, column],
            )

    def test_training_sequence_includes_current_row(self) -> None:
        features = np.arange(60, dtype=np.float32).reshape(30, 2)
        targets = np.zeros((30, 3), dtype=np.float32)
        sequence, _ = sequence_data(
            features, targets, np.asarray([15]), window=14
        )
        expected = sequence_window(features, current_index=15, window=14)
        np.testing.assert_array_equal(sequence[0], expected)
        np.testing.assert_array_equal(sequence[0][-1], features[15])

    def test_split_boundaries_have_horizon_embargoes(self) -> None:
        for horizon in (1, 7, 30):
            with self.subTest(horizon=horizon):
                split = chronological_split(1300, horizon)
                train = split["train"]
                tuning = split["validation_tuning"]
                calibration = split["validation_calibration"]
                holdout = split["holdout"]
                self.assertLess(train[-1] + horizon, tuning[0])
                self.assertLess(tuning[-1] + horizon, calibration[0])
                self.assertLess(calibration[-1] + horizon, holdout[0])
                self.assertFalse(set(train) & set(tuning))
                self.assertFalse(set(calibration) & set(holdout))

    def test_metrics_and_range_coverages_are_separate(self) -> None:
        actual = np.asarray([[110.0, 90.0, 100.0], [120.0, 95.0, 105.0]])
        predicted = np.asarray([[112.0, 89.0, 101.0], [117.0, 94.0, 103.0]])
        result = metrics(actual, predicted)
        self.assertEqual(result["mae_close_usd"], 1.5)
        self.assertEqual(result["mae_high_usd"], 2.5)
        self.assertEqual(result["mae_low_usd"], 1.0)

        coverage_actual = actual.copy()
        coverage_actual[:, 2] = 200.0
        calibration = {
            "95": {
                "low_margin_usd": 0.1,
                "high_margin_usd": 0.1,
                "close_margin_usd": 0.1,
            },
            "99": {
                "low_margin_usd": 100.0,
                "high_margin_usd": 100.0,
                "close_margin_usd": 100.0,
            },
        }
        coverage = range_coverages(coverage_actual, predicted, calibration)
        self.assertLess(coverage["95"], coverage["99"])

    def test_cache_uses_ttl_and_returns_independent_dataframes(self) -> None:
        engine._data_cache.clear()
        calls = 0

        def load_frame() -> pd.DataFrame:
            nonlocal calls
            calls += 1
            return pd.DataFrame({"value": [1]})

        with patch(
            "aura_price_engine.time.monotonic",
            side_effect=[0, 0, 10, 31, 31, 31],
        ):
            first = engine._cached_value("test-frame", 30, load_frame)
            first.loc[0, "value"] = 99
            second = engine._cached_value("test-frame", 30, load_frame)
            third = engine._cached_value("test-frame", 30, load_frame)

        self.assertEqual(calls, 2)
        self.assertEqual(second.loc[0, "value"], 1)
        self.assertEqual(third.loc[0, "value"], 1)
        self.assertEqual(engine.LIVE_PRICE_TTL_SECONDS, 30)
        self.assertEqual(engine.MARKET_DATA_TTL_SECONDS, 300)
        engine._data_cache.pop("test-frame", None)


if __name__ == "__main__":
    unittest.main()

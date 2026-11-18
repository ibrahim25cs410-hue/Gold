import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import aura_price_engine as engine
from build_dataset import add_relative_features, download_daily, download_hourly
from price_features import (
    ECONOMIC_FEATURES, MACRO_TICKERS, align_daily_macro, align_lagged_daily_close,
    daily_features, event_seasonality_features, gold_target_prices, hourly_features,
    realized_volatility, residual_feedback, sequence_window,
)
from train_price_forecaster import (
    chronological_split, conformal_calibration, metrics, range_coverages, sequence_data,
    train_horizon,
)


class ForecastIntegrityTests(unittest.TestCase):
    def test_daily_macro_only_becomes_available_next_day_and_carries_across_gaps(self) -> None:
        source = pd.Series([10.0, 20.0, 40.0], index=pd.to_datetime([
            "2025-07-02", "2025-07-03", "2025-07-07",
        ]))
        target = pd.to_datetime([
            "2025-07-02 21:00", "2025-07-03 23:00", "2025-07-04 00:00",
            "2025-07-05 10:00", "2025-07-07 14:00", "2025-07-08 00:00",
        ])
        expected = [np.nan, 10.0, 20.0, 20.0, 20.0, 40.0]
        np.testing.assert_equal(align_lagged_daily_close(source, target).to_numpy(), expected)
        prefix = source.iloc[:2]
        np.testing.assert_equal(
            align_lagged_daily_close(prefix, target[:-1]).to_numpy(), expected[:-1],
        )

    @staticmethod
    def _macro_inputs() -> dict[str, pd.Series]:
        sessions = pd.bdate_range("2024-10-01", periods=150)
        times = np.arange(len(sessions))
        return {
            name: pd.Series(10 + 10 * offset + times * 0.1 + np.sin(times / (2 + offset)),
                            index=sessions).drop(sessions[57 + offset])
            for offset, name in enumerate(MACRO_TICKERS)
        }

    def test_macro_and_gold_future_mutations_leave_all_feature_prefixes_unchanged(self) -> None:
        originals = self._macro_inputs()
        cutoff_day = pd.Timestamp("2025-03-03")
        changed = {name: source.where(source.index < cutoff_day, source * 2)
                   for name, source in originals.items()}
        for hourly in (True, False):
            with self.subTest(hourly=hourly):
                dates = (pd.date_range("2025-01-01", "2025-03-15", freq="h") if hourly
                         else pd.bdate_range("2024-11-15", "2025-03-15"))
                cutoff = cutoff_day + pd.Timedelta(hours=13) if hourly else cutoff_day
                close = pd.Series(2000 + np.sin(np.arange(len(dates)) / 4) * 30, index=dates)
                bars = pd.DataFrame({"Close": close, "Gold_High": close + 10, "Gold_Low": close - 10})
                updated = bars.copy()
                updated.loc[updated.index > cutoff] *= 1.3
                baseline = bars.join(align_daily_macro(originals, dates))
                modified = updated.join(align_daily_macro(changed, dates))
                make_features = hourly_features if hourly else daily_features
                expected = make_features(baseline)
                actual = make_features(modified)
                pd.testing.assert_frame_equal(expected.loc[:cutoff], actual.loc[:cutoff])
                for name in ECONOMIC_FEATURES:
                    self.assertTrue(expected.loc[cutoff:, name].notna().all(), name)
                for name in ("geopolitical_war_risk", "psych_sentiment_index", "real_rate_usd_pressure"):
                    self.assertNotEqual(expected[name].iloc[-1], actual[name].iloc[-1], name)
                pd.testing.assert_series_equal(
                    realized_volatility(baseline["Close"], 7).loc[:cutoff],
                    realized_volatility(modified["Close"], 7).loc[:cutoff],
                )

    def test_calendar_features_use_scheduled_sessions_even_with_missing_bars(self) -> None:
        dates = pd.to_datetime(["2025-01-02", "2025-01-07", "2025-04-17",
                                "2025-04-18", "2025-06-30", "2025-07-01",
                                "2025-07-03", "2025-07-04", "2025-07-07",
                                "2025-10-15", "2025-11-26", "2025-12-24",
                                "2025-12-26"])
        calendar = event_seasonality_features(dates, hourly=False)
        for date in ("2025-04-17", "2025-04-18", "2025-07-03", "2025-07-04",
                     "2025-11-26", "2025-12-24", "2025-12-26"):
            self.assertEqual(calendar.loc[date, "is_major_holiday_window"], 1, date)
        self.assertEqual(calendar.loc["2025-06-30", "is_turn_of_month"], 1)
        self.assertEqual(calendar.loc["2025-07-03", "is_turn_of_month"], 1)
        self.assertEqual(calendar.loc["2025-07-07", "is_turn_of_month"], 0)
        self.assertEqual(calendar.loc["2025-01-07", "is_turn_of_month"], 0)
        self.assertEqual(calendar.loc["2025-01-02", "nfp_event_week"], 1)
        self.assertEqual(calendar.loc["2025-01-07", "nfp_event_week"], 0)
        self.assertEqual(calendar.loc["2025-01-02", "festival_demand_regime"], 2)
        self.assertEqual(calendar.loc["2025-10-15", "festival_demand_regime"], 1)
        pd.testing.assert_frame_equal(calendar.iloc[:5], event_seasonality_features(dates[:5], hourly=False))
        hours = pd.to_datetime(["2025-03-07 18:00", "2025-03-07 19:00",
                                "2025-07-11 17:00", "2025-07-11 18:00"])
        np.testing.assert_array_equal(
            event_seasonality_features(hours, hourly=True)["is_friday_close_window"], [0, 1, 0, 1],
        )

    def test_ingestion_uses_only_complete_gold_candles_and_all_daily_macro_series(self) -> None:
        as_of = pd.Timestamp("2025-03-03 18:30", tz="UTC")
        daily_dates = pd.bdate_range("2025-01-01", "2025-03-03")
        hourly_dates = pd.date_range("2025-03-01", as_of.tz_localize(None).floor("h"), freq="h")

        def download(ticker, period, interval, columns):
            self.assertEqual(ticker, "GC=F")
            dates = hourly_dates if interval == "1h" else daily_dates
            return pd.DataFrame(2000.0, index=dates, columns=columns)

        with patch("build_dataset.download_market", side_effect=download):
            hourly = download_hourly(self._macro_inputs(), as_of)
            daily = download_daily(self._macro_inputs(), as_of)
        self.assertEqual(hourly.index[-1], pd.Timestamp("2025-03-03 17:00"))
        self.assertEqual(daily.index[-1], pd.Timestamp("2025-02-28"))
        self.assertIn("TIP_Close", hourly)
        self.assertIn("TIP_Close", daily)

    def test_shifted_targets_and_residual_feedback_use_original_bar_clock(self) -> None:
        data = pd.DataFrame({"Close": [100, 101, 102, 103], "High": [101, 102, 103, 104],
                             "Low": [90, 91, 92, 93], "Target_Close": 1000})
        self.assertEqual(gold_target_prices(data, 1).iloc[0]["Target_Close"], 101)
        for horizon in (1, 7, 30):
            with self.subTest(horizon=horizon):
                residuals = np.full((100, 3), np.nan)
                residuals[12] = [1, 2, 3]
                before = residual_feedback(residuals, horizon, window=5, gain=0.25)
                changed = residuals.copy()
                changed[13] = [300, 400, 500]
                after = residual_feedback(changed, horizon, window=5, gain=0.25)
                np.testing.assert_array_equal(before[:13 + horizon], after[:13 + horizon])
                np.testing.assert_array_equal(before[12 + horizon], [0.25, 0.5, 0.75])
                np.testing.assert_array_equal(before[17 + horizon], [0, 0, 0])
                self.assertFalse(np.array_equal(before[13 + horizon], after[13 + horizon]))

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
                self.assertEqual(train[-1] + horizon + 1, tuning[0])
                self.assertEqual(tuning[-1] + horizon + 1, calibration[0])
                self.assertEqual(calibration[-1] + horizon + 1, holdout[0])
                if horizon != 1:
                    self.assertEqual(len(tuning) + len(calibration), 200)
                self.assertFalse(set(train) & set(tuning))
                self.assertFalse(set(calibration) & set(holdout))

    def test_metrics_and_range_coverages_are_separate(self) -> None:
        actual = np.asarray([[110.0, 90.0, 100.0], [120.0, 95.0, 105.0]])
        predicted = np.asarray([[112.0, 89.0, 101.0], [117.0, 94.0, 103.0]])
        result = metrics(actual, predicted)
        self.assertEqual(result["mae_close_usd"], 1.5)
        self.assertEqual(result["mae_high_usd"], 2.5)
        self.assertEqual(result["mae_low_usd"], 1.0)

        quantiles = conformal_calibration(np.arange(1, 101, dtype=float))
        self.assertEqual(quantiles, {"80": 81.0, "95": 96.0})
        actual_close = np.asarray([105.0, 110.0])
        predicted_close = np.asarray([100.0, 100.0])
        coverages = range_coverages(actual_close, predicted_close,
                                    np.asarray([2.5, 10.0]), {"80": 1.0, "95": 2.0})
        self.assertEqual(coverages, {"80": 50.0, "95": 100.0})

    def test_unresolved_holdout_outcomes_cannot_change_parameters_or_forecasts(self) -> None:
        index = pd.bdate_range("2023-01-02", periods=450)
        close = pd.Series(2000.0 + np.random.default_rng(8).normal(size=len(index)).cumsum(), index=index)
        data = pd.DataFrame({"Close": close, "Gold_High": close + 1, "Gold_Low": close - 1})
        features = pd.DataFrame({"prior_return": close.pct_change()}, index=index)
        original = train_horizon(7, data, features)
        amended = data.copy()
        amended.iloc[-1] += 100
        changed = train_horizon(7, amended, features)
        for key in ("ensemble_weights", "damping_factor", "conformal_intervals"):
            self.assertEqual(original[key], changed[key], key)
        np.testing.assert_array_equal(original["holdout_audit"]["predicted_prices"],
                                      changed["holdout_audit"]["predicted_prices"])
        self.assertNotEqual(original["metrics"]["mae_close_usd"], changed["metrics"]["mae_close_usd"])

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

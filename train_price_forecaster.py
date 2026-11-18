from __future__ import annotations

from hashlib import file_digest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_dataset import DAILY_FILE, ROOT, OUTPUT_FILE as HOURLY_FILE
from price_features import (
    BUNDLE_VERSION, RESIDUAL_GAIN, RESIDUAL_WINDOW, blend_returns,
    daily_features, gold_target_prices, hourly_features, range_bounds,
    realized_volatility, residual_feedback, sequence_window,
)


BUNDLE_FILE = ROOT / "aura_price_models.pkl"
SEED = 42
HOLDOUT_ROWS = {1: 200, 7: 100, 30: 100}
LSTM_WINDOW = 14


class PriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True
        )
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.output(output[:, -1])


def sequence_data(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = indices[indices >= window - 1]
    return (
        np.stack([sequence_window(features, int(i), window) for i in indices]).astype(
            np.float32
        ),
        targets[indices].astype(np.float32),
    )


def train_lstm(
    features: pd.DataFrame, targets: np.ndarray, train_indices: np.ndarray,
) -> tuple[PriceLSTM, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(features.iloc[train_indices])
    y_scaler = StandardScaler().fit(targets[train_indices])
    train_x, train_y = sequence_data(
        x_scaler.transform(features), y_scaler.transform(targets),
        train_indices, LSTM_WINDOW,
    )
    model = PriceLSTM(train_x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=128, shuffle=False,
    )
    for _ in range(12):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
    return model, x_scaler, y_scaler


def predict_lstm(
    model: PriceLSTM,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    features: pd.DataFrame,
    indices: np.ndarray,
    window: int,
) -> np.ndarray:
    scaled_x = x_scaler.transform(features)
    sequences, _ = sequence_data(
        scaled_x, np.zeros((len(features), 3)), indices, window
    )
    model.eval()
    with torch.no_grad():
        return y_scaler.inverse_transform(
            model(torch.from_numpy(sequences)).numpy()
        )


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = np.abs(actual - predicted)
    close_actual = np.maximum(np.abs(actual[:, 2]), 1e-8)
    mape_close = float(np.mean(errors[:, 2] / close_actual) * 100)
    return {
        "mae_usd": float(np.mean(errors)),
        "mae_high_usd": float(np.mean(errors[:, 0])),
        "mae_low_usd": float(np.mean(errors[:, 1])),
        "mae_close_usd": float(np.mean(errors[:, 2])),
        "mape_close_pct": mape_close,
    }


def chronological_split(n_rows: int, horizon: int) -> dict[str, np.ndarray]:
    holdout_start = n_rows - HOLDOUT_ROWS[horizon]
    validation_end = holdout_start - horizon
    validation_rows = (200 + horizon) if horizon != 1 else max(200, holdout_start // 5)
    validation_start = validation_end - validation_rows
    train_end = validation_start - horizon
    tuning_end = validation_start + (validation_rows - horizon) // 2
    calibration_start = tuning_end + horizon
    if train_end < 2 * LSTM_WINDOW:
        raise ValueError(f"Too few rows for the embargoed {horizon}-bar split.")
    return {
        "train": np.arange(train_end),
        "validation_tuning": np.arange(validation_start, tuning_end),
        "validation_calibration": np.arange(calibration_start, validation_end),
        "holdout": np.arange(holdout_start, n_rows),
    }


def select_ensemble(
    catboost: np.ndarray, secondary: np.ndarray, targets: np.ndarray,
    dollar_vol: np.ndarray, horizon: int,
) -> tuple[dict[str, float], float]:
    best_score = float("inf")
    selected = ({"catboost": 0.0}, 0.0)
    for cat_weight in np.linspace(0.0, 1.0, 11):
        weights = {"catboost": float(cat_weight)}
        for shrinkage in np.linspace(0.0, 1.0, 11):
            base = blend_returns(catboost, secondary, weights, float(shrinkage))
            correction = residual_feedback(targets - base, horizon, RESIDUAL_WINDOW, RESIDUAL_GAIN)
            score = float(np.mean(np.abs(targets[:, 2] - base[:, 2] - correction[:, 2]) * dollar_vol))
            if score < best_score:
                best_score = score
                selected = weights, float(shrinkage)
    return selected


def conformal_calibration(scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Conformal calibration requires finite validation residuals.")
    ordered = np.sort(scores)
    return {str(percent): float(ordered[min(len(ordered), (percent * (len(ordered) + 1) + 99) // 100) - 1])
            for percent in (80, 95)}


def range_coverages(
    actual_close: np.ndarray, predicted_close: np.ndarray, dollar_vol: np.ndarray,
    conformal: dict[str, float],
) -> dict[str, float]:
    coverages = {}
    for level, quantile in conformal.items():
        low, high = range_bounds(predicted_close, dollar_vol, quantile)
        coverages[level] = float(np.mean((actual_close >= low) & (actual_close <= high)) * 100)
    return coverages


def train_horizon(horizon: int, data: pd.DataFrame, features: pd.DataFrame) -> dict:
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("Expected unique, chronologically ordered bars.")

    features = features.ffill()
    target_prices = gold_target_prices(data, horizon)
    vol = realized_volatility(data["Close"], horizon)
    ready = features.notna().all(axis=1) & vol.notna()
    if not ready.any():
        raise ValueError("No complete feature rows. Rebuild both market datasets first.")
    start = ready.to_numpy().argmax()
    # Slice off warm-up and unresolved tail without compressing the trading-bar clock.
    resolved = slice(start, -horizon)
    features, target_prices = features.iloc[resolved], target_prices.iloc[resolved]
    current = data["Close"].iloc[resolved].to_numpy()[:, None]
    volatility = vol.iloc[resolved].to_numpy()[:, None]
    actual = target_prices.to_numpy()
    dollar_vol = current * volatility
    targets = (actual - current) / dollar_vol
    if not np.isfinite(targets).all():
        raise ValueError("Invalid gold prices or normalized targets in the modeling snapshot.")
    split = chronological_split(len(features), horizon)
    train, tune, calibration, holdout = (
        split[name] for name in ("train", "validation_tuning", "validation_calibration", "holdout")
    )

    hourly = horizon == 1
    catboost = CatBoostRegressor(
        iterations=550 if hourly else 500,
        depth=5 if hourly else 2,
        learning_rate=0.03 if hourly else 0.015,
        l2_leaf_reg=3 if hourly else 100,
        loss_function="MultiRMSE", verbose=False, random_seed=SEED,
        allow_writing_files=False, thread_count=2,
    )
    catboost.fit(features.iloc[train], targets[train])
    if hourly:
        lstm, x_scaler, y_scaler = train_lstm(features, targets, train)
        extra = {
            "lstm_state": {name: tensor.detach().cpu().numpy().copy()
                           for name, tensor in lstm.state_dict().items()},
            "lstm_window": LSTM_WINDOW, "x_scaler": x_scaler, "y_scaler": y_scaler,
        }

        def predict_secondary(indices: np.ndarray) -> np.ndarray:
            return predict_lstm(lstm, x_scaler, y_scaler, features, indices, LSTM_WINDOW)
    else:
        scaler = StandardScaler().fit(features.iloc[train])
        ridge = Ridge(alpha=150.0).fit(scaler.transform(features.iloc[train]), targets[train])
        extra = {"ridge": ridge, "ridge_scaler": scaler}

        def predict_secondary(indices: np.ndarray) -> np.ndarray:
            return ridge.predict(scaler.transform(features.iloc[indices]))

    weights, shrinkage = select_ensemble(
        catboost.predict(features.iloc[tune]), predict_secondary(tune), targets[tune],
        dollar_vol[tune, 0], horizon,
    )
    # The finalized blend first makes out-of-sample predictions on calibration.
    # Embargo rows supply no residuals. Feedback crosses boundaries only after maturity.
    base = np.full_like(targets, np.nan)
    base[calibration] = blend_returns(
        catboost.predict(features.iloc[calibration]), predict_secondary(calibration), weights, shrinkage,
    )
    calibration_adjusted = base + residual_feedback(
        targets - base, horizon, RESIDUAL_WINDOW, RESIDUAL_GAIN,
    )
    scores = np.abs(targets[calibration, 2] - calibration_adjusted[calibration, 2])
    conformal = conformal_calibration(scores)

    # Holdout evaluation starts only once every model and calibration parameter is fixed.
    base[holdout] = blend_returns(
        catboost.predict(features.iloc[holdout]), predict_secondary(holdout), weights, shrinkage,
    )
    correction = residual_feedback(targets - base, horizon, RESIDUAL_WINDOW, RESIDUAL_GAIN)
    predicted = current[holdout] + dollar_vol[holdout] * (base[holdout] + correction[holdout])
    actual = actual[holdout]
    naive = np.repeat(current[holdout], 3, axis=1)
    model_metrics, baseline_metrics = metrics(actual, predicted), metrics(actual, naive)
    mae, baseline_mae = model_metrics["mae_close_usd"], baseline_metrics["mae_close_usd"]
    skill = 100.0 * (1 - mae / baseline_mae) if baseline_mae else None
    coverage = range_coverages(actual[:, 2], predicted[:, 2], dollar_vol[holdout, 0], conformal)
    ranges = {}
    for level, quantile in conformal.items():
        low, high = range_bounds(predicted[:, 2], dollar_vol[holdout, 0], quantile)
        ranges[f"range_{level}_low"], ranges[f"range_{level}_high"] = low, high
    importance = pd.Series(catboost.feature_importances_, index=features.columns)
    return {
        "catboost": catboost,
        "features": list(features.columns),
        "top_features": importance.nlargest(5).to_dict(),
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "beats_baseline": mae < baseline_mae,
        "skill_score_pct": skill,
        "range_coverage_80_pct": coverage["80"],
        "range_coverage_95_pct": coverage["95"],
        "conformal_intervals": conformal,
        "ensemble_weights": weights,
        "damping_factor": shrinkage,
        "residual_window": RESIDUAL_WINDOW,
        "residual_gain": RESIDUAL_GAIN,
        "feedback_start": features.index[calibration[0]],
        "feedback_embargo": features.index[calibration[-1] + 1:holdout[0]],
        "audit_split": {
            "train_end": int(train[-1] + 1),
            "validation_tuning_start": int(tune[0]),
            "validation_tuning_end": int(tune[-1] + 1),
            "validation_calibration_start": int(calibration[0]),
            "validation_calibration_end": int(calibration[-1] + 1),
            "holdout_start": int(holdout[0]),
            "holdout_end": int(holdout[-1] + 1),
            "embargo_rows": horizon,
        },
        "audit_index": features.index,
        "holdout_audit": {
            "dates": features.index[holdout],
            "target_dates": data.index[start + holdout + horizon],
            "actual_prices": actual,
            "predicted_prices": predicted,
            "baseline_prices": naive,
            **ranges,
        },
        **extra,
    }


def print_holdout(models: dict[int, dict]) -> None:
    summary = []
    for horizon, model in models.items():
        summary.append({
            "Horizon": f"{horizon}{'H' if horizon == 1 else 'D'}",
            "Close MAE ($)": model["metrics"]["mae_close_usd"],
            "Baseline MAE ($)": model["baseline_metrics"]["mae_close_usd"],
            "MAPE (%)": model["metrics"]["mape_close_pct"],
            "Skill (%)": model["skill_score_pct"],
            "80% coverage": model["range_coverage_80_pct"],
            "95% coverage": model["range_coverage_95_pct"],
        })
    print("\nHoldout Test")
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for row, model in zip(summary, models.values()):
        importance = ", ".join(f"{name} ({value:.2f})" for name, value in model["top_features"].items())
        print(f"{row['Horizon']} top CatBoost importances: {importance}")


def read_snapshots() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    frames, digests = [], {}
    for path in (HOURLY_FILE, DAILY_FILE):
        with path.open("rb") as dataset:
            digests[path.name] = file_digest(dataset, "sha256").hexdigest()
            dataset.seek(0)
            frames.append(pd.read_csv(dataset, parse_dates=["Date"]).set_index("Date"))
    return frames[0], frames[1], digests


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    hourly, daily, digests = read_snapshots()
    hourly_x, daily_x = hourly_features(hourly), daily_features(daily)
    models = {}
    for horizon, data, features in ((1, hourly, hourly_x), (7, daily, daily_x), (30, daily, daily_x)):
        print(f"Training horizon {horizon}...", flush=True)
        models[horizon] = train_horizon(horizon, data, features)
    temporary = BUNDLE_FILE.with_suffix(".pkl.tmp")
    joblib.dump({
        "version": BUNDLE_VERSION, "models": models, "dataset_sha256": digests,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "horizons": (1, 7, 30),
    }, temporary, compress=3)
    temporary.replace(BUNDLE_FILE)
    print_holdout(models)


if __name__ == "__main__":
    main()

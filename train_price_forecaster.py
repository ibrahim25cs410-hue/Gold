from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_dataset import download_daily_indicators
from price_features import (
    RISK_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    blend_returns,
    daily_features,
    flatten,
    hourly_features,
    sequence_window,
)


ROOT = Path(__file__).resolve().parent
HOURLY_FILE = ROOT / "gold_full_dataset.csv"
BUNDLE_FILE = ROOT / "aura_price_models.pkl"
SEED = 42
HOLDOUT_ROWS = {1: 200, 7: 100, 30: 100}
VALIDATION_FRACTION = 0.20
LSTM_WINDOW = 14


def download_daily() -> pd.DataFrame:
    frames = []
    for name, ticker in {
        "Gold": "GC=F",
        "Oil": "CL=F",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
    }.items():
        raw = flatten(
            yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        )
        if raw.empty:
            raise RuntimeError(f"تعذر جلب البيانات اليومية للرمز {ticker}.")
        raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index)).tz_localize(None)
        if name == "Gold":
            part = raw[["Open", "High", "Low", "Close", "Volume"]]
            part = part.rename(
                columns={"High": "Gold_High", "Low": "Gold_Low"}
            )
        else:
            part = raw[["Close"]].rename(columns={"Close": f"{name}_Close"})
        frames.append(part)
    data = frames[0].join(frames[1:], how="left").sort_index()
    data = data.join(download_daily_indicators(data.index)).ffill().dropna()
    return data


def make_targets(data: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = data["Close"]
    if horizon == 1:
        target_prices = data[TARGET_COLUMNS]
    else:
        target_prices = pd.DataFrame(
            {
                "Target_High": data["Gold_High"].shift(-horizon),
                "Target_Low": data["Gold_Low"].shift(-horizon),
                "Target_Close": close.shift(-horizon),
            },
            index=data.index,
        )
    targets = target_prices.div(close, axis=0) - 1.0
    return targets, target_prices


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
    features: pd.DataFrame,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[PriceLSTM, StandardScaler, StandardScaler, int]:
    window = LSTM_WINDOW
    x_scaler = StandardScaler().fit(features.iloc[train_indices])
    y_scaler = StandardScaler().fit(targets[train_indices])
    scaled_x = x_scaler.transform(features)
    scaled_y = y_scaler.transform(targets)
    train_x, train_y = sequence_data(scaled_x, scaled_y, train_indices, window)
    valid_x, valid_y = sequence_data(scaled_x, scaled_y, validation_indices, window)
    model = PriceLSTM(train_x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=128,
        shuffle=False,
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, 101):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(
                model(torch.from_numpy(valid_x)), torch.from_numpy(valid_y)
            ).item()
        scheduler.step(valid_loss)
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    if best_state is None:
        raise RuntimeError("تعذر حفظ أفضل أوزان LSTM.")
    model.load_state_dict(best_state)
    return model, x_scaler, y_scaler, window


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
        "mape_pct": mape_close,
        "accuracy_pct": 100.0 - mape_close,
    }


def conformal_calibration(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float]]:
    """Calibrate finite-sample asymmetric margins on an independent validation slice."""
    def conformal_quantile(values: np.ndarray, level: float) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            raise ValueError("Conformal calibration requires validation residuals.")
        rank = min(values.size, int(np.ceil((values.size + 1) * level)))
        return float(np.partition(values, rank - 1)[rank - 1])

    result: dict[str, dict[str, float]] = {}
    for level in (0.95, 0.99):
        result[f"{int(level * 100)}"] = {
            "close_margin_usd": conformal_quantile(
                np.abs(actual[:, 2] - predicted[:, 2]), level
            ),
            "high_margin_usd": conformal_quantile(
                np.abs(actual[:, 0] - predicted[:, 0]), level
            ),
            "low_margin_usd": conformal_quantile(
                np.abs(actual[:, 1] - predicted[:, 1]), level
            ),
        }
    return result


def chronological_split(n_rows: int, horizon: int) -> dict[str, np.ndarray]:
    holdout_rows = HOLDOUT_ROWS[horizon]
    test_start = n_rows - holdout_rows
    validation_end = test_start - horizon
    validation_rows = max(
        int(np.ceil(test_start * VALIDATION_FRACTION)),
        2 * horizon + 60,
    )
    validation_start = validation_end - validation_rows
    train_end = validation_start - horizon
    validation_size = validation_end - validation_start
    tuning_size = (validation_size - horizon) // 2
    calibration_start = validation_start + tuning_size + horizon

    if (
        train_end < LSTM_WINDOW
        or tuning_size < 20
        or validation_end - calibration_start < 30
        or test_start < validation_end + horizon
    ):
        raise ValueError(
            f"بيانات غير كافية لتقسيم زمني مع embargo للأفق {horizon}."
        )

    return {
        "train": np.arange(0, train_end),
        "validation_tuning": np.arange(validation_start, validation_start + tuning_size),
        "validation_calibration": np.arange(calibration_start, validation_end),
        "holdout": np.arange(test_start, n_rows),
        "embargo": np.asarray([horizon]),
    }


def select_ensemble(
    catboost_returns: np.ndarray,
    secondary_returns: np.ndarray,
    current_prices: np.ndarray,
    actual_prices: np.ndarray,
) -> tuple[dict[str, float], float]:
    best_score = float("inf")
    best_weights: dict[str, float] | None = None
    best_damping = 1.0
    for catboost_weight in np.linspace(0.0, 1.0, 21):
        weights = {"catboost": float(catboost_weight)}
        for damping_factor in np.linspace(0.5, 1.0, 11):
            returns = blend_returns(
                catboost_returns,
                secondary_returns,
                weights,
                float(damping_factor),
            )
            predicted_prices = current_prices * (1.0 + returns)
            score = float(np.mean(np.abs(actual_prices - predicted_prices)))
            if score < best_score:
                best_score = score
                best_weights = weights
                best_damping = float(damping_factor)
    if best_weights is None:
        raise RuntimeError("تعذر اختيار أوزان الدمج من مجموعة التحقق.")
    return best_weights, best_damping


def range_coverages(
    actual: np.ndarray,
    predicted: np.ndarray,
    conformal: dict[str, dict[str, float]],
) -> dict[str, float]:
    coverages = {}
    for level in ("95", "99"):
        low, high = range_bounds(predicted, conformal[level])
        coverages[level] = float(
            np.mean((actual[:, 2] >= low) & (actual[:, 2] <= high)) * 100
        )
    return coverages


def range_bounds(
    predicted: np.ndarray, margins: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    low = np.minimum(
        predicted[:, 1] - margins["low_margin_usd"],
        predicted[:, 2] - margins["close_margin_usd"],
    )
    high = np.maximum(
        predicted[:, 0] + margins["high_margin_usd"],
        predicted[:, 2] + margins["close_margin_usd"],
    )
    return low, high


def train_horizon(
    horizon: int,
    data: pd.DataFrame,
    features: pd.DataFrame,
    comparison_label: str | None = None,
) -> dict:
    targets, target_prices = make_targets(data, horizon)
    frame = features.join(targets).replace([np.inf, -np.inf], np.nan).dropna()
    features = frame[features.columns]
    targets = frame[targets.columns]
    target_prices = target_prices.loc[frame.index]
    close = data.loc[frame.index, "Close"]
    split = chronological_split(len(frame), horizon)
    train_indices = split["train"]
    tuning_indices = split["validation_tuning"]
    calibration_indices = split["validation_calibration"]
    test_indices = split["holdout"]

    if horizon == 1:
        model = CatBoostRegressor(
            iterations=1200,
            depth=7,
            learning_rate=0.03,
            loss_function="MultiRMSE",
            verbose=False,
            random_seed=SEED,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(features.iloc[train_indices], targets.iloc[train_indices])
        lstm, x_scaler, y_scaler, window = train_lstm(
            features, targets.to_numpy(), train_indices, tuning_indices
        )
        cat_tuning = np.asarray(model.predict(features.iloc[tuning_indices])).reshape(-1, 3)
        lstm_tuning = predict_lstm(
            lstm, x_scaler, y_scaler, features, tuning_indices, window
        )
        cat_calibration = np.asarray(
            model.predict(features.iloc[calibration_indices])
        ).reshape(-1, 3)
        lstm_calibration = predict_lstm(
            lstm, x_scaler, y_scaler, features, calibration_indices, window
        )
        cat_test = np.asarray(model.predict(features.iloc[test_indices])).reshape(-1, 3)
        lstm_test = predict_lstm(
            lstm, x_scaler, y_scaler, features, test_indices, window
        )
        secondary_tuning = lstm_tuning
        secondary_calibration = lstm_calibration
        secondary_test = lstm_test
        extra = {
            "lstm_state": {
                name: value.detach().cpu().numpy().copy()
                for name, value in lstm.state_dict().items()
            },
            "lstm_input_size": len(features.columns),
            "lstm_window": window,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        }
    else:
        ridge_scaler = StandardScaler().fit(features.iloc[train_indices])
        ridge = Ridge(alpha=150.0)
        ridge.fit(
            ridge_scaler.transform(features.iloc[train_indices]),
            targets.iloc[train_indices],
        )
        model = CatBoostRegressor(
            iterations=800,
            depth=2,
            l2_leaf_reg=100,
            learning_rate=0.015,
            loss_function="MultiRMSE",
            verbose=False,
            random_seed=SEED,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(features.iloc[train_indices], targets.iloc[train_indices])
        ridge_tuning = ridge.predict(
            ridge_scaler.transform(features.iloc[tuning_indices])
        )
        cat_tuning = np.asarray(model.predict(features.iloc[tuning_indices])).reshape(-1, 3)
        ridge_calibration = ridge.predict(
            ridge_scaler.transform(features.iloc[calibration_indices])
        )
        cat_calibration = np.asarray(
            model.predict(features.iloc[calibration_indices])
        ).reshape(-1, 3)
        ridge_test = ridge.predict(ridge_scaler.transform(features.iloc[test_indices]))
        cat_test = np.asarray(model.predict(features.iloc[test_indices])).reshape(-1, 3)
        secondary_tuning = ridge_tuning
        secondary_calibration = ridge_calibration
        secondary_test = ridge_test
        extra = {
            "ridge": ridge,
            "ridge_scaler": ridge_scaler,
        }

    tuning_current = close.iloc[tuning_indices].to_numpy()[:, None]
    tuning_actual = target_prices.iloc[tuning_indices].to_numpy()
    weights, damping_factor = select_ensemble(
        cat_tuning, secondary_tuning, tuning_current, tuning_actual
    )
    tuning_returns = blend_returns(
        cat_tuning, secondary_tuning, weights, damping_factor
    )
    tuning_predicted = tuning_current * (1.0 + tuning_returns)
    bias_correction = np.mean(tuning_actual - tuning_predicted, axis=0)

    calibration_current = close.iloc[calibration_indices].to_numpy()[:, None]
    calibration_actual = target_prices.iloc[calibration_indices].to_numpy()
    calibration_returns = blend_returns(
        cat_calibration, secondary_calibration, weights, damping_factor
    )
    calibration_predicted = (
        calibration_current * (1.0 + calibration_returns) + bias_correction
    )
    conformal = conformal_calibration(calibration_actual, calibration_predicted)

    current = close.iloc[test_indices].to_numpy()[:, None]
    blended_returns = blend_returns(cat_test, secondary_test, weights, damping_factor)
    predicted = current * (1.0 + blended_returns) + bias_correction
    actual = target_prices.iloc[test_indices].to_numpy()
    naive = np.repeat(current, 3, axis=1)
    model_metrics = metrics(actual, predicted)
    baseline_metrics = metrics(actual, naive)
    coverage = range_coverages(actual, predicted, conformal)
    low_95, high_95 = range_bounds(predicted, conformal["95"])
    low_99, high_99 = range_bounds(predicted, conformal["99"])
    beats_baseline = (
        model_metrics["mae_close_usd"] < baseline_metrics["mae_close_usd"]
    )
    print(
        f"{comparison_label + ' | ' if comparison_label else ''}"
        f"Horizon {horizon} ({'hourly' if horizon == 1 else 'trading days'}): "
        f"train={len(train_indices)} tune={len(tuning_indices)} "
        f"calibration={len(calibration_indices)} holdout={len(test_indices)} | "
        f"Close MAE=${model_metrics['mae_close_usd']:.4f} "
        f"(baseline ${baseline_metrics['mae_close_usd']:.4f}) | "
        f"MAPE={model_metrics['mape_close_pct']:.2f}% | "
        f"Coverage95={coverage['95']:.2f}% "
        f"Coverage99={coverage['99']:.2f}% | beats_baseline={beats_baseline}"
    )
    return {
        "catboost": model,
        "features": list(features.columns),
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "beats_baseline": beats_baseline,
        "range_coverage_95_pct": coverage["95"],
        "range_coverage_99_pct": coverage["99"],
        "range_coverage_pct": coverage["99"],
        "conformal_intervals": conformal,
        "bias_correction_usd": bias_correction,
        "ensemble_weights": weights,
        "damping_factor": damping_factor,
        "audit_split": {
            "train_end": int(train_indices[-1] + 1),
            "validation_tuning_start": int(tuning_indices[0]),
            "validation_tuning_end": int(tuning_indices[-1] + 1),
            "validation_calibration_start": int(calibration_indices[0]),
            "validation_calibration_end": int(calibration_indices[-1] + 1),
            "holdout_start": int(test_indices[0]),
            "holdout_end": int(test_indices[-1] + 1),
            "embargo_rows": horizon,
        },
        "audit_index": frame.index,
        "holdout_audit": {
            "dates": frame.index[test_indices],
            "actual_prices": actual,
            "predicted_prices": predicted,
            "baseline_prices": naive,
            "range_95_low": low_95,
            "range_95_high": high_95,
            "range_99_low": low_99,
            "range_99_high": high_99,
            "metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "range_coverage_95_pct": coverage["95"],
            "range_coverage_99_pct": coverage["99"],
        },
        "source": "hourly" if horizon == 1 else "daily",
        **extra,
    }


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    hourly = pd.read_csv(HOURLY_FILE, parse_dates=["Date"]).set_index("Date")
    daily = download_daily()
    inputs = {
        1: (hourly, hourly_features(hourly)),
        7: (daily, daily_features(daily)),
        30: (daily, daily_features(daily)),
    }
    models = {}
    comparison = {}
    for horizon, (data, risk_features) in inputs.items():
        baseline_features = risk_features.drop(
            columns=RISK_FEATURE_COLUMNS, errors="raise"
        )
        common_rows = risk_features.notna().all(axis=1)
        data = data.loc[common_rows]
        risk_features = risk_features.loc[common_rows]
        baseline_features = baseline_features.loc[common_rows]

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        without_risk = train_horizon(
            horizon, data, baseline_features, comparison_label="Without GVZ/VIX"
        )
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        with_risk = train_horizon(
            horizon, data, risk_features, comparison_label="With GVZ/VIX"
        )
        baseline_mae = without_risk["metrics"]["mae_close_usd"]
        risk_mae = with_risk["metrics"]["mae_close_usd"]
        skill_score = (
            100.0 * (baseline_mae - risk_mae) / baseline_mae
            if baseline_mae
            else 0.0
        )
        comparison[horizon] = {
            "without_risk_close_mae_usd": baseline_mae,
            "with_risk_close_mae_usd": risk_mae,
            "skill_score_pct": skill_score,
            "without_risk_coverage_95_pct": without_risk["range_coverage_95_pct"],
            "without_risk_coverage_99_pct": without_risk["range_coverage_99_pct"],
            "with_risk_coverage_95_pct": with_risk["range_coverage_95_pct"],
            "with_risk_coverage_99_pct": with_risk["range_coverage_99_pct"],
        }
        with_risk["risk_feature_comparison"] = comparison[horizon]
        models[horizon] = with_risk

    comparison_rows = [
        {
            "Horizon": f"{horizon}{'H' if horizon == 1 else 'D'}",
            "Close MAE without GVZ/VIX ($)": result["without_risk_close_mae_usd"],
            "Close MAE with GVZ/VIX ($)": result["with_risk_close_mae_usd"],
            "Skill Score (%)": result["skill_score_pct"],
            "Coverage 95% without/with": (
                f"{result['without_risk_coverage_95_pct']:.2f} / "
                f"{result['with_risk_coverage_95_pct']:.2f}"
            ),
            "Coverage 99% without/with": (
                f"{result['without_risk_coverage_99_pct']:.2f} / "
                f"{result['with_risk_coverage_99_pct']:.2f}"
            ),
        }
        for horizon, result in comparison.items()
    ]
    print("\nPaired risk-feature ablation on identical chronological splits:")
    print(
        pd.DataFrame(comparison_rows).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    joblib.dump(
        {
            "version": 1,
            "models": models,
            "risk_feature_comparison": comparison,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "horizons": (1, 7, 30),
        },
        BUNDLE_FILE,
    )
    print(f"تم حفظ حزمة توقع الأسعار في {BUNDLE_FILE.name}.")


if __name__ == "__main__":
    main()

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

from price_features import TARGET_COLUMNS, daily_features, flatten, hourly_features


ROOT = Path(__file__).resolve().parent
HOURLY_FILE = ROOT / "gold_full_dataset.csv"
BUNDLE_FILE = ROOT / "aura_price_models.pkl"
SEED = 42


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
    return frames[0].join(frames[1:], how="left").sort_index().ffill().dropna()


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
    indices = indices[indices >= window]
    return (
        np.stack([features[i - window : i] for i in indices]).astype(np.float32),
        targets[indices].astype(np.float32),
    )


def train_lstm(
    features: pd.DataFrame,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[PriceLSTM, StandardScaler, StandardScaler, int]:
    window = 14
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
    mae = float(np.mean(np.abs(actual - predicted)))
    mape = float(
        np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8)))
        * 100
    )
    return {
        "mae_usd": mae,
        "mape_pct": mape,
        "accuracy_pct": 100.0 - mape,
    }


def conformal_calibration(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float]]:
    """Calibrate asymmetric high/low margins from validation residuals."""
    result: dict[str, dict[str, float]] = {}
    for level in (0.95, 0.99):
        result[f"{int(level * 100)}"] = {
            "close_margin_usd": float(
                np.quantile(np.abs(actual[:, 2] - predicted[:, 2]), level)
            ),
            "high_margin_usd": float(
                np.quantile(np.abs(actual[:, 0] - predicted[:, 0]), level)
            ),
            "low_margin_usd": float(
                np.quantile(np.abs(actual[:, 1] - predicted[:, 1]), level)
            ),
        }
    return result


def train_horizon(
    horizon: int,
    data: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    targets, target_prices = make_targets(data, horizon)
    frame = features.join(targets).replace([np.inf, -np.inf], np.nan).dropna()
    features = frame[features.columns]
    targets = frame[targets.columns]
    target_prices = target_prices.loc[frame.index]
    close = data.loc[frame.index, "Close"]
    split = len(frame) - (200 if horizon == 1 else 100)
    train_indices = np.arange(split)
    test_indices = np.arange(split, len(frame))
    validation_start = max(20, int(len(train_indices) * 0.9))
    fit_indices = train_indices[:validation_start]
    validation_indices = train_indices[validation_start:]
    if len(validation_indices) < 10:
        validation_indices = fit_indices[-min(20, len(fit_indices)) :]

    if horizon == 1:
        # Keep the proven hourly CatBoost + LSTM path unchanged.
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
            features, targets.to_numpy(), fit_indices, validation_indices
        )
        cat_validation = model.predict(features.iloc[validation_indices])
        lstm_validation = predict_lstm(
            lstm, x_scaler, y_scaler, features, validation_indices, window
        )
        validation_returns = 0.60 * cat_validation + 0.40 * lstm_validation
        cat_returns = model.predict(features.iloc[test_indices])
        lstm_returns = predict_lstm(
            lstm, x_scaler, y_scaler, features, test_indices, window
        )
        blended_returns = 0.60 * cat_returns + 0.40 * lstm_returns
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
        # Longer horizons use regularized, damped returns to avoid level drift.
        ridge_scaler = StandardScaler().fit(features.iloc[fit_indices])
        ridge = Ridge(alpha=150.0)
        ridge.fit(
            ridge_scaler.transform(features.iloc[fit_indices]),
            targets.iloc[fit_indices],
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
        model.fit(features.iloc[fit_indices], targets.iloc[fit_indices])
        ridge_validation = ridge.predict(
            ridge_scaler.transform(features.iloc[validation_indices])
        )
        cat_validation = model.predict(features.iloc[validation_indices])
        ridge_test = ridge.predict(ridge_scaler.transform(features.iloc[test_indices]))
        cat_test = model.predict(features.iloc[test_indices])
        damping = 0.5
        validation_returns = damping * (0.50 * ridge_validation + 0.50 * cat_validation)
        blended_returns = damping * (0.50 * ridge_test + 0.50 * cat_test)
        extra = {
            "ridge": ridge,
            "ridge_scaler": ridge_scaler,
            "damping_factor": damping,
        }
    validation_current = close.iloc[validation_indices].to_numpy()[:, None]
    validation_actual = target_prices.iloc[validation_indices].to_numpy()
    validation_predicted = validation_current * (1.0 + validation_returns)
    if horizon == 1:
        bias_correction = np.zeros(3)
    else:
        # Correct the systematic multi-day level bias using validation only.
        bias_correction = np.mean(validation_actual - validation_predicted, axis=0)
        validation_predicted = validation_predicted + bias_correction
    current = close.iloc[test_indices].to_numpy()[:, None]
    predicted = current * (1.0 + blended_returns) + bias_correction
    actual = target_prices.iloc[test_indices].to_numpy()
    conformal = conformal_calibration(validation_actual, validation_predicted)
    naive = np.repeat(current, 3, axis=1)
    model_metrics = metrics(actual, predicted)
    baseline_metrics = metrics(actual, naive)
    interval = conformal["95"]
    predicted_low = predicted[:, 1] - interval["low_margin_usd"]
    predicted_high = predicted[:, 0] + interval["high_margin_usd"]
    coverage = float(
        np.mean(
            (actual[:, 2] >= predicted_low) & (actual[:, 2] <= predicted_high)
        )
        * 100
    )
    beats_baseline = model_metrics["mae_usd"] < baseline_metrics["mae_usd"]
    print(
        f"Horizon {horizon}: rows={len(frame)} | "
        f"MAE=${model_metrics['mae_usd']:.4f} "
        f"(baseline ${baseline_metrics['mae_usd']:.4f}) | "
        f"MAPE={model_metrics['mape_pct']:.2f}% | "
        f"Accuracy={model_metrics['accuracy_pct']:.2f}% | "
        f"Coverage95={coverage:.2f}% | beats_baseline={beats_baseline}"
    )
    return {
        "catboost": model,
        "features": list(features.columns),
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "beats_baseline": beats_baseline,
        "range_coverage_pct": coverage,
        "conformal_intervals": conformal,
        "bias_correction_usd": bias_correction,
        "source": "hourly" if horizon == 1 else "daily",
        **extra,
    }


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    hourly = pd.read_csv(HOURLY_FILE, parse_dates=["Date"]).set_index("Date")
    hourly_data = hourly.copy()
    hourly_features_data = hourly_features(hourly)
    daily = download_daily()
    daily_features_data = daily_features(daily)
    models = {
        1: train_horizon(1, hourly_data, hourly_features_data),
        7: train_horizon(7, daily, daily_features_data),
        30: train_horizon(30, daily, daily_features_data),
    }
    joblib.dump(
        {
            "version": 1,
            "models": models,
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "horizons": (1, 7, 30),
        },
        BUNDLE_FILE,
    )
    print(f"تم حفظ حزمة توقع الأسعار في {BUNDLE_FILE.name}.")


if __name__ == "__main__":
    main()

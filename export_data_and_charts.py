from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_price_forecaster import BUNDLE_FILE, print_holdout, read_snapshots
from price_features import BUNDLE_VERSION, ECONOMIC_FEATURES, daily_features, hourly_features

ROOT = Path(__file__).resolve().parent
CLEANED_FILE = ROOT / "cleaned_market_data.csv"
DATASET_FILE = ROOT / "aura_gold_dataset.csv"
CHART_DIR = ROOT / "charts"


def _clean_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))
    if data.index.tz is not None:
        data.index = data.index.tz_convert("UTC").tz_localize(None)
    data.index.name = "Date"
    data = data.loc[~data.index.isna()]
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    columns = [
        "Open", "High", "Low", "Close", "Volume", "Oil_Close",
        "US10Y_Yield", "DXY", "GVZ_Close", "VIX_Close", "TIP_Close",
    ]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"الأعمدة الساعية المطلوبة غير موجودة: {missing}")
    data = data[columns].apply(pd.to_numeric, errors="coerce")
    data[["Oil_Close", "US10Y_Yield", "DXY", "GVZ_Close", "VIX_Close"]] = data[
        ["Oil_Close", "US10Y_Yield", "DXY", "GVZ_Close", "VIX_Close"]
    ].ffill()
    return data.dropna(subset=columns)


def _normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))
    if data.index.tz is not None:
        data.index = data.index.tz_convert("UTC").tz_localize(None)
    data.index.name = "Date"
    data = data.loc[~data.index.isna()]
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    columns = [
        "Open",
        "Gold_High",
        "Gold_Low",
        "Close",
        "Volume",
        "Oil_Close",
        "US10Y_Close",
        "DXY_Close",
        "GVZ_Close",
        "VIX_Close",
        "TIP_Close",
    ]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"الأعمدة اليومية المطلوبة غير موجودة: {missing}")
    data = data[columns].apply(pd.to_numeric, errors="coerce")
    data[["Oil_Close", "US10Y_Close", "DXY_Close", "GVZ_Close", "VIX_Close"]] = data[
        ["Oil_Close", "US10Y_Close", "DXY_Close", "GVZ_Close", "VIX_Close"]
    ].ffill()
    return data.dropna(
        subset=columns
    )


def _to_export_frame(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    rename = {
            "Open": "Gold_Open",
            "High": "Gold_High",
            "Low": "Gold_Low",
            "Gold_High": "Gold_High",
            "Gold_Low": "Gold_Low",
            "Close": "Gold_Close",
            "Volume": "Gold_Volume",
            "US10Y_Close": "US10Y_Yield",
            "DXY_Close": "DXY",
        }
    return data.rename(columns=rename).assign(Frequency=frequency).reset_index()


def _forecast_holdout(
    horizon: int,
    features: pd.DataFrame,
    bundle: dict,
) -> dict:
    meta = bundle["models"][horizon]
    audit = meta["holdout_audit"]
    frame_index = pd.DatetimeIndex(meta["audit_index"])
    split = meta["audit_split"]
    return {
        "dates": pd.DatetimeIndex(audit["target_dates"]),
        "actual": audit["actual_prices"],
        "predicted": audit["predicted_prices"],
        "low_80": audit["range_80_low"],
        "high_80": audit["range_80_high"],
        "low_95": audit["range_95_low"],
        "high_95": audit["range_95_high"],
        "metrics": meta["metrics"],
        "baseline_metrics": meta["baseline_metrics"],
        "features": features.reindex(frame_index),
        "train_indices": np.arange(split["train_end"]),
        "frame_index": frame_index,
        "split": split,
    }


def _save_csvs(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    clean_hourly = _to_export_frame(hourly, "hourly")
    clean_daily = _to_export_frame(daily, "daily")
    cleaned = pd.concat([clean_hourly, clean_daily], ignore_index=True, sort=False)
    cleaned = cleaned.sort_values(["Date", "Frequency"]).reset_index(drop=True)
    cleaned.to_csv(CLEANED_FILE, index=False)
    print(
        "Data cleaning:"
        f" hourly {len(hourly):,} rows after cleaning,"
        f" daily {len(daily):,} rows after cleaning;"
        f" combined {len(cleaned):,} rows."
    )
    return cleaned


def _save_feature_dataset(
    hourly: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    hourly_feature_data = hourly_features(hourly)
    daily_feature_data = daily_features(daily)
    hourly_joined = hourly.drop(columns=hourly_feature_data.columns, errors="ignore").join(hourly_feature_data)
    daily_joined = daily.drop(columns=daily_feature_data.columns, errors="ignore").join(daily_feature_data)
    hourly_joined = hourly_joined.drop(
        columns=[c for c in hourly_joined if c.startswith("Target_") or c == "Direction_Label"],
        errors="ignore",
    )
    daily_joined = daily_joined.drop(
        columns=[c for c in daily_joined if c.startswith("Target_")],
        errors="ignore",
    )
    hourly_export = _to_export_frame(hourly_joined, "hourly")
    daily_export = _to_export_frame(daily_joined, "daily")
    result = pd.concat([hourly_export, daily_export], ignore_index=True, sort=False)
    result = result.sort_values(["Date", "Frequency"]).reset_index(drop=True)
    result.to_csv(DATASET_FILE, index=False)
    print(f"Feature dataset: {len(result):,} rows, {len(result.columns):,} columns.")
    return result


def _plots(
    hourly: pd.DataFrame,
    bundle: dict,
    evaluations: dict[int, dict],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as exc:
        raise RuntimeError(
            "يتطلب إنشاء الرسوم تثبيت matplotlib داخل .venv وإضافتها إلى requirements.txt."
        ) from exc

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    background = "#111827"
    panel = "#172033"
    gold = "#f4c542"
    blue = "#60a5fa"
    green = "#34d399"
    red = "#fb7185"
    plt.rcParams.update(
        {
            "figure.facecolor": background,
            "axes.facecolor": panel,
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "grid.color": "#334155",
            "font.size": 10,
        }
    )

    # 01: Chronological partitions matching the hourly model evaluation.
    meta = bundle["models"][1]
    split = meta["audit_split"]
    frame_index = evaluations[1]["frame_index"]
    train_end = split["train_end"]
    tuning_start = split["validation_tuning_start"]
    tuning_end = split["validation_tuning_end"]
    calibration_start = split["validation_calibration_start"]
    calibration_end = split["validation_calibration_end"]
    holdout_start = split["holdout_start"]
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.plot(hourly.index, hourly["Close"], color=gold, linewidth=1.1, label="Gold close")
    ax.axvspan(frame_index[0], frame_index[train_end - 1], color=blue, alpha=0.11, label="Train")
    ax.axvspan(frame_index[tuning_start], frame_index[tuning_end - 1], color=green, alpha=0.16, label="Validation tuning")
    ax.axvspan(frame_index[calibration_start], frame_index[calibration_end - 1], color="#a78bfa", alpha=0.16, label="Conformal calibration")
    ax.axvspan(frame_index[holdout_start], frame_index[-1], color=red, alpha=0.18, label="Holdout test")
    ax.set_title("Causal hourly gold data and embargoed chronological split")
    ax.set_ylabel("Gold price (USD)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.28)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.savefig(CHART_DIR / "01_data_cleaning_overview.png", dpi=300)
    plt.close(fig)

    # 02: CatBoost importance and relationship of leading features to next return.
    features = meta["features"]
    importance = np.asarray(meta["catboost"].feature_importances_).reshape(-1)
    all_importance = pd.Series(importance, index=features).sort_values(ascending=False)
    selected = list(all_importance.head(12).index)
    selected.extend(
        name
        for name in ECONOMIC_FEATURES
        if name in all_importance.index and name not in selected
    )
    ranking = all_importance.loc[selected]
    colors = [
        "#f472b6" if any(tag in name.lower() for tag in ("gvz", "vix"))
        else "#f59e0b" if any(tag in name.lower() for tag in ("dxy", "dollar"))
        else "#a78bfa" if any(tag in name.lower() for tag in ("10y", "us10y", "treasury"))
        else blue
        for name in ranking.index
    ]
    eval1 = evaluations[1]
    train_index = eval1["frame_index"][eval1["train_indices"]]
    hourly_features_frame = eval1["features"].loc[train_index]
    hourly_next_return = hourly["Close"].shift(-1).reindex(train_index) / hourly[
        "Close"
    ].reindex(train_index) - 1
    correlations = []
    for name in ranking.index:
        if name in hourly_features_frame:
            correlations.append(hourly_features_frame[name].corr(hourly_next_return) if hourly_features_frame[name].nunique() > 1 else 0.0)
        else:
            correlations.append(np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    y = np.arange(len(ranking))
    ax1.barh(y, ranking.values, color=colors, alpha=0.9)
    ax1.set_yticks(y, labels=ranking.index)
    ax1.invert_yaxis()
    ax1.set_title("Leading CatBoost features and economic signals")
    ax1.set_xlabel("Relative importance")
    ax1.grid(axis="x", alpha=0.25)
    corr = np.nan_to_num(np.asarray(correlations, dtype=float), nan=0.0)
    corr_colors = [green if value >= 0 else red for value in corr]
    ax2.barh(y, corr, color=corr_colors, alpha=0.9)
    ax2.axvline(0, color="#e2e8f0", linewidth=0.8)
    ax2.set_yticks(y, labels=ranking.index)
    ax2.invert_yaxis()
    ax2.set_title("Feature correlation with next-hour gold return")
    ax2.set_xlabel("Pearson correlation")
    ax2.grid(axis="x", alpha=0.25)
    fig.savefig(CHART_DIR / "02_feature_importance.png", dpi=300)
    plt.close(fig)

    # Align realized closes and forecasts by target date, with a panel per horizon.
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    for ax, horizon, label in zip(axes, (1, 7, 30), ("1H", "7 trading days", "30 trading days")):
        result = evaluations[horizon]
        dates = result["dates"]
        ax.fill_between(dates, result["low_95"], result["high_95"],
                        color=blue, alpha=0.16, label="95% risk interval")
        ax.fill_between(dates, result["low_80"], result["high_80"],
                        color=blue, alpha=0.30, label="80% likely interval")
        ax.plot(dates, result["actual"][:, 2], color="#f8fafc", linewidth=1, label="Actual close")
        ax.plot(dates, result["predicted"][:, 2], color=gold, linewidth=1.2, label="Forecast close")
        ax.set_title(f"{label} holdout close forecasts")
        ax.set_ylabel("Gold close (USD)")
        ax.set_xlabel("Target date")
        ax.grid(alpha=0.28)
        ax.legend(ncol=4, loc="upper left", frameon=False)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.savefig(CHART_DIR / "03_holdout_prediction_intervals.png", dpi=300)
    plt.close(fig)

    # 04: Close-only holdout MAE versus naive baseline.
    horizons = (1, 7, 30)
    x = np.arange(len(horizons))
    width = 0.34
    model_mae = [evaluations[h]["metrics"]["mae_close_usd"] for h in horizons]
    baseline_mae = [
        evaluations[h]["baseline_metrics"]["mae_close_usd"] for h in horizons
    ]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x - width / 2, model_mae, width, label="Model MAE", color=blue)
    ax.bar(x + width / 2, baseline_mae, width, label="Naive baseline MAE", color="#64748b")
    ax.set_xticks(x, labels=["1H", "7 trading days", "30 trading days"])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Close MAE (USD; lower is better)")
    ax.set_title("Holdout close-price error benchmark")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for i, (model_error, baseline_error) in enumerate(zip(model_mae, baseline_mae)):
        ax.text(i - width / 2, model_error, f"{model_error:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, baseline_error, f"{baseline_error:.1f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(CHART_DIR / "04_horizon_accuracy_benchmark.png", dpi=300)
    plt.close(fig)


def main() -> None:
    bundle = joblib.load(BUNDLE_FILE)
    hourly_raw, daily_raw, digests = read_snapshots()
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("dataset_sha256") != digests:
        raise ValueError("Market snapshots do not match the trained bundle. Retrain before exporting.")
    hourly, daily = _clean_hourly(hourly_raw), _normalize_daily(daily_raw)
    print(
        f"Cleaning row counts: hourly {len(hourly_raw):,} to {len(hourly):,}; "
        f"daily {len(daily_raw):,} to {len(daily):,}."
    )
    _save_csvs(hourly, daily)
    _save_feature_dataset(hourly_raw, daily_raw)
    hourly_x, daily_x = hourly_features(hourly_raw).ffill(), daily_features(daily_raw).ffill()
    evaluations = {
        horizon: _forecast_holdout(horizon, hourly_x if horizon == 1 else daily_x, bundle)
        for horizon in (1, 7, 30)
    }
    print_holdout(bundle["models"])
    _plots(hourly, bundle, evaluations)
    print(f"Saved both CSV exports and four 300-DPI charts to {CHART_DIR}.")


if __name__ == "__main__":
    main()

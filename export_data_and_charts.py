from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_price_forecaster import (
    HOURLY_FILE,
    daily_features,
    download_daily,
    hourly_features,
)

ROOT = Path(__file__).resolve().parent
BUNDLE_FILE = ROOT / "aura_price_models.pkl"
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
    columns = ["Open", "High", "Low", "Close", "Volume", "Oil_Close", "US10Y_Yield", "DXY"]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"الأعمدة الساعية المطلوبة غير موجودة: {missing}")
    data = data[columns].apply(pd.to_numeric, errors="coerce")
    data[["Oil_Close", "US10Y_Yield", "DXY"]] = data[
        ["Oil_Close", "US10Y_Yield", "DXY"]
    ].ffill()
    return data.dropna(subset=["Open", "High", "Low", "Close", "Volume", "Oil_Close", "US10Y_Yield", "DXY"])


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
    ]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"الأعمدة اليومية المطلوبة غير موجودة: {missing}")
    data = data[columns].apply(pd.to_numeric, errors="coerce")
    data[["Oil_Close", "US10Y_Close", "DXY_Close"]] = data[
        ["Oil_Close", "US10Y_Close", "DXY_Close"]
    ].ffill()
    return data.dropna(
        subset=["Open", "Gold_High", "Gold_Low", "Close", "Volume", "Oil_Close", "US10Y_Close", "DXY_Close"]
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
        "dates": pd.DatetimeIndex(audit["dates"]),
        "actual": audit["actual_prices"],
        "predicted": audit["predicted_prices"],
        "low_99": audit["range_99_low"],
        "high_99": audit["range_99_high"],
        "coverage_95": audit["range_coverage_95_pct"],
        "coverage_99": audit["range_coverage_99_pct"],
        "metrics": audit["metrics"],
        "baseline_metrics": audit["baseline_metrics"],
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
    hourly_joined = pd.concat(
        [hourly, hourly_feature_data.drop(columns=hourly.columns, errors="ignore")],
        axis=1,
    )
    daily_joined = pd.concat(
        [daily, daily_feature_data.drop(columns=daily.columns, errors="ignore")],
        axis=1,
    )
    hourly_joined = hourly_joined.drop(
        columns=[c for c in hourly_joined if c.startswith("Target_")],
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
    ranking = pd.Series(importance, index=features).sort_values(ascending=False).head(15)
    colors = [
        "#f59e0b" if any(tag in name.lower() for tag in ("dxy", "dollar"))
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
            correlations.append(hourly_features_frame[name].corr(hourly_next_return))
        else:
            correlations.append(np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    y = np.arange(len(ranking))
    ax1.barh(y, ranking.values, color=colors, alpha=0.9)
    ax1.set_yticks(y, labels=ranking.index)
    ax1.invert_yaxis()
    ax1.set_title("Top 15 CatBoost feature importances")
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

    # 03: Holdout forecasts and validation-calibrated 99% intervals.
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    horizon_names = {1: "1H session", 7: "7 trading days", 30: "30 trading days"}
    for horizon, color in ((1, gold), (7, blue), (30, green)):
        result = evaluations[horizon]
        dates = pd.DatetimeIndex(result["dates"])
        if horizon == 30:
            ax.plot(
                dates,
                result["actual"][:, 2],
                color="#f8fafc",
                linewidth=1.2,
                alpha=0.7,
                label="Actual close",
            )
        ax.plot(
            dates,
            result["predicted"][:, 2],
            color=color,
            linewidth=1.5,
            label=f"{horizon_names[horizon]} predicted",
        )
        ax.fill_between(
            dates,
            result["low_99"],
            result["high_99"],
            color=color,
            alpha=0.09,
            label=f"{horizon_names[horizon]} 99% interval",
        )
    ax.set_title("Actual vs predicted gold close with 99% calibrated intervals")
    ax.set_ylabel("Gold price (USD)")
    ax.set_xlabel("Holdout date")
    ax.grid(alpha=0.28)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.savefig(CHART_DIR / "03_actual_vs_predicted_99ci.png", dpi=300)
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
    if not HOURLY_FILE.exists() or not BUNDLE_FILE.exists():
        raise FileNotFoundError("gold_full_dataset.csv و aura_price_models.pkl مطلوبان.")
    bundle = joblib.load(BUNDLE_FILE)
    hourly_raw = pd.read_csv(HOURLY_FILE, parse_dates=["Date"]).set_index("Date")
    daily_raw = download_daily()
    hourly = _clean_hourly(hourly_raw)
    daily = _normalize_daily(daily_raw)
    hourly_model = hourly_raw.loc[hourly.index].copy()
    print(
        "Cleaning row counts:"
        f" hourly before={len(hourly_raw):,}, after={len(hourly):,};"
        f" daily before={len(daily_raw):,}, after={len(daily):,}."
    )
    _save_csvs(hourly, daily)
    _save_feature_dataset(hourly_model, daily)

    evaluations = {}
    summary_rows = []
    for horizon, data in ((1, hourly_model), (7, daily), (30, daily)):
        features = hourly_features(data) if horizon == 1 else daily_features(data)
        result = _forecast_holdout(horizon, features, bundle)
        evaluations[horizon] = result
        summary_rows.append(
            {
                "Horizon": horizon,
                "High_MAE_USD": result["metrics"]["mae_high_usd"],
                "Low_MAE_USD": result["metrics"]["mae_low_usd"],
                "Close_MAE_USD": result["metrics"]["mae_close_usd"],
                "Baseline_Close_MAE_USD": result["baseline_metrics"]["mae_close_usd"],
                "MAPE_Close_Pct": result["metrics"]["mape_close_pct"],
                "Range_Coverage_95_Pct": result["coverage_95"],
                "Range_Coverage_99_Pct": result["coverage_99"],
                "Beats_Baseline": (
                    result["metrics"]["mae_close_usd"]
                    < result["baseline_metrics"]["mae_close_usd"]
                ),
            }
        )
    print("\nFinal training and audit summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _plots(hourly, bundle, evaluations)
    expected = [
        "01_data_cleaning_overview.png",
        "02_feature_importance.png",
        "03_actual_vs_predicted_99ci.png",
        "04_horizon_accuracy_benchmark.png",
    ]
    missing = [name for name in expected if not (CHART_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"لم تُحفظ المخططات المطلوبة: {missing}")
    print(f"\nتم حفظ الملفين و4 مخططات بدقة 300 DPI في {CHART_DIR}.")
    for name in expected:
        print(f"[OK] {name} ({(CHART_DIR / name).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

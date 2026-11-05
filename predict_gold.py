"""Command-line prediction using the same leakage-safe engine as the API."""
import gold_engine as ge


if __name__ == "__main__":
    for horizon in ge.HORIZONS:
        result = ge.predict(horizon)
        print(
            f"{horizon} جلسة: {result['predicted_direction']} | "
            f"{result['predicted_pct_change']:+.3f}% | "
            f"${result['target_price']:.2f}"
        )

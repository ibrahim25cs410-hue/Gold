"""Analyze historical ledger rows without writing prices or changing the ledger."""

import json
import sys
import argparse
from collections import Counter, defaultdict
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .ledger import metrics
from .holdout import in_holdout, ledger_utc, load_bounds


def paired_interval(entries, baseline):
    sessions = defaultdict(list)
    for entry in entries:
        model, reference = entry[2], entry[baseline]
        stamp = ledger_utc(model["target_close"]).replace(tzinfo=timezone.utc)
        local = stamp.astimezone(ZoneInfo("America/Chicago"))
        session = local.date() + timedelta(days=local.hour >= 17)
        sessions[session].append(abs(model["prediction"] - model["actual"])
                                 - abs(reference["prediction"] - reference["actual"]))
    if len(sessions) < 10:
        return None
    blocks = [sessions[day] for day in sorted(sessions)]
    totals, sizes = np.array([sum(block) for block in blocks]), np.array([len(block) for block in blocks])
    rng = np.random.default_rng(42)
    output = {}
    for width in (1, 2):
        starts = rng.integers(len(blocks), size=(3000, (len(blocks) + width - 1) // width, 1))
        indices = ((starts + np.arange(width)) % len(blocks)).reshape(3000, -1)[:, :len(blocks)]
        means = totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)
        output[f"{width}_session_blocks"] = np.quantile(means, [.025, .975]).tolist()
    return output


def analyze(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["group_id"]:
            groups[row["group_id"]].append(row)
    roles = {"original": "champion", "Chronos-2": "foundation:chronos2-zeroshot-hourly",
             "TimesFM 2.5": "foundation:timesfm25-zeroshot-hourly"}
    counts, delays, lags, exceptional = Counter(), [], [], []
    samples = {name: [] for name in roles}
    for entries in groups.values():
        by_role = {row["role"]: row for row in entries}
        base = by_role.get("persistence")
        champion = by_role.get("champion")
        if not champion or champion["version"] != "original" or not champion["observation_close"]:
            continue
        reference, issued, target = map(ledger_utc, (
            champion["observation_close"], champion["issued"], champion["target_close"]))
        offset = (target - reference).total_seconds()
        lead = (target - issued).total_seconds()
        issue_delay = (issued - reference).total_seconds()
        counts["original_groups"] += 1
        counts["unsettled_original"] += champion["actual"] is None
        delays.append(issue_delay)
        if lead > 3600:
            exceptional.append({"reference": str(reference), "issued": str(issued), "target": str(target),
                                "reference_to_target_seconds": offset, "issued_to_target_seconds": lead})
        if champion["observed"]:
            lags.append((ledger_utc(champion["observed"]) - target).total_seconds())
        tag = "fixed_hour" if offset == 3600 else "other_elapsed_horizon"
        counts[tag] += 1
        def aligned(row):
            return (row and row["actual"] is not None
                    and row["issued"] == champion["issued"] and row["target_close"] == champion["target_close"]
                    and row["origin_close"] == champion["origin_close"] and row["actual"] == champion["actual"]
                    and row["actual"] > 0 and row["prediction"] > 0 and row["origin_close"] > 0
                    and row["observed"] and reference <= issued < target <= ledger_utc(row["observed"]))

        if not aligned(base):
            counts["missing_or_misaligned_baseline"] += 1
            continue
        drift = by_role.get("drift")
        move = base["actual"] - base["origin_close"]
        regime = "stable" if abs(move) <= 1 else "up" if move > 0 else "down"
        for name, role in roles.items():
            candidate = by_role.get(role)
            if aligned(candidate):
                samples[name].append((tag, regime, candidate, base, drift if aligned(drift) else None))

    def report(entries):
        def score(rows):
            return metrics([dict(row, reference=row["origin_close"], lower80=None, upper80=None) for row in rows])

        drift_paired = [entry for entry in entries if entry[4] is not None]
        return {"model": score([entry[2] for entry in entries]),
                "persistence": score([entry[3] for entry in entries]),
                "paired_mae_ci_vs_persistence": paired_interval(entries, 3),
                "paired_mae_ci_vs_drift": paired_interval(drift_paired, 4),
                "drift_paired": {"model": score([entry[2] for entry in drift_paired]),
                                 "drift": score([entry[4] for entry in drift_paired])}}

    return {"rows": len(rows), "without_group": sum(not r["group_id"] for r in rows),
            "counts": dict(counts), "exceptional": exceptional,
            "issue_delay_bounds_seconds": [min(delays), max(delays)] if delays else None,
            "observed_delay_bounds_seconds": [min(lags), max(lags)] if lags else None,
            "scores": {name: {"all_legacy": report(entries),
                              "fixed_hour": report([e for e in entries if e[0] == "fixed_hour"]),
                              "fixed_hour_by_move": {key: report([e for e in entries if e[0] == "fixed_hour" and e[1] == key])
                                                     for key in ("up", "down", "stable")}}
                       for name, entries in samples.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    rows = json.load(sys.stdin)
    if args.protocol:
        bounds = load_bounds(args.protocol)
        rows = [row for row in rows if in_holdout(row["issued"], bounds)]
    print(json.dumps(analyze(rows), indent=2))

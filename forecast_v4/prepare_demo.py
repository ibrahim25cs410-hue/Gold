"""Prepare an offline, relative-price display from real settled GC=F records."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from .ledger import metrics
from .holdout import ledger_utc, load_bounds

MODELS = {
    "original": "champion",
    "Chronos-2": "foundation:chronos2-zeroshot-hourly",
    "drift20": "drift",
    "persistence": "persistence",
}


def prepare(rows, captured, period):
    groups = defaultdict(dict)
    rejected = Counter()
    conflicts = set()
    for row in rows:
        group = row["group_id"]
        if not group:
            rejected["no_group"] += 1
            continue
        role = row["role"]
        if role in groups[group]:
            conflicts.add(group)
        groups[group][role] = row
    records = []
    for group, members in groups.items():
        champion = members.get("champion")
        if group in conflicts or not champion or champion["version"] != "original":
            rejected["invalid_group_or_model"] += 1
            continue
        ref, issue, target = (ledger_utc(champion[key])
                              for key in ("observation_close", "issued", "target_close"))
        if period and not period[0] <= issue < period[1]:
            continue
        if target - ref != timedelta(hours=1) or not ref <= issue < target:
            rejected["different_horizon_or_timing"] += 1
            continue
        if champion["actual"] is None:
            rejected["unsettled"] += 1
            continue
        baseline = members.get("persistence")

        def aligned(row):
            return (row and all(row[k] == champion[k] for k in ("issued", "target_close", "actual", "origin_close"))
                    and row["actual"] > 0 and row["origin_close"] > 0
                    and row["prediction"] is not None and row["prediction"] > 0
                    and row["observed"] and ledger_utc(row["observed"]) >= target)

        if not aligned(champion) or not aligned(baseline) or baseline["prediction"] != champion["origin_close"]:
            rejected["missing_or_misaligned_baseline"] += 1
            continue
        available = {name: members[role] for name, role in MODELS.items() if aligned(members.get(role))}
        records.append((champion, available))
    records.sort(key=lambda entry: entry[0]["target_close"])
    cohorts = {"original": ("original", "drift20", "persistence"),
               "common": ("original", "Chronos-2", "drift20", "persistence")}
    report = {"captured": captured, "scope": "interim" if period else "historical", "excluded": dict(rejected),
              "groups_read": len(groups), "cohorts": {}}
    for cohort, names in cohorts.items():
        selected = [(champion, models) for champion, models in records if all(name in models for name in names)]
        scores = {name: metrics([dict(models[name], reference=champion["origin_close"], lower80=None, upper80=None)
                                 for champion, models in selected]) for name in names}
        points = [
            {"reference": champion["observation_close"], "issue": champion["issued"],
             "target": champion["target_close"], "observed": champion["observed"],
             "actual": champion["actual"] - champion["origin_close"],
             "predicted": {name: models[name]["prediction"] - champion["origin_close"] for name in names}}
            for champion, models in selected
        ]
        report["cohorts"][cohort] = {"scores": scores, "rows": points}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captured", required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--holdout", action="store_true")
    scope.add_argument("--before-holdout", action="store_true")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    period = None
    rows = json.load(sys.stdin)
    if args.holdout or args.before_holdout:
        bounds = load_bounds(Path(__file__).with_name("hourly_holdout.json"))
        if args.holdout:
            period = bounds
        else:
            rows = [row for row in rows if ledger_utc(row["issued"]) < bounds[0]]
    report = prepare(rows, args.captured, period)
    variable = "AURA_INTERIM" if args.holdout else "AURA_DEMO"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, suffix=".tmp",
                            delete=False) as output:
        output.write(f"window.{variable} = " + json.dumps(report, allow_nan=False) + ";\n")
    Path(output.name).replace(args.output)
    print(json.dumps({key: len(value["rows"]) for key, value in report["cohorts"].items()}))


if __name__ == "__main__":
    main()

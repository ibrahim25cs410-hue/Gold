"""UTC handling for the frozen hourly evaluation protocol and ledger."""

import json
from datetime import datetime, timezone


def ledger_utc(value):
    """Normalize to naive UTC, the convention of the existing ledger."""
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        return stamp
    return stamp.astimezone(timezone.utc).replace(tzinfo=None)


def load_bounds(path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    start = ledger_utc(protocol["start_utc"])
    end = ledger_utc(protocol["end_utc"])
    if start >= end:
        raise ValueError("Hourly holdout start must precede its end")
    return start, end


def in_holdout(issued, bounds):
    start, end = bounds
    return start <= ledger_utc(issued) < end

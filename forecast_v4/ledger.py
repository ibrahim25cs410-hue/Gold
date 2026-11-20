"""Read and append versioned forecasts without changing the legacy ledger."""

from contextlib import contextmanager
from math import sqrt
import sqlite3

from .market import HORIZONS, stamp


@contextmanager
def connect(path, readonly=False):
    if readonly:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=8)
    else:
        db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        with db:
            yield db
    finally:
        db.close()


def initialize(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY,
            horizon TEXT NOT NULL CHECK (horizon IN ('15m','30m','1h','daily_close')),
            instrument TEXT NOT NULL, source TEXT NOT NULL, contract TEXT NOT NULL,
            definition TEXT NOT NULL, model TEXT NOT NULL, revision TEXT NOT NULL,
            release_hash TEXT NOT NULL, context_hash TEXT NOT NULL,
            origin_utc TEXT NOT NULL, issued_utc TEXT NOT NULL,
            target_utc TEXT NOT NULL, target_label TEXT NOT NULL,
            reference REAL NOT NULL CHECK (reference > 0),
            prediction REAL NOT NULL CHECK (prediction > 0),
            lower80 REAL, upper80 REAL,
            CHECK (origin_utc <= issued_utc AND issued_utc < target_utc),
            CHECK ((lower80 IS NULL AND upper80 IS NULL) OR
                   (lower80 IS NOT NULL AND upper80 IS NOT NULL AND lower80 <= upper80)),
            UNIQUE (instrument, horizon, origin_utc)
        );
        CREATE TABLE IF NOT EXISTS outcomes (
            issue_id INTEGER PRIMARY KEY REFERENCES issues(id),
            actual REAL NOT NULL CHECK (actual > 0),
            source_available_utc TEXT NOT NULL, observed_utc TEXT NOT NULL,
            CHECK (source_available_utc <= observed_utc)
        );
        CREATE TABLE IF NOT EXISTS status (
            id INTEGER PRIMARY KEY CHECK (id=1), checked_utc TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS worker_events (
            id INTEGER PRIMARY KEY, at_utc TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS market_identity (
            id INTEGER PRIMARY KEY CHECK (id=1), instrument TEXT NOT NULL, intraday_definition TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS by_due ON issues(target_utc);
        CREATE TRIGGER IF NOT EXISTS verify_actual_order BEFORE INSERT ON outcomes
        WHEN NEW.source_available_utc < (SELECT target_utc FROM issues WHERE id=NEW.issue_id)
        BEGIN SELECT RAISE(ABORT, 'outcome precedes target'); END;
    """)
    for table in ("issues", "outcomes"):
        for action in ("UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS prevent_{table}_{action} BEFORE {action} ON {table}
                           BEGIN SELECT RAISE(ABORT, 'immutable audit record'); END""")


def pending(db, at):
    return db.execute("""SELECT i.* FROM issues i LEFT JOIN outcomes o ON o.issue_id=i.id
                         WHERE o.issue_id IS NULL AND i.target_utc<=? ORDER BY i.target_utc""", (stamp(at),))


def history(db, limit=100):
    return [dict(row) for row in db.execute("""
        SELECT i.*, o.actual, o.observed_utc, o.source_available_utc,
               abs(i.prediction-o.actual) AS absolute_error,
               CASE WHEN o.actual IS NOT NULL THEN 'settled'
                    WHEN i.target_utc <= strftime('%Y-%m-%dT%H:%M:%f','now') || '000+00:00'
                    THEN 'awaiting_actual' ELSE 'issued' END AS forecast_status
        FROM issues i
        LEFT JOIN outcomes o ON o.issue_id=i.id ORDER BY i.issued_utc DESC, i.id DESC LIMIT ?
    """, (limit,))]


def latest_issues(db):
    return {row["horizon"]: dict(row) for row in db.execute("""
        SELECT i.*, o.actual, o.observed_utc, o.source_available_utc FROM issues i
        LEFT JOIN outcomes o ON o.issue_id=i.id
        WHERE i.id=(SELECT j.id FROM issues j WHERE j.horizon=i.horizon
                    ORDER BY j.issued_utc DESC, j.id DESC LIMIT 1)
    """)}


def metrics(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "baseline_mae": None,
                "mape": None, "smape": None, "directional_accuracy": None, "directional_n": 0,
                "baseline": None, "improvement_pct": None, "coverage80": None, "interval_n": 0}
    def sign(value):
        return (value > 0) - (value < 0)

    def point_metrics(field):
        differences = [row[field] - row["actual"] for row in rows]
        return {
            "mae": sum(map(abs, differences)) / n,
            "rmse": sqrt(sum(error**2 for error in differences) / n),
            "bias": sum(differences) / n,
            "mape": 100 * sum(abs(error) / row["actual"] for error, row in zip(differences, rows)) / n,
            "smape": 200 * sum(abs(error) / (abs(row[field]) + abs(row["actual"]))
                               for error, row in zip(differences, rows)) / n,
            "directional_accuracy": 100 * sum(
                sign(row[field] - row["reference"]) == sign(row["actual"] - row["reference"])
                for row in rows
            ) / n,
            "directional_n": n,
        }
    scores = point_metrics("prediction")
    baseline = point_metrics("reference")
    intervals = [row for row in rows if row["lower80"] is not None and row["upper80"] is not None]
    return {"n": n, **scores, "baseline": baseline, "baseline_mae": baseline["mae"],
            "improvement_pct": 100 * (1-scores["mae"]/baseline["mae"]) if baseline["mae"] else None,
            "coverage80": sum(r["lower80"] <= r["actual"] <= r["upper80"] for r in intervals) / len(intervals)
            if intervals else None, "interval_n": len(intervals)}


def performance(db):
    rows = db.execute("""
        SELECT horizon, definition, model, revision, source, instrument, prediction, reference,
               actual, lower80, upper80
        FROM issues i JOIN outcomes o ON o.issue_id=i.id ORDER BY i.issued_utc
    """)
    groups = {}
    dimensions = ("horizon", "definition", "model", "revision", "source", "instrument")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in dimensions), []).append(row)
    return [
        {**dict(zip(dimensions, keys)), **metrics(items)}
        for keys, items in groups.items() if keys[0] in HORIZONS
    ]


def legacy_history(path, limit=30):
    """Legacy horizon labels are retained verbatim in a read-only archive endpoint."""
    with connect(path, readonly=True) as db:
        rows = db.execute("""
            SELECT f.id, f.issued, f.origin, f.h AS legacy_horizon, f.version, f.target_close,
                   f.origin_close, f.prediction, o.actual, o.observed
            FROM forecasts f LEFT JOIN outcomes o ON o.id=f.id
            ORDER BY f.issued DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]

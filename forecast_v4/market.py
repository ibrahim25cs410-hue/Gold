"""Validate provider snapshots against an approved, independently versioned session clock."""

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite
import re


HORIZONS = {"15m": 15, "30m": 30, "1h": 60, "daily_close": None}
YAHOO_DAILY = "yahoo_gc_f_daily_unadjusted_close"
TRADE_CLOSE = "trade_bar_close"
SPOT = "XAU/USD"
SPOT_SIDES = {"bid", "ask", "mid"}
DATED_GC = re.compile(r"GC[FGHJKMNQUVXZ]\d{1,4}")


def utc(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("Explicit timezone is required")
    return stamp.astimezone(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError("Invalid price")
    return float(value)


@dataclass(frozen=True)
class Session:
    label: str
    opened: datetime
    closed: datetime

    def slots(self, minutes):
        start = self.opened
        while start < self.closed:
            end = min(start + timedelta(minutes=minutes), self.closed)
            yield start, end
            start = end


@dataclass(frozen=True)
class Bar:
    start: datetime
    end: datetime
    available: datetime
    close: float
    contract: str
    opening: float
    high: float
    low: float
    volume: float | None


@dataclass(frozen=True)
class Target:
    horizon: str
    definition: str
    due: datetime
    label: str
    steps: int


def unique_rows(rows):
    """Keep identical repetitions and quarantine every conflicting key."""
    accepted, conflicts = {}, set()
    for key, row in rows:
        if key in conflicts:
            continue
        if key in accepted and accepted[key] != row:
            accepted.pop(key)
            conflicts.add(key)
        else:
            accepted[key] = row
    return accepted, len(conflicts)


class Market:
    def __init__(self, settings, payload, at):
        self.at = at
        self.source = settings.get("source")
        self.instrument = settings.get("instrument", "GC=F")
        if self.instrument not in {"GC=F", "COMEX:GC", SPOT}:
            raise ValueError("Unsupported instrument")
        self.price_side = settings.get("price_side")
        if self.instrument == SPOT and self.price_side not in SPOT_SIDES:
            raise ValueError("Spot requires a fixed bid, ask, or mid price")
        self.intraday_definition = (f"spot_{self.price_side}_bar_close"
                                    if self.instrument == SPOT else TRADE_CLOSE)
        self.daily_definition = daily_definition(settings)
        self.expirations = {contract: utc(expiration) for contract, expiration
                            in settings.get("contract_expiry_utc", {}).items()}
        self.minutes = settings.get("bar_minutes", 60)
        if self.minutes not in (5, 15, 30, 60):
            raise ValueError("Unsupported source resolution")
        self.interval = timedelta(minutes=self.minutes)
        self.lag = timedelta(seconds=settings.get("max_origin_lag_seconds", 240))
        if not timedelta(0) < self.lag < self.interval:
            raise ValueError("Origin latency must be shorter than a bar")
        self.calendar_from = utc(settings["calendar_from"]) if settings.get("calendar_from") else None
        self.calendar_until = utc(settings["calendar_until"]) if settings.get("calendar_until") else None
        sessions = []
        for entry in settings.get("sessions", []):
            label = date.fromisoformat(entry["date"]).isoformat()
            opened, closed = utc(entry["open"]), utc(entry["close"])
            if opened >= closed:
                raise ValueError("Empty market session")
            sessions.append(Session(label, opened, closed))
        self.sessions = sorted(sessions, key=lambda s: s.opened)
        self.by_label = {session.label: session for session in self.sessions}
        if len(self.by_label) != len(self.sessions) or any(
            left.closed > right.opened for left, right in zip(self.sessions, self.sessions[1:])
        ):
            raise ValueError("Calendar sessions overlap or their dates repeat")
        self.schedule = {end: (start, session) for session in self.sessions for start, end in session.slots(self.minutes)}
        self.ends = list(self.schedule)
        self.active = next((s for s in self.sessions if s.opened <= at < s.closed), None)
        if payload and (payload["source"] != self.source or payload["instrument"] != self.instrument):
            raise ValueError("Unexpected source or instrument")
        self.rejected = 0
        self.cleaning_log = []
        intraday = []
        for index, raw in enumerate(payload.get("bars", [])):
            try:
                start, end, available = (utc(raw[k]) for k in ("start", "end", "available_at"))
                contract = raw.get("contract")
                if self.instrument == SPOT:
                    if raw["price_side"] != self.price_side or contract not in (None, SPOT):
                        raise ValueError("Spot quote side or instrument differs")
                    contract = SPOT
                elif (not isinstance(contract, str) or
                      (not DATED_GC.fullmatch(contract) and not (self.instrument == "GC=F" and contract == "GC=F"))):
                    raise ValueError("Contract or provider series code required")
                if raw["complete"] is not True or not start < end <= available <= at:
                    raise ValueError("Unfinished or unavailable bar")
                if self.instrument == "COMEX:GC":
                    expiry = self.expirations.get(contract)
                    if expiry is None or end > expiry:
                        raise ValueError("Missing expiry or bar after contract's last trading time")
                if end not in self.schedule or self.schedule[end][0] != start:
                    raise ValueError("Bar outside verified market schedule")
                price = positive(raw["close"])
                low, high, opening = (positive(raw[k]) for k in ("low", "high", "open"))
                if not low <= min(opening, price) <= max(opening, price) <= high:
                    raise ValueError("Invalid OHLC")
                volume = raw.get("volume")
                if volume is not None and (isinstance(volume, bool) or not isinstance(volume, (int, float))
                        or not isfinite(volume) or volume < 0 or not settings.get("volume_kind")):
                    raise ValueError("Volume value or definition invalid")
                intraday.append((end, Bar(start, end, available, price, contract, opening, high, low, volume)))
            except (KeyError, TypeError, ValueError) as error:
                self.rejected += 1
                self.cleaning_log.append({"feed": "bars", "row": index, "reason": str(error)})
        self.bars, conflicts = unique_rows(intraday)
        self.rejected += conflicts
        self.cleaning_log.extend({"feed": "bars", "end_utc": stamp(end),
                                  "reason": "duplicate" if end in self.bars else "conflicting observations"}
                                 for end, count in Counter(end for end, _ in intraday).items() if count > 1)
        daily = []
        for raw in payload.get("daily", []):
            try:
                session = self.by_label[raw["session_date"]]
                seen = utc(raw["available_at"])
                if (not self.daily_definition or raw["definition"] != self.daily_definition
                        or raw["final"] is not True or not session.closed <= seen <= at):
                    raise ValueError("Daily label unavailable or wrong price definition")
                daily.append((session.label, (positive(raw["close"]), seen)))
            except (KeyError, TypeError, ValueError):
                self.rejected += 1
        self.daily, conflicts = unique_rows(daily)
        self.rejected += conflicts

    @property
    def state(self):
        if not self.calendar_from or not self.calendar_until or not self.calendar_from <= self.at < self.calendar_until:
            return "unknown"
        return "open" if self.active else "closed"

    def latest(self):
        return self.bars[max(self.bars)] if self.bars else None

    def set_time(self, at):
        self.at = at
        self.active = next((s for s in self.sessions if s.opened <= at < s.closed), None)

    def origin(self):
        if self.state != "open":
            return None
        expected = max((end for end in self.ends if self.active.opened < end <= self.at), default=None)
        bar = self.bars.get(expected)
        if bar and self.at - bar.end <= self.lag and bar.end - bar.start == self.interval:
            return bar
        return None

    def context(self, origin, length):
        position = self.ends.index(origin.end)
        ends = self.ends[max(0, position + 1 - length):position + 1]
        bars = [self.bars.get(end) for end in ends]
        if len(bars) != length or any(
            bar is None or bar.available > self.at or bar.contract != origin.contract
            or bar.end - bar.start != self.interval
            for bar in bars
        ):
            return None
        return [bar.close for bar in bars]

    def target(self, horizon, origin):
        minutes = HORIZONS[horizon]
        if minutes is None:
            if not self.daily_definition:
                return None
            due = self.active.closed
            duration = (due - origin.end).total_seconds() / 60
            if duration <= 0 or duration % self.minutes:
                return None
            return Target(horizon, self.daily_definition, due, self.active.label, int(duration // self.minutes))
        due = origin.end + timedelta(minutes=minutes)
        if minutes % self.minutes or due > self.active.closed or due not in self.schedule:
            return None
        return Target(horizon, self.intraday_definition, due, stamp(due), minutes // self.minutes)

    def actual(self, row):
        due = utc(row["target_utc"])
        if self.at < due or row["source"] != self.source:
            return None
        if row["horizon"] == "daily_close":
            return self.daily.get(row["target_label"]) if row["definition"] == self.daily_definition else None
        if row["definition"] != self.intraday_definition:
            return None
        bar = self.bars.get(due)
        if bar and bar.contract == row["contract"]:
            return bar.close, bar.available
        return None


def daily_definition(settings):
    if settings.get("instrument") == SPOT:
        revision = settings.get("spot_daily_close_revision")
        if revision and re.fullmatch(r"[A-Za-z0-9_.-]+", revision):
            return f"spot_{settings['price_side']}_provider_close:{revision}"
    if settings.get("source") == "Yahoo Finance" and settings.get("instrument", "GC=F") == "GC=F":
        return YAHOO_DAILY
    return None

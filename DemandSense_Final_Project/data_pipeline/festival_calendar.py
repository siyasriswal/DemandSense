"""
Festival and wedding-season calendar.

The festival dates for 2022 to 2024 are the ones used to generate the dataset.
Dates for 2025 to 2027 are approximate (festival dates move every year) and are
only used when we forecast beyond the last day of data. Update them from an
official calendar before using the forecasts for real decisions.
"""
import numpy as np
import pandas as pd

from config import FESTIVAL_WINDOW, WEDDING_MONTHS

FESTIVALS = {
    "Diwali":                  ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"],
    "Durga Puja / Navratri":   ["2022-10-01", "2023-10-20", "2024-10-09", "2025-10-01", "2026-10-19", "2027-10-08"],
    "Ganesh Chaturthi":        ["2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14", "2027-09-04"],
    "Holi":                    ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"],
    "Eid":                     ["2022-05-02", "2023-04-21", "2024-04-10", "2025-03-31", "2026-03-20", "2027-03-10"],
}

CALENDAR_START = pd.Timestamp("2022-01-01")
CALENDAR_END = pd.Timestamp("2027-12-31")

_FESTIVAL_DAYS = np.array(sorted(
    pd.to_datetime(d).value // 86_400_000_000_000
    for dates in FESTIVALS.values() for d in dates
))


def festival_score(dates, window=FESTIVAL_WINDOW):
    """0 to 1 score: 1.0 on a festival day, falling in a straight line to 0 at `window` days away."""
    days = pd.DatetimeIndex(pd.to_datetime(dates)).values.astype("datetime64[D]").astype(np.int64)
    nearest = np.abs(days[:, None] - _FESTIVAL_DAYS[None, :]).min(axis=1)
    score = np.where(nearest > window, 0.0, np.maximum(0.0, 1 - nearest / window))
    return np.round(score, 2)


def wedding_season(dates):
    months = pd.DatetimeIndex(pd.to_datetime(dates)).month
    return np.isin(months, list(WEDDING_MONTHS)).astype(int)


def festival_name_for(date, window=FESTIVAL_WINDOW):
    """Name of the festival within `window` days of `date`, or None."""
    d = pd.Timestamp(date)
    for name, dates in FESTIVALS.items():
        for fd in dates:
            if abs((pd.Timestamp(fd) - d).days) <= window:
                return name
    return None


def upcoming_festivals(from_date, count=3):
    """Next `count` festival dates on or after `from_date`, as a list of (date, name)."""
    start = pd.Timestamp(from_date)
    found = [(pd.Timestamp(d), name) for name, ds in FESTIVALS.items() for d in ds if pd.Timestamp(d) >= start]
    return sorted(found)[:count]


def covered(dates):
    """True if every date falls inside the years the calendar knows about."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return bool(d.min() >= CALENDAR_START and d.max() <= CALENDAR_END)

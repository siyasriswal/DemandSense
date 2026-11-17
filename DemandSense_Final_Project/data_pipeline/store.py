"""
Data store: keeps the dataset that the models train on ("active dataset") and
handles everything that changes it:
    * uploading a whole CSV (replace the data, or append to it)
    * typing in a day's sales from the dashboard
    * resetting back to the original dataset

The original file in data/raw is never edited.
"""
import filecmp
import shutil
import threading
from datetime import datetime

import pandas as pd

from config import ACTIVE_CSV, MIN_ROWS_PER_ITEM, RAW_CSV, SALES_LOG
from data_pipeline.festival_calendar import CALENDAR_END, CALENDAR_START, festival_score, wedding_season

COLUMNS = ["date", "category", "item", "demand_units_kg", "is_wedding_season",
           "festival_proximity_score", "day_of_week", "month", "year"]

# Column names we accept in uploaded files (after lower-casing and replacing spaces with _).
ALIASES = {
    "date": ["date", "day", "ds", "sale_date", "sales_date", "timestamp"],
    "item": ["item", "product", "product_name", "name", "commodity", "sku"],
    "demand_units_kg": ["demand_units_kg", "demand", "kg", "quantity", "qty", "units", "sales", "sold",
                        "sold_kg", "usage", "units_sold", "sales_kg", "daily_sales", "daily_usage"],
    "category": ["category", "type", "group"],
}

_lock = threading.Lock()


class DataError(ValueError):
    """Raised when an uploaded file or entry cannot be used. The message is shown to the user."""


def _clean_name(c):
    return str(c).strip().lower().replace(" ", "_")


def _item_key(name):
    return str(name).strip().lower().replace(" ", "_")


def ensure_active():
    ACTIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ACTIVE_CSV.exists():
        shutil.copyfile(RAW_CSV, ACTIVE_CSV)


def load():
    ensure_active()
    return pd.read_csv(ACTIVE_CSV, parse_dates=["date"])


def _save(df):
    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    tmp = ACTIVE_CSV.with_suffix(".tmp")
    out[COLUMNS].to_csv(tmp, index=False)
    tmp.replace(ACTIVE_CSV)


def _add_calendar_columns(df):
    df = df.copy()
    df["is_wedding_season"] = wedding_season(df["date"])
    df["festival_proximity_score"] = festival_score(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def _category_lookup(existing):
    return {_item_key(i): (i, c) for i, c in existing[["item", "category"]].drop_duplicates().itertuples(index=False)}


def normalize(raw, existing=None):
    """
    Turn an uploaded table into the standard layout.
    Returns (clean_dataframe, report). Raises DataError when the file is unusable.
    """
    if raw is None or raw.empty:
        raise DataError("The file is empty.")
    raw = raw.copy()
    raw.columns = [_clean_name(c) for c in raw.columns]

    chosen = {}
    for target, names in ALIASES.items():
        for n in names:
            if n in raw.columns:
                chosen[target] = n
                break
    missing = [c for c in ("date", "item", "demand_units_kg") if c not in chosen]
    if missing:
        raise DataError(
            f"Could not find column(s): {', '.join(missing)}. The file needs a date column, an item column "
            f"and a quantity column (for example: date, item, demand_units_kg). Found: {', '.join(raw.columns)}")

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[chosen["date"]], errors="coerce", dayfirst=False),
        "item": raw[chosen["item"]].astype(str).str.strip(),
        "demand_units_kg": pd.to_numeric(raw[chosen["demand_units_kg"]], errors="coerce"),
    })
    df["category"] = raw[chosen["category"]].astype(str).str.strip() if "category" in chosen else None

    report = {"rows_in": int(len(df)), "dropped": {}, "warnings": [], "new_items": [], "column_map": chosen}

    bad_date = df["date"].isna()
    bad_value = df["demand_units_kg"].isna()
    negative = df["demand_units_kg"] < 0
    bad_item = df["item"].isin(["", "nan", "None"])
    drop = bad_date | bad_value | negative | bad_item
    for label, mask in (("bad_date", bad_date), ("bad_quantity", bad_value & ~bad_date),
                        ("negative_quantity", negative), ("blank_item", bad_item)):
        if mask.sum():
            report["dropped"][label] = int(mask.sum())
    df = df[~drop].copy()
    if df.empty:
        raise DataError("No usable rows left after removing invalid dates and quantities.")

    # Match item names to ones we already know (case and spaces ignored) so "paneer veg mix" is not a new item.
    known = _category_lookup(existing) if existing is not None and not existing.empty else {}
    canonical, cats = [], []
    for item, cat in zip(df["item"], df["category"]):
        key = _item_key(item)
        if key in known:
            name, known_cat = known[key]
            canonical.append(name)
            cats.append(known_cat)
        else:
            canonical.append(item.replace(" ", "_"))
            cats.append(cat if isinstance(cat, str) and cat not in ("", "nan", "None") else "Other")
    df["item"], df["category"] = canonical, cats
    report["new_items"] = sorted(set(df["item"]) - {v[0] for v in known.values()})

    dup = df.duplicated(["date", "item"], keep="last")
    if dup.sum():
        report["dropped"]["duplicate_date_item"] = int(dup.sum())
        report["warnings"].append(
            f"{int(dup.sum())} duplicate rows (same date and item) were found. Only the last one of each was kept.")
        df = df[~dup]

    df["demand_units_kg"] = df["demand_units_kg"].round(1)
    df = _add_calendar_columns(df.assign(date=df["date"].dt.normalize()))

    if df["date"].min() < CALENDAR_START or df["date"].max() > CALENDAR_END:
        report["warnings"].append(
            f"Some dates are outside {CALENDAR_START.year} to {CALENDAR_END.year}. The festival calendar only covers "
            "those years, so festival score is 0 outside them.")

    report.update(rows_ok=int(len(df)), date_min=str(df["date"].min().date()), date_max=str(df["date"].max().date()),
                  items=sorted(df["item"].unique().tolist()))
    return df[COLUMNS], report


def short_history_warnings(df):
    """Warnings for items whose total history is too short to forecast."""
    counts = df.groupby("item")["date"].count()
    return [f"{item}: only {n} days of data (needs {MIN_ROWS_PER_ITEM}). It is stored but not forecast yet."
            for item, n in counts.items() if n < MIN_ROWS_PER_ITEM]


def replace_with(df):
    with _lock:
        _save(df)


def append(df):
    """Add rows to the active dataset. Rows for a date and item that already exist overwrite the old value."""
    with _lock:
        current = load()
        merged = pd.concat([current, df], ignore_index=True)
        merged = merged.drop_duplicates(["date", "item"], keep="last")
        _save(merged)
        return len(merged)


def add_daily(date, entries, note=""):
    """
    Record one day's sales. `entries` is a list of {"item": str, "kg": float}.
    Existing values for the same date and item are overwritten. Every entry is also written to the sales log.
    """
    if not entries:
        raise DataError("Add at least one item.")
    day = pd.Timestamp(date).normalize()
    if day > pd.Timestamp.today().normalize() + pd.Timedelta(days=1):
        raise DataError("The date is in the future. Daily sales can only be entered for today or earlier.")
    raw = pd.DataFrame({"date": [day] * len(entries), "item": [e["item"] for e in entries],
                        "demand_units_kg": [e["kg"] for e in entries]})
    current = load()
    clean, report = normalize(raw, current)
    if clean.empty:
        raise DataError("None of the entries could be used.")
    total = append(clean)

    SALES_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = clean[["date", "item", "demand_units_kg"]].copy()
    log["date"] = log["date"].dt.strftime("%Y-%m-%d")
    log["entered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log["note"] = note
    log.to_csv(SALES_LOG, mode="a", header=not SALES_LOG.exists(), index=False)
    return {"saved": int(len(clean)), "total_rows": int(total), "report": report}


def sales_log(limit=50):
    if not SALES_LOG.exists():
        return pd.DataFrame(columns=["date", "item", "demand_units_kg", "entered_at", "note"])
    log = pd.read_csv(SALES_LOG, keep_default_na=False)
    return log.tail(limit).iloc[::-1].reset_index(drop=True)


def reset():
    with _lock:
        shutil.copyfile(RAW_CSV, ACTIVE_CSV)
        if SALES_LOG.exists():
            SALES_LOG.unlink()


def summary(df=None):
    df = load() if df is None else df
    per_item = df.groupby(["item", "category"]).agg(rows=("date", "count"), first_date=("date", "min"),
                                                    last_date=("date", "max"), avg_kg=("demand_units_kg", "mean")).reset_index()
    return {
        "rows": int(len(df)), "items": int(df["item"].nunique()),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "is_original": bool(filecmp.cmp(ACTIVE_CSV, RAW_CSV, shallow=False)),
        "per_item": [
            {"item": r.item, "category": r.category, "rows": int(r.rows), "first_date": str(r.first_date.date()),
             "last_date": str(r.last_date.date()), "avg_kg": round(float(r.avg_kg), 1),
             "forecastable": bool(r.rows >= MIN_ROWS_PER_ITEM)} for r in per_item.itertuples()],
    }


def template_csv():
    return ("date,item,demand_units_kg\n2025-01-01,Rose,52.5\n2025-01-01,Marigold,41.0\n"
            "2025-01-02,Rose,49.8\n2025-01-02,Marigold,38.2\n")

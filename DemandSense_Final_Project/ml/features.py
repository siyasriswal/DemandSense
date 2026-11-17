"""Turns a table of (date, item, demand) into the numbers the model learns from."""
import numpy as np
import pandas as pd

FEATURES = [
    "is_wedding_season",         # 1 in wedding months
    "festival_proximity_score",  # 0 to 1, highest on the festival day
    "doy_sin", "doy_cos",        # day of year as a circle, so 31 Dec is next to 1 Jan
    "dow_num",                   # 0 = Monday ... 6 = Sunday
    "item_encoded",              # which item this row is about
]


def add_features(df, encoder):
    out = df.copy()
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    out["dow_num"] = out["date"].dt.dayofweek
    out["item_encoded"] = encoder.transform(out["item"])
    return out


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def mape(actual, predicted):
    """Percentage error. Days with demand of 1 kg or less are skipped so near-zero days do not blow it up."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    keep = a > 1
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs(a[keep] - p[keep]) / a[keep]) * 100)

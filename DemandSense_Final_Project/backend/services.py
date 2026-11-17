"""Shared state and helper logic for the API: loads the model, reads data, reloads after retraining."""
import io
import threading

import pandas as pd

from config import DEFAULT_SHELF_LIFE, REPORT_DIR, SHELF_LIFE_DAYS
from data_pipeline import store
from ml import train as trainer
from ml.predict import Forecaster, ModelNotReady

_lock = threading.RLock()
_forecaster = None


def forecaster():
    global _forecaster
    with _lock:
        if _forecaster is None:
            try:
                _forecaster = Forecaster()
            except ModelNotReady:
                trainer.train()          # first run: train on whatever data is active
                _forecaster = Forecaster()
        return _forecaster


def retrain():
    global _forecaster
    with _lock:
        metrics = trainer.train()
        _forecaster = Forecaster()
        return metrics


def shelf_life(item):
    return SHELF_LIFE_DAYS.get(item, DEFAULT_SHELF_LIFE)


def read_upload(filename, content):
    """Read an uploaded .csv or .xlsx file into a DataFrame."""
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding_errors="replace")
    except Exception as exc:
        raise store.DataError(f"The file could not be read as a table ({exc}).")


def seasonality(df):
    df = df.copy()
    monthly = df.groupby(["item", df["date"].dt.month])["demand_units_kg"].mean().unstack().round(1)
    lifts = []
    for item, g in df.groupby("item"):
        normal = g[(g["is_wedding_season"] == 0) & (g["festival_proximity_score"] == 0)]["demand_units_kg"].mean()
        wed = g[(g["is_wedding_season"] == 1) & (g["festival_proximity_score"] == 0)]["demand_units_kg"].mean()
        fest = g[(g["festival_proximity_score"] >= 0.5)]["demand_units_kg"].mean()
        lifts.append({"item": item,
                      "wedding_lift": round(float(wed / normal), 2) if normal and pd.notna(wed) else None,
                      "festival_lift": round(float(fest / normal), 2) if normal and pd.notna(fest) else None})
    return {"months": list(range(1, 13)),
            "monthly": [{"item": i, "values": [None if pd.isna(v) else float(v) for v in monthly.loc[i].reindex(range(1, 13))]}
                        for i in monthly.index],
            "lifts": lifts}


def phase2_benchmark():
    path = REPORT_DIR / "phase2_benchmark.csv"
    return pd.read_csv(path).to_dict("records") if path.exists() else []

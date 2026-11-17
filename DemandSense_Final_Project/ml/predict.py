"""Loads the saved model and produces forecasts for any item and any future dates."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from data_pipeline.festival_calendar import festival_name_for, festival_score, wedding_season
from ml.features import FEATURES, add_features
from ml.train import META_PATH, MODEL_PATH


class ModelNotReady(RuntimeError):
    pass


class UnknownItem(KeyError):
    pass


class Forecaster:
    def __init__(self):
        if not (MODEL_PATH.exists() and META_PATH.exists()):
            raise ModelNotReady("No trained model found. Run `python -m ml.train` or retrain from the dashboard.")
        self.meta = json.loads(META_PATH.read_text())
        self.model = xgb.XGBRegressor()
        self.model.load_model(MODEL_PATH)
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.array(self.meta["classes"])

    @property
    def items(self):
        return list(self.meta["classes"])

    def residuals(self, item):
        return self.meta["residuals"].get(item, self.meta["residuals"]["__all__"])

    def forecast(self, item, start, days):
        if item not in self.meta["classes"]:
            raise UnknownItem(item)
        dates = pd.date_range(pd.Timestamp(start), periods=int(days), freq="D")
        frame = pd.DataFrame({"date": dates, "item": item})
        frame["is_wedding_season"] = wedding_season(dates)
        frame["festival_proximity_score"] = festival_score(dates)
        feats = add_features(frame, self.encoder)
        pred = np.clip(self.model.predict(feats[FEATURES]), 0, None)
        r = self.residuals(item)
        out = pd.DataFrame({
            "date": dates, "forecast": pred,
            "lower": np.clip(pred + r["q05"], 0, None), "upper": pred + r["q95"],
            "is_wedding_season": frame["is_wedding_season"], "festival_score": frame["festival_proximity_score"],
            "festival": [festival_name_for(d) if s > 0 else None for d, s in zip(dates, frame["festival_proximity_score"])],
        })
        out[["forecast", "lower", "upper"]] = out[["forecast", "lower", "upper"]].round(1)
        return out

"""
Model training.

1. Hold out the last 90 days and train on everything before them. Predict the
   held-out days and compute MAE and MAPE, then compare against two simple
   baselines (item average, same day last year).
2. Re-train on ALL the data. This final model is the one the dashboard uses.

Run:  python -m ml.train
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from config import MIN_ROWS_PER_ITEM, MODEL_DIR, REPORT_DIR, TEST_DAYS, XGB_PARAMS
from data_pipeline import store
from ml.features import FEATURES, add_features, mae, mape

MODEL_PATH = MODEL_DIR / "xgb_model.json"
META_PATH = MODEL_DIR / "model_meta.json"
BACKTEST_PATH = REPORT_DIR / "backtest.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"


class TrainingError(RuntimeError):
    pass


def _fit(frame):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(frame[FEATURES], frame["demand_units_kg"])
    return model


def train(df=None, test_days=TEST_DAYS, save=True):
    df = store.load() if df is None else df.copy()
    counts = df.groupby("item")["date"].count()
    eligible = sorted(counts[counts >= MIN_ROWS_PER_ITEM].index)
    skipped = sorted(set(counts.index) - set(eligible))
    if not eligible:
        raise TrainingError(f"No item has {MIN_ROWS_PER_ITEM} or more days of data, so nothing can be trained.")
    df = df[df["item"].isin(eligible)].sort_values("date").reset_index(drop=True)

    encoder = LabelEncoder().fit(eligible)
    data = add_features(df, encoder)

    # last-year baseline: demand of the same item 364 days earlier (same weekday)
    last_year = data.set_index(["item", "date"])["demand_units_kg"]
    lookup = pd.MultiIndex.from_arrays([data["item"], data["date"] - pd.Timedelta(days=364)])
    data["last_year"] = last_year.reindex(lookup).values

    span = (data["date"].max() - data["date"].min()).days + 1
    test_days = int(min(test_days, max(14, span // 5)))
    split_date = data["date"].max() - pd.Timedelta(days=test_days)
    train_df, test_df = data[data["date"] <= split_date], data[data["date"] > split_date].copy()
    if len(train_df) < 200 or test_df.empty:
        raise TrainingError("Not enough history to hold out a test period. Upload at least a few months of data.")

    holdout_model = _fit(train_df)
    test_df["predicted"] = np.clip(holdout_model.predict(test_df[FEATURES]), 0, None)
    item_mean = train_df.groupby("item")["demand_units_kg"].mean()
    test_df["naive"] = test_df["item"].map(item_mean)
    test_df["last_year_pred"] = test_df["last_year"].fillna(test_df["naive"])
    ly_coverage = float(test_df["last_year"].notna().mean())

    y = test_df["demand_units_kg"]
    ladder = [
        {"model": "Naive: item average", "mae": mae(y, test_df["naive"]), "mape": mape(y, test_df["naive"])},
        {"model": "Same day last year", "mae": mae(y, test_df["last_year_pred"]), "mape": mape(y, test_df["last_year_pred"]),
         "note": f"last-year value available for {ly_coverage:.0%} of test days"},
        {"model": "XGBoost (wedding, festival, calendar features)", "mae": mae(y, test_df["predicted"]),
         "mape": mape(y, test_df["predicted"])},
    ]
    base = ladder[0]["mae"]
    for row in ladder:
        row["mae"], row["mape"] = round(row["mae"], 2), round(row["mape"], 2)
        row["improvement_pct"] = round((1 - row["mae"] / base) * 100, 1) if base else 0.0

    per_item, residuals = [], {}
    all_resid = (y - test_df["predicted"]).values
    for item, g in test_df.groupby("item"):
        resid = (g["demand_units_kg"] - g["predicted"]).values
        use = resid if len(resid) >= 10 else all_resid
        residuals[item] = {"q05": float(np.quantile(use, 0.05)), "q95": float(np.quantile(use, 0.95)),
                           "std": float(np.std(use))}
        per_item.append({"item": item, "category": g["item"].map(df.drop_duplicates("item").set_index("item")["category"]).iloc[0],
                         "naive_mae": round(mae(g["demand_units_kg"], g["naive"]), 2),
                         "model_mae": round(mae(g["demand_units_kg"], g["predicted"]), 2),
                         "model_mape": round(mape(g["demand_units_kg"], g["predicted"]), 2)})
    for p in per_item:
        p["improvement_pct"] = round((1 - p["model_mae"] / p["naive_mae"]) * 100, 1) if p["naive_mae"] else 0.0
    residuals["__all__"] = {"q05": float(np.quantile(all_resid, 0.05)), "q95": float(np.quantile(all_resid, 0.95)),
                            "std": float(np.std(all_resid))}

    final_model = _fit(data)
    importance = dict(zip(FEATURES, [float(v) for v in final_model.feature_importances_]))

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes": encoder.classes_.tolist(), "features": FEATURES,
        "categories": df.drop_duplicates("item").set_index("item")["category"].to_dict(),
        "residuals": residuals, "feature_importance": importance,
        "n_rows": int(len(data)), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "data_start": str(data["date"].min().date()), "data_end": str(data["date"].max().date()),
        "test_start": str(test_df["date"].min().date()), "test_end": str(test_df["date"].max().date()),
        "test_days": test_days, "skipped_items": skipped,
    }
    metrics = {"trained_at": meta["trained_at"], "test_days": test_days, "test_window": [meta["test_start"], meta["test_end"]],
               "n_train": meta["n_train"], "n_test": meta["n_test"], "ladder": ladder, "per_item": per_item,
               "xgb_mae": ladder[-1]["mae"], "xgb_mape": ladder[-1]["mape"], "baseline_mae": ladder[0]["mae"],
               "skipped_items": skipped}

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        final_model.save_model(MODEL_PATH)
        META_PATH.write_text(json.dumps(meta, indent=2))
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        back = test_df[["date", "item", "demand_units_kg", "predicted", "naive"]].rename(columns={"demand_units_kg": "actual"})
        back["date"] = back["date"].dt.strftime("%Y-%m-%d")
        back.round(2).to_csv(BACKTEST_PATH, index=False)
    return metrics


if __name__ == "__main__":
    m = train()
    print(f"Trained on {m['n_train']} rows, tested on the last {m['test_days']} days ({m['n_test']} rows)")
    for r in m["ladder"]:
        print(f"  {r['model']:<48} MAE {r['mae']:>6.2f} kg   MAPE {r['mape']:>6.2f}%")

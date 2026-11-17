"""
Synthetic dataset generator (12 items, 1 Jan 2022 to 31 Dec 2024).

Real item-level sales data was not available for this project, so demand is
generated from simple, explainable rules:
    demand = base x wedding boost x festival boost x weekend effect x yearly growth + noise
Every item has its own boosts, e.g. Rose and Jasmine rise in wedding season,
Marigold rises around festivals, and Mango sells almost only from April to June.
The random seed is fixed, so running this always gives the same file.

Run:  python -m data_pipeline.generate_dataset
"""
import numpy as np
import pandas as pd

from config import RAW_CSV, WEDDING_MONTHS
from data_pipeline.festival_calendar import festival_score

START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2024-12-31")

# base = average daily demand in kg, noise = random variation,
# season_only = months in which the item is really sold (None = all year)
ITEMS = {
    "Flowers": {
        "Rose":     {"base": 45, "noise": 6,  "wedding_boost": 3.2, "festival_boost": 1.6, "season_only": None},
        "Marigold": {"base": 40, "noise": 6,  "wedding_boost": 1.8, "festival_boost": 3.6, "season_only": None},
        "Jasmine":  {"base": 25, "noise": 4,  "wedding_boost": 3.0, "festival_boost": 1.8, "season_only": None},
        "Lotus":    {"base": 10, "noise": 2,  "wedding_boost": 1.3, "festival_boost": 2.4, "season_only": None},
    },
    "Vegetables": {
        "Onion":          {"base": 120, "noise": 14, "wedding_boost": 2.0, "festival_boost": 1.3, "season_only": None},
        "Potato":         {"base": 110, "noise": 13, "wedding_boost": 1.9, "festival_boost": 1.3, "season_only": None},
        "Tomato":         {"base": 90,  "noise": 12, "wedding_boost": 1.8, "festival_boost": 1.4, "season_only": None},
        "Paneer_Veg_Mix": {"base": 60,  "noise": 9,  "wedding_boost": 2.6, "festival_boost": 1.2, "season_only": None},
    },
    "Fruits": {
        "Banana":      {"base": 55, "noise": 7, "wedding_boost": 1.5, "festival_boost": 2.2, "season_only": None},
        "Apple":       {"base": 40, "noise": 6, "wedding_boost": 1.8, "festival_boost": 1.6, "season_only": None},
        "Mango":       {"base": 50, "noise": 8, "wedding_boost": 1.4, "festival_boost": 1.2, "season_only": {4, 5, 6}},
        "Pomegranate": {"base": 30, "noise": 5, "wedding_boost": 1.5, "festival_boost": 2.0, "season_only": None},
    },
}


def generate(seed=42):
    np.random.seed(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    fest = dict(zip(dates, festival_score(dates)))
    rows = []
    for category, items in ITEMS.items():
        for item, cfg in items.items():
            for d in dates:
                in_wedding = d.month in WEDDING_MONTHS
                fw = float(fest[d])
                if cfg["season_only"] and d.month not in cfg["season_only"]:
                    # out of season: a trickle of demand only
                    demand = max(0, np.random.normal(cfg["base"] * 0.05, cfg["noise"] * 0.3))
                    demand = round(demand, 1)
                else:
                    wedding_mult = cfg["wedding_boost"] if in_wedding else 1.0
                    if in_wedding and d.weekday() in (5, 6):
                        wedding_mult *= 1.15
                    festival_mult = 1 + fw * (cfg["festival_boost"] - 1)
                    weekly_mult = 1.1 if d.weekday() == 6 else 1.0
                    trend_mult = 1 + 0.04 * (d.year - START_DATE.year)
                    noise = np.random.normal(0, cfg["noise"])
                    demand = max(0, round(cfg["base"] * wedding_mult * festival_mult * weekly_mult * trend_mult + noise, 1))
                rows.append({
                    "date": d.strftime("%Y-%m-%d"), "category": category, "item": item,
                    "demand_units_kg": demand, "is_wedding_season": int(in_wedding),
                    "festival_proximity_score": round(fw, 2), "day_of_week": d.strftime("%A"),
                    "month": d.month, "year": d.year,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print("Dataset written:", RAW_CSV, df.shape)

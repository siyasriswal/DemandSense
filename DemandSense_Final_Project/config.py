"""Project-wide settings. Every module imports paths and constants from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Tests point this at a temporary folder so they never touch real project data.
HOME = Path(os.environ.get("DEMANDSENSE_HOME", ROOT))

DATA_DIR = HOME / "data"
RAW_CSV = DATA_DIR / "raw" / "demand_dataset_v2_items.csv"   # original dataset, never modified
ACTIVE_CSV = DATA_DIR / "active_dataset.csv"                  # the data the models are trained on
SALES_LOG = DATA_DIR / "sales_log.csv"                        # every daily entry typed into the dashboard
MODEL_DIR = HOME / "models"
REPORT_DIR = HOME / "reports"

API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = os.environ.get("DEMANDSENSE_API", f"http://{API_HOST}:{API_PORT}")

TEST_DAYS = 90            # last 90 days are held out to measure accuracy
MIN_ROWS_PER_ITEM = 120   # items with less history than this are not forecast
FESTIVAL_WINDOW = 4       # festival score falls to 0 four days away from a festival

# Assumed shelf life in days, used by the stock planner (editable in the dashboard).
SHELF_LIFE_DAYS = {
    "Rose": 3, "Marigold": 3, "Jasmine": 2, "Lotus": 2,
    "Onion": 14, "Potato": 21, "Tomato": 6, "Paneer_Veg_Mix": 4,
    "Banana": 5, "Apple": 14, "Mango": 5, "Pomegranate": 10,
}
DEFAULT_SHELF_LIFE = 5

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)

WEDDING_MONTHS = {11, 12, 1, 2, 4, 5, 6}

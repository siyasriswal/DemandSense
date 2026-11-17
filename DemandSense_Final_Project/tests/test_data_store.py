import pandas as pd
import pytest

from data_pipeline import generate_dataset, store


def test_generator_matches_shipped_dataset():
    fresh = generate_dataset.generate()
    shipped = pd.read_csv(store.RAW_CSV)
    assert fresh.shape == shipped.shape == (13152, 9)
    assert (fresh["demand_units_kg"].values == shipped["demand_units_kg"].values).all()


def test_normalize_accepts_other_column_names_and_item_spellings():
    store.ensure_active()
    existing = store.load()
    raw = pd.DataFrame({"Date": ["2025-01-01", "2025-01-01", "not a date"],
                        "Product": ["rose", "paneer veg mix", "Rose"], "Daily Sales": [50, 40, 3]})
    clean, report = store.normalize(raw, existing)
    assert sorted(clean["item"]) == ["Paneer_Veg_Mix", "Rose"]
    assert report["dropped"] == {"bad_date": 1}
    assert clean["is_wedding_season"].tolist() == [1, 1]      # January


def test_normalize_rejects_missing_columns():
    with pytest.raises(store.DataError, match="Could not find"):
        store.normalize(pd.DataFrame({"foo": [1]}))


def test_normalize_drops_negative_and_duplicate_rows():
    raw = pd.DataFrame({"date": ["2025-02-01", "2025-02-01", "2025-02-02"], "item": ["Rose", "Rose", "Rose"],
                        "kg": [10, 12, -5]})
    clean, report = store.normalize(raw)
    assert len(clean) == 1 and clean["demand_units_kg"].iloc[0] == 12
    assert report["dropped"]["negative_quantity"] == 1


def test_daily_entry_upserts_and_logs():
    store.reset()
    before = len(store.load())
    store.add_daily("2025-01-01", [{"item": "Rose", "kg": 55.5}])
    store.add_daily("2025-01-01", [{"item": "Rose", "kg": 60.0}])       # same day again: overwrite
    df = store.load()
    assert len(df) == before + 1
    assert df[(df["item"] == "Rose") & (df["date"] == "2025-01-01")]["demand_units_kg"].iloc[0] == 60.0
    assert len(store.sales_log()) == 2
    store.reset()
    assert len(store.load()) == before


def test_future_dates_are_refused():
    with pytest.raises(store.DataError, match="future"):
        store.add_daily("2999-01-01", [{"item": "Rose", "kg": 1}])

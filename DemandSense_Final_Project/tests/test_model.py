import pandas as pd
import pytest

from data_pipeline import store
from ml import train
from ml.features import mae, mape
from ml.predict import Forecaster, UnknownItem


def test_metrics_by_hand():
    assert mae([10, 20, 30], [12, 18, 33]) == pytest.approx((2 + 2 + 3) / 3)
    assert mape([100, 0.5], [90, 5]) == pytest.approx(10.0)         # the 0.5 kg day is ignored


@pytest.fixture(scope="module")
def trained():
    store.reset()
    return train.train()


def test_model_beats_the_simple_average(trained):
    assert trained["xgb_mae"] < trained["baseline_mae"] * 0.4
    assert trained["n_test"] == 1080 and trained["test_days"] == 90


def test_forecast_shape_and_ordering(trained):
    f = Forecaster().forecast("Marigold", "2025-10-25", 14)
    assert len(f) == 14
    assert (f["lower"] <= f["forecast"]).all() and (f["forecast"] <= f["upper"]).all()
    assert (f["forecast"] >= 0).all()


def test_marigold_peaks_at_diwali(trained):
    f = Forecaster().forecast("Marigold", "2025-10-10", 21).set_index("date")["forecast"]
    assert f["2025-10-20"] > 1.5 * f["2025-10-10"]                    # Diwali 2025 vs a normal day


def test_mango_is_low_outside_season(trained):
    fc = Forecaster()
    assert fc.forecast("Mango", "2025-05-01", 1)["forecast"].iloc[0] > 5 * fc.forecast("Mango", "2025-09-01", 1)["forecast"].iloc[0]


def test_unknown_item(trained):
    with pytest.raises(UnknownItem):
        Forecaster().forecast("Dragonfruit", "2025-01-01", 3)


def test_training_needs_enough_data():
    tiny = store.load().head(50)
    with pytest.raises(train.TrainingError):
        train.train(tiny, save=False)

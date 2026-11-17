import io

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from data_pipeline import store

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_data():
    store.reset()
    yield
    store.reset()


def test_health_and_items():
    assert client.get("/health").json()["status"] == "ok"
    items = client.get("/items").json()
    assert len(items) == 12 and all(i["forecastable"] for i in items)


def test_forecast_endpoint():
    r = client.get("/forecast", params={"item": "Rose", "days": 10}).json()
    assert r["start"] == "2025-01-01" and len(r["forecast"]) == 10
    assert client.get("/forecast", params={"item": "Nope"}).status_code == 404
    assert client.get("/forecast", params={"item": "Rose", "days": 999}).status_code == 422


def test_summary_metrics_and_insights():
    assert len(client.get("/forecast/summary", params={"days": 7}).json()["items"]) == 12
    m = client.get("/metrics").json()
    assert m["live"]["xgb_mae"] < 10 and len(m["phase2_benchmark"]) == 7
    assert len(client.get("/insights/seasonality").json()["monthly"]) == 12
    assert client.get("/backtest", params={"item": "Onion"}).json()


def test_planner_flags_wastage_when_overstocked():
    ok = client.post("/planner", json={"item": "Rose", "stock_kg": 100, "days": 7}).json()
    assert ok["recommended_order_kg"] > 0 and ok["wastage_risk"] == "Low"
    over = client.post("/planner", json={"item": "Rose", "stock_kg": 5000, "days": 7}).json()
    assert over["wastage_risk"] == "High" and over["recommended_order_kg"] == 0
    assert client.post("/planner", json={"item": "Rose", "stock_kg": -1}).status_code == 422


def test_daily_sales_entry_changes_the_dataset():
    before = client.get("/dataset/summary").json()["rows"]
    r = client.post("/sales/daily", json={"date": "2025-01-01", "entries": [{"item": "Rose", "kg": 58.0},
                                                                         {"item": "Onion", "kg": 130.0}], "retrain": False})
    assert r.status_code == 200 and r.json()["saved"] == 2
    assert client.get("/dataset/summary").json()["rows"] == before + 2
    assert len(client.get("/sales/log").json()) == 2
    # the forecast now starts the day after the new last date
    assert client.get("/forecast", params={"item": "Rose", "days": 3}).json()["start"] == "2025-01-02"


def test_csv_upload_append_and_retrain():
    csv = "date,product,quantity\n" + "\n".join(f"2025-01-{d:02d},Rose,{50 + d}" for d in range(1, 11)) + "\n2025-01-03,Dragonfruit,12\n"
    r = client.post("/dataset/upload", files={"file": ("sales.csv", io.BytesIO(csv.encode()), "text/csv")},
                    data={"mode": "append", "retrain": "true"})
    body = r.json()
    assert r.status_code == 200 and body["report"]["rows_ok"] == 11
    assert body["report"]["new_items"] == ["Dragonfruit"]
    assert any("Dragonfruit" in w for w in body["report"]["warnings"])        # too little history to forecast
    assert body["training"]["xgb_mae"] < 10
    assert "Dragonfruit" not in [i["item"] for i in client.get("/items").json() if i["trained"]]


def test_bad_uploads_give_clear_errors():
    r = client.post("/dataset/upload", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")}, data={"mode": "append"})
    assert r.status_code == 400 and "Could not find" in r.json()["detail"]
    r = client.post("/dataset/upload", files={"file": ("x.csv", io.BytesIO(b""), "text/csv")}, data={"mode": "append"})
    assert r.status_code == 400


def test_replace_with_too_little_data_reports_training_error():
    csv = "date,item,demand_units_kg\n2025-01-01,Rose,50\n2025-01-02,Rose,52\n"
    r = client.post("/dataset/upload", files={"file": ("s.csv", io.BytesIO(csv.encode()), "text/csv")},
                    data={"mode": "replace", "retrain": "true"})
    assert r.status_code == 200 and "training_error" in r.json()
    client.post("/dataset/reset")
    assert client.get("/dataset/summary").json()["is_original"]

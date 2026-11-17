"""
DemandSense backend (REST API).

Start:  uvicorn backend.main:app --reload --port 8000
Docs:   http://127.0.0.1:8000/docs   (every endpoint can be tried from this page)
"""
import json
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from backend import services
from backend.planner import plan
from backend.schemas import DailySales, PlannerRequest
from data_pipeline import store
from data_pipeline.festival_calendar import festival_name_for, upcoming_festivals
from ml import train as trainer
from ml.predict import ModelNotReady, UnknownItem

app = FastAPI(title="DemandSense API", version="1.0",
              description="Seasonal demand forecasting for flowers, vegetables and fruits.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(store.DataError)
async def data_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(trainer.TrainingError)
async def training_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _fc():
    try:
        return services.forecaster()
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc))


def _item_or_404(fc, item):
    if item not in fc.items:
        raise HTTPException(404, f"'{item}' is not a forecastable item. Known items: {', '.join(fc.items)}")


# ---------------------------------------------------------------- basic info
@app.get("/health")
def health():
    ready = trainer.MODEL_PATH.exists()
    return {"status": "ok", "model_ready": ready}


@app.get("/items")
def items():
    summary = store.summary()
    fc = _fc()
    rows = []
    for r in summary["per_item"]:
        r["trained"] = r["item"] in fc.items
        r["shelf_life_days"] = services.shelf_life(r["item"])
        rows.append(r)
    return rows


@app.get("/dataset/summary")
def dataset_summary():
    s = store.summary()
    s["model"] = {k: _fc().meta[k] for k in ("trained_at", "data_start", "data_end", "n_rows", "test_start", "test_end",
                                              "skipped_items")}
    return s


@app.get("/dataset/template", response_class=PlainTextResponse)
def dataset_template():
    return PlainTextResponse(store.template_csv(), media_type="text/csv")


# ---------------------------------------------------------------- history and forecast
@app.get("/history")
def history(item: str, days: int = Query(180, ge=7, le=2000)):
    df = store.load()
    df = df[df["item"] == item]
    if df.empty:
        raise HTTPException(404, f"No data for '{item}'.")
    df = df[df["date"] > df["date"].max() - pd.Timedelta(days=days)]
    return [{"date": d.strftime("%Y-%m-%d"), "demand_kg": float(v), "wedding": int(w), "festival_score": float(f)}
            for d, v, w, f in zip(df["date"], df["demand_units_kg"], df["is_wedding_season"], df["festival_proximity_score"])]


@app.get("/forecast")
def forecast(item: str, days: int = Query(30, ge=1, le=180), start: Optional[str] = None):
    fc = _fc()
    _item_or_404(fc, item)
    first = pd.Timestamp(start) if start else store.load()["date"].max() + pd.Timedelta(days=1)
    out = fc.forecast(item, first, days)
    return {"item": item, "start": str(first.date()), "days": days, "model_trained_at": fc.meta["trained_at"],
            "total_kg": round(float(out["forecast"].sum()), 1),
            "forecast": [{"date": r.date.strftime("%Y-%m-%d"), "forecast": r.forecast, "lower": r.lower, "upper": r.upper,
                          "wedding": int(r.is_wedding_season), "festival_score": float(r.festival_score),
                          "festival": r.festival} for r in out.itertuples()]}


@app.get("/forecast/summary")
def forecast_summary(days: int = Query(7, ge=1, le=60)):
    fc = _fc()
    df = store.load()
    first = df["date"].max() + pd.Timedelta(days=1)
    recent_start = df["date"].max() - pd.Timedelta(days=days - 1)
    rows = []
    for item in fc.items:
        out = fc.forecast(item, first, days)
        recent = df[(df["item"] == item) & (df["date"] >= recent_start)]["demand_units_kg"]
        recent_total = float(recent.sum()) if len(recent) else None
        peak = out.loc[out["forecast"].idxmax()]
        total = float(out["forecast"].sum())
        rows.append({"item": item, "category": fc.meta["categories"].get(item, "Other"), "total_kg": round(total, 1),
                     "avg_kg_per_day": round(total / days, 1), "peak_day": peak["date"].strftime("%Y-%m-%d"),
                     "peak_kg": float(peak["forecast"]),
                     "recent_total_kg": None if recent_total is None else round(recent_total, 1),
                     "change_pct": None if not recent_total else round((total / recent_total - 1) * 100, 1)})
    return {"start": str(first.date()), "days": days, "items": sorted(rows, key=lambda r: -r["total_kg"])}


@app.get("/calendar/upcoming")
def calendar_upcoming(n: int = 4, after: Optional[str] = None):
    start = pd.Timestamp(after) if after else store.load()["date"].max() + pd.Timedelta(days=1)
    return [{"date": str(d.date()), "festival": name, "days_away": int((d - start).days)} for d, name in upcoming_festivals(start, n)]


# ---------------------------------------------------------------- model performance
@app.get("/metrics")
def metrics():
    path = trainer.METRICS_PATH
    if not path.exists():
        raise HTTPException(503, "No metrics yet. Train the model first.")
    live = json.loads(path.read_text())
    return {"live": live, "phase2_benchmark": services.phase2_benchmark(),
            "is_original_data": store.summary()["is_original"]}


@app.get("/backtest")
def backtest(item: str):
    if not trainer.BACKTEST_PATH.exists():
        raise HTTPException(503, "No backtest yet. Train the model first.")
    df = pd.read_csv(trainer.BACKTEST_PATH)
    df = df[df["item"] == item]
    if df.empty:
        raise HTTPException(404, f"No backtest rows for '{item}'.")
    return df.to_dict("records")


@app.get("/model/info")
def model_info():
    meta = _fc().meta
    return {k: meta[k] for k in ("trained_at", "features", "feature_importance", "n_rows", "n_train", "n_test",
                                 "data_start", "data_end", "test_start", "test_end", "test_days", "skipped_items")} | {
        "items": meta["classes"]}


@app.post("/model/retrain")
def model_retrain():
    return services.retrain()


@app.get("/insights/seasonality")
def insights():
    return services.seasonality(store.load())


# ---------------------------------------------------------------- changing the data
@app.post("/dataset/upload")
async def dataset_upload(file: UploadFile = File(...), mode: str = Form("append"), retrain: bool = Form(True)):
    if mode not in ("append", "replace"):
        raise HTTPException(400, "mode must be 'append' or 'replace'.")
    raw = services.read_upload(file.filename, await file.read())
    existing = None if mode == "replace" else store.load()
    clean, report = store.normalize(raw, existing)
    if mode == "replace":
        store.replace_with(clean)
    else:
        store.append(clean)
    report["mode"] = mode
    report["warnings"] += store.short_history_warnings(store.load())
    result = {"report": report, "summary": store.summary(), "training": None}
    if retrain:
        try:
            result["training"] = services.retrain()
        except trainer.TrainingError as exc:
            result["training_error"] = str(exc)
    return result


@app.post("/sales/daily")
def sales_daily(body: DailySales):
    result = store.add_daily(body.date, [e.model_dump() for e in body.entries], body.note or "")
    result["training"] = services.retrain() if body.retrain else None
    return result


@app.get("/sales/log")
def sales_log(limit: int = Query(50, ge=1, le=500)):
    return store.sales_log(limit).to_dict("records")


@app.post("/dataset/reset")
def dataset_reset():
    store.reset()
    return {"summary": store.summary(), "training": services.retrain()}


# ---------------------------------------------------------------- stock planner
@app.post("/planner")
def planner(body: PlannerRequest):
    fc = _fc()
    _item_or_404(fc, body.item)
    start = store.load()["date"].max() + pd.Timedelta(days=1)
    shelf = body.shelf_life_days or services.shelf_life(body.item)
    forecast_df = fc.forecast(body.item, start, max(body.days, shelf))
    result = plan(forecast_df, fc.residuals(body.item)["std"], body.stock_kg, shelf, body.service_level, days=body.days)
    result["item"] = body.item
    return result

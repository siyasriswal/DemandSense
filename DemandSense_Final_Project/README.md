# DemandSense

Seasonal demand forecasting for event-driven perishable goods (flowers, catering vegetables and fruits).
Team POKEMONS, MCA AI/DS Section A, PBL final phase.

DemandSense predicts daily demand in kg for 12 items, lets a shop owner add real sales every day or upload a
full history, and turns the forecast into a buying suggestion with a wastage warning.

## Architecture

```
CSV data  ->  Data layer  ->  ML layer (XGBoost)  ->  FastAPI backend  --JSON over HTTP-->  Streamlit dashboard
(uploads,     (validate,      (train, forecast,        (REST, port 8000)                     (port 8501)
 daily sales)  festival        intervals)
               features)
```

* Front end: Streamlit, 8 pages (Overview, Forecast, Stock planner, Daily sales entry, Upload data, Model performance, Seasonal insights, About).
* Back end: FastAPI REST API. The dashboard never touches the model or the files directly, it only calls the API.
* No third party or external APIs are used. The API is our own.

## Run it in VS Code

Needs Python 3.10 or newer.

```bash
# 1. open this folder in VS Code, then open the terminal (Ctrl + `)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install
pip install -r requirements.txt

# 3. start everything (API + dashboard, opens the browser)
python start.py
```

* Dashboard: http://localhost:8501
* API and its interactive docs: http://localhost:8000/docs

The first start trains the model (about 10 seconds). To run the two parts separately, use two terminals:

```bash
python -m uvicorn backend.main:app --port 8000
python -m streamlit run frontend/app.py --server.port 8501
```

Run the tests: `python -m pytest`  (25 tests, they use a temporary copy of the data).

## Using the dashboard

| Page | What it does |
|---|---|
| Overview | Next-N-day demand for all items, upcoming festivals, model accuracy |
| Forecast | One item: history, forecast, 90% interval, festival markers, table download |
| Stock planner | Stock in hand + shelf life + safety level gives order quantity and wastage risk |
| Daily sales entry | Add today's sales, several days in a grid (time series), or see the sales log. Optionally retrains |
| Upload data | Upload a full CSV or Excel history, append or replace, validation report, retrain |
| Model performance | Holdout MAE and MAPE, comparison with baselines and the Phase 2 models, backtest chart |
| Seasonal insights | Month by item heat map, wedding and festival lift |
| About | Architecture and technology |

Upload format: any CSV or XLSX with a date column, an item column and a quantity in kg. Common column names are
recognised (date, item or product, demand_units_kg or kg or quantity or sales). A template can be downloaded
from the Upload page. "Reset to original data" on that page brings back the project dataset.

## Folder structure

```
config.py                 settings (paths, model parameters, shelf lives)
start.py                  starts API and dashboard together
data_pipeline/            festival calendar, dataset generator, data store (validation, upload, daily entry)
ml/                       features, training and evaluation, forecaster
backend/                  FastAPI app (main.py), schemas, services, stock planner
frontend/                 Streamlit app, API client, shared UI, views/ (one file per page)
data/raw/                 original dataset (13,152 rows, 12 items, 2022 to 2024)
data/active_dataset.csv   working copy that uploads and daily entries change (created on first run)
models/  reports/         trained model and evaluation outputs (created by training)
notebooks/                Phase 2 Colab notebook (model comparison with Prophet)
tests/                    pytest suite
docs/                     team roles and viva preparation
```

## API endpoints

GET: /health, /items, /dataset/summary, /dataset/template, /history, /forecast, /forecast/summary,
/calendar/upcoming, /metrics, /backtest, /model/info, /insights/seasonality, /sales/log.
POST: /model/retrain, /dataset/upload, /sales/daily, /dataset/reset, /planner.

## Model in one paragraph

One XGBoost regressor for all items. Inputs are calendar features only: wedding season flag, festival proximity
score, day of year (sin and cos), day of week and the item. The last 90 days are held out for testing
(MAE about 8.1 kg, MAPE about 15%). Because there are no lag features, the model can forecast many days ahead.
Prediction intervals come from the 5th and 95th percentile of the holdout errors of each item.

## Honest notes

* The dataset is synthetic (generated with seasonal and festival patterns), built because real shop data was not available.
* Festival dates for 2022 to 2024 are those used by the generator, 2025 to 2027 are approximate.
* Shelf lives used by the stock planner are assumptions (see config.py).
* Prophet is used only in the Phase 2 notebook, not in the app.

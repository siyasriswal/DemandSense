---
title: "DemandSense: Team Roles and Viva Preparation"
subtitle: "Team POKEMONS, MCA AI/DS Section A, PBL Final Phase"
---

# 1. Project in one minute (everyone must know this)

**Title:** Seasonal Demand Forecasting for Event-Driven Perishable Goods (Flowers, Catering Vegetables and Fruits).

**Problem:** Flowers, vegetables and fruits sell in sharp waves around weddings and festivals. If a shop stocks too little it loses sales, and if it stocks too much the goods rot. 

**Solution:** DemandSense predicts the daily demand in kg for 12 items, lets the shop add real sales every day or upload a full history, and converts the forecast into a buying quantity with a wastage warning.

**How it is built:** a Streamlit dashboard (front end) talks over JSON/HTTP to a FastAPI REST backend, which uses a data layer (validation, festival features) and an ML layer (XGBoost).

**Headline result:** on the last 90 days of data (held out, never used for training) the model has an MAE of about 8.1 kg and a MAPE of about 15%. The simple item-average baseline has an MAE of 34 kg, so the model is about 76% better. "Same day last year" gives 16 kg.

| Member | Name | Student ID | Area |
|---|---|---|---|
| 1 | Siya Sriswal (Team Lead) | 2501650 | Backend API, integration, testing, documentation |
| 2 | Priyanshu Jimiwal | 2501614 | Machine learning: features, model, evaluation, forecasting |
| 3 | Ashish Negi | 2501533 | Data: dataset, festival calendar, validation, uploads |
| 4 | Priyansu Nautiyal | 2501615 | Front end: dashboard pages, charts, user experience |

**Rule for the viva:** every member should be able to explain the whole flow in Section 1, then go deep on their own part. Teachers often ask a member about another member's part to check teamwork.

**Demo order (about 8 minutes):** Member 4 opens the dashboard and shows Overview and Forecast. Member 3 uploads a CSV and adds daily sales. Member 2 shows Model performance. Member 1 shows the Stock planner, the API docs page (localhost:8000/docs) and runs the tests.

# 2. Technology map

| Technology | Why we used it | Owner |
|---|---|---|
| Python 3 | One language for data, ML, API and dashboard | All |
| pandas, NumPy | Reading, cleaning, reshaping data | 3 and 2 |
| scikit-learn | LabelEncoder for item names | 2 |
| XGBoost | The forecasting model | 2 |
| Prophet | Comparison model, only in the Phase 2 notebook | 2 |
| FastAPI, Uvicorn | REST API and its server | 1 |
| Pydantic | Request validation (schemas) | 1 |
| Streamlit | Dashboard | 4 |
| Plotly | Interactive charts | 4 |
| pytest | Automated tests | 1 |
| CSV / XLSX files | Data storage (simple, no database needed at this scale) | 3 |

# 3. Member 1: Siya Sriswal (Team Lead), Backend API, Integration and Testing

**Role:** builds the REST backend that sits between the dashboard and the model, owns the stock planner logic, makes sure all parts work together, writes tests and documentation.

**Files owned:** `backend/main.py`, `backend/schemas.py`, `backend/services.py`, `backend/planner.py`, `start.py`, `config.py`, `tests/test_api.py`, `README.md`.

**Technology:** FastAPI, Uvicorn, Pydantic, requests/HTTP, pytest, JSON.

**What she should be able to explain**

* Why a backend at all: the dashboard only shows data, the API owns the model and data. Any other client (mobile app, another website) could use the same API.
* The list of endpoints and what each does (see the list in README).
* How an upload or a daily entry flows: request, validation, saved to the data store, optional retrain, model reloaded.
* The stock planner formula (Q1 below).
* How `start.py` launches both servers, and how tests run on a temporary copy of the data.

**Demo duty:** show `localhost:8000/docs`, call `/forecast` from the docs page, run the Stock planner, run `python -m pytest`.

## Viva questions and answers

**Q1. How does the stock planner decide how much to order?**
It takes the forecast for the planning period, adds a safety stock and subtracts what is already in stock. Safety stock = z x residual standard deviation x square root of the cycle days. z comes from the service level (95% gives z of about 1.65). The residual standard deviation is the model's typical error for that item on the holdout data. The cycle is the smaller of the planning period and the shelf life, because goods that spoil cannot be bought for a long period at once. Order = forecast demand + safety stock - stock on hand (never below zero).

**Q2. How is the wastage risk calculated?**
Spoilage is the stock that will still be left after the shelf-life window, that is, stock minus the forecast demand during the shelf life. Risk is Low if that is under 10% of the stock, Medium under 30%, otherwise High.

**Q3. What type of API is this?**
A REST API built by us with FastAPI. It uses HTTP methods (GET to read, POST to change data) and sends JSON. It runs on localhost port 8000. We do not use any external or third party API.

**Q4. Why FastAPI and not Flask or Django?**
FastAPI gives automatic request validation through Pydantic, automatic interactive documentation at `/docs`, and is fast. Flask would also work but needs extra libraries for validation. Django is too heavy for an API of this size.

**Q5. What does Pydantic do here?**
It defines the shape of each request (for example item, stock_kg, days). If a value is wrong, such as negative stock, the API returns a 422 error automatically without our code running. Our test `test_planner_flags_wastage_when_overstocked` checks this.

**Q6. How do you handle errors?**
Our own exceptions are mapped to HTTP codes: bad user data (DataError) gives 400, an item that does not exist gives 404, a model that cannot train on too little data (TrainingError) gives 422. The dashboard shows the message in plain words instead of a crash.

**Q7. What do your tests cover and why do they use a temporary copy of data?**
25 tests cover the festival calendar, data validation and column aliases, daily entry overwrite, the generator reproducing the dataset, the metrics by hand, model quality, forecast ordering (lower <= forecast <= upper), Diwali peak for marigold, mango season, and every important endpoint including bad uploads. They run on a temporary copy so that testing never changes the real dataset.

**Q8. How do the frontend and backend communicate?**
The frontend uses the `requests` library in `frontend/api_client.py` to call the API and receives JSON. Read calls are cached for 2 minutes and the cache is cleared after any change to the data.

**Q9. What happens when the model is retrained while the app is running?**
`retrain()` in `services.py` trains using a lock, saves the new model files and reloads the in-memory forecaster, so the next request uses the new model without restarting the server.

**Q10. How would you deploy this for real use?**
Put the API and dashboard in Docker containers or on a cloud service, replace the CSV store with a database such as PostgreSQL, add login, and schedule retraining (for example weekly). This is future work.

**Q11. What are the limitations of the backend?**
CSV storage is fine for one shop but not for many users writing at once. There is no authentication. Both are listed as future work.

# 4. Member 2: Priyanshu Jimiwal, Machine Learning

**Role:** designs the features, trains and evaluates the model, produces forecasts and prediction intervals, and compared our model with other models in Phase 2.

**Files owned:** `ml/features.py`, `ml/train.py`, `ml/predict.py`, `notebooks/Phase2_DemandSense_Colab.ipynb`, `reports/*`, `tests/test_model.py`.

**Technology:** XGBoost, scikit-learn, pandas, NumPy, Prophet (notebook only).

**What he should be able to explain**

* The 6 features and why each exists.
* Why XGBoost, and why a time-based split instead of a random split.
* How MAE and MAPE are calculated, and the 90-day holdout.
* The benchmark ladder and what each baseline proves.
* How prediction intervals are made, and why the final model is refit on all data.

**Demo duty:** open Model performance, explain the table and the backtest chart, then Forecast for Marigold around Diwali.

## Viva questions and answers

**Q1. Which model do you use and why?**
XGBoost regression. Our data is tabular with strong calendar effects and non-linear interactions (for example a festival matters a lot for marigold but not for onion). Gradient boosted trees handle that well with little tuning, train in seconds, and beat the other models we tried (Section 7).

**Q2. What are the input features?**
`is_wedding_season` (1 in wedding months), `festival_proximity_score` (1.0 on a festival day, falling linearly to 0 over 4 days before and after), `doy_sin` and `doy_cos` (day of year as a circle so 31 Dec and 1 Jan are neighbours), `dow_num` (day of week), `item_encoded` (which item). The target is `demand_units_kg`.

**Q3. Why sin and cos of the day of year?**
Day 365 and day 1 are one day apart but numerically far apart. Putting the day on a circle with sin and cos removes that jump and lets trees learn smooth yearly seasonality.

**Q4. How is MAE calculated for the 90-day test?**
We keep the last 90 days (3 Oct to 31 Dec 2024) for testing and train on everything earlier. For each of the 1,080 test rows (90 days x 12 items) we take the absolute difference between actual and predicted kg, and average them. MAE = (1/n) x sum of |actual - predicted|. It is in kg, so 8.1 means we are wrong by about 8.1 kg per item per day on average.

**Q5. How is MAPE calculated, and why skip small values?**
MAPE = average of |actual - predicted| / actual x 100. If the actual is 0 or tiny the percentage explodes, so rows with actual of 1 kg or less are skipped. We report 14.98%.

**Q6. Why a time split and not a random split?**
Forecasting predicts the future. A random split would let the model see days after the test day, which is leakage and gives fake high accuracy. A time split mimics real use.

**Q7. Why is the app MAE 8.14 but the notebook says 8.13?**
Same model and parameters. The row order of the data differs slightly between the notebook and the app, and XGBoost's sampling (subsample 0.8) is order sensitive, so the result differs in the second decimal. It is not an error.

**Q8. What are the baselines and what do they prove?**
Naive item average: 34.06 kg MAE, the "do nothing smart" level. Same day last year (364 days earlier, to keep the weekday): 15.97 kg, a strong seasonal baseline. XGBoost: about 8.1 kg. So the model removes about 76% of the naive error and about half of the same-day-last-year error.

**Q9. How are the prediction intervals made?**
For each item we take the errors (residuals) on the holdout period. The 5th and 95th percentiles of those errors are added to the forecast to get a lower and upper bound, so about 90% of actual values should fall inside.

**Q10. Why do you refit on all data at the end?**
The holdout is only for measuring. For real forecasts we want the most recent 90 days included too, as they hold the latest festival pattern. The reported metric still comes from the honest holdout run.

**Q11. Your model has no lag features. Is that a weakness?**
It is a trade-off. In the Phase 2 notebook an XGBoost with lag features reached 8.08 kg, almost the same, but it can only predict the next day because it needs yesterday's real sales. Our calendar-only model can forecast 14 or 30 days ahead, which is what stock planning needs. We chose usefulness over a tiny gain in MAE.

**Q12. Why not LSTM or deep learning?**
We have about 13 thousand rows, the pattern is calendar driven, and a tree model already reaches the noise level of the data. Deep learning would be slower and needs more data without clear benefit. It is future work.

**Q13. What are the XGBoost parameters?**
300 trees, max depth 5, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, random_state 42 for repeatability. Tuning in Phase 2 (8.15) did not beat the defaults (8.13), so the model is not sensitive to them.

**Q14. Where does Prophet fit?**
Prophet is a time series library from Meta. We used it in the Phase 2 notebook as a comparison: with dates only 13.51 kg, with our wedding and festival regressors 8.43 kg. It is not in the app because XGBoost was better and simpler to serve.

# 5. Member 3: Ashish Negi, Data, Festival Calendar and Uploads

**Role:** owns the dataset and everything that turns raw data into model-ready data: the generator, the festival calendar, validation and cleaning, the upload and daily sales features.

**Files owned:** `data_pipeline/generate_dataset.py`, `data_pipeline/festival_calendar.py`, `data_pipeline/store.py`, `data/`, `tests/test_calendar.py`, `tests/test_data_store.py`.

**Technology:** pandas, NumPy, CSV and Excel (openpyxl), Python datetime.

**What he should be able to explain**

* Why the data is synthetic and how it was generated (honest answer in Q1).
* The festival score and the wedding season definition.
* What validation does to an uploaded file, and how daily entries are stored.
* The difference between append and replace, and the sales log.

**Demo duty:** upload a CSV (append), show the validation report, add a day on Daily sales entry, show the sales log, then Reset to original data.

## Viva questions and answers

**Q1. Is the data real?**
No, it is synthetic. Real shop data was not available, so we generated 3 years (2022 to 2024) of daily demand for 12 items (13,152 rows) with a generator that applies realistic rules: wedding months lift flowers, festivals lift specific items, mango sells only in summer, weekends differ, and random noise is added. The generator is in the project and our test proves it recreates the dataset exactly. The app also accepts real data through upload, which is the intended real-world use.

**Q2. How do you define the festival score?**
1.0 on the festival day, then it drops linearly by 0.25 per day over 4 days before and after, and 0 outside that. So the day after Diwali is 0.75 and 5 days away is 0. It gives the model a smooth "closeness to festival" signal instead of a single yes or no.

**Q3. Which festivals and which dates?**
Diwali, Durga Puja / Navratri, Ganesh Chaturthi, Holi and Eid. Dates for 2022 to 2024 are the ones used in the data, 2025 to 2027 are approximate and should be checked against an official calendar before real use.

**Q4. What is wedding season?**
A flag that is 1 in the months November, December, January, February, April, May and June, when many weddings take place in India.

**Q5. What checks happen when a file is uploaded?**
Columns are matched by name aliases (date, item or product, kg or quantity or sales). Rows with an invalid date, missing or negative quantity are dropped and counted. Item names are cleaned to the canonical form (for example "rose" becomes "Rose"). Duplicates (same date and item) are removed. The calendar features are added. The user sees a report of rows accepted and dropped, new items found and warnings.

**Q6. What if an item has too little history?**
The model needs at least 120 rows per item to learn its pattern. Items below that are kept in the data but not forecast, and a warning tells the user how many days are needed. We tested this with an item named Dragonfruit.

**Q7. Append versus replace?**
Append merges the new rows into the current dataset and, if the same date and item exist, the new value wins. Replace discards the current dataset and uses only the file. Reset to original brings back the project dataset.

**Q8. How does daily entry work and what is the sales log?**
The user enters today's kg per item (or a multi-day grid). It is saved as rows in the active dataset (same date and item is overwritten) and also written to `sales_log.csv` with the time of entry, as an audit trail. Future dates are refused. The user can then retrain so the model learns from it.

**Q9. What cleaning did the raw data need?**
The generated data is clean, but the pipeline handles what real shop data brings: text dates, mixed capitalisation, missing values, negatives and duplicates.

**Q10. Why CSV and not a database?**
One shop, about 13 thousand rows, and the user may want to open the file in Excel. A CSV is simple and transparent. For many shops we would move to a database.

**Q11. How is demand different from sales?**
Sales can be capped by stock (if we ran out, real demand was higher). We treat the recorded kg as demand. This is a known limitation, and we mention it as future work.

# 6. Member 4: Priyansu Nautiyal, Front End and Dashboard

**Role:** designs and builds the dashboard users see: navigation, forms, charts, tables, messages and the look of the whole app.

**Files owned:** `frontend/app.py`, `frontend/ui.py`, `frontend/api_client.py`, `frontend/views/*.py`, `.streamlit/config.toml`.

**Technology:** Streamlit, Plotly, requests, HTML/CSS inside Streamlit.

**What he should be able to explain**

* Each page, what it shows, and which API call feeds it.
* Why Streamlit and Plotly.
* How state, caching and error messages work.
* The design choices (colour, layout, plain words for a shop owner).

**Demo duty:** walk through all pages, set a forecast horizon, show the festival markers, download a forecast table.

## Viva questions and answers

**Q1. Why Streamlit?**
It lets us build an interactive web dashboard in Python without separate HTML, CSS and JavaScript, so the team can focus on data and ML. It is widely used for data apps.

**Q2. Why Plotly?**
Charts are interactive (hover, zoom, legend toggle), which suits forecast and interval charts, and Plotly integrates directly with Streamlit.

**Q3. How does the dashboard get its data?**
Through `api_client.py`, which calls the FastAPI endpoints with `requests` and returns JSON. The pages contain no model code at all.

**Q4. What does the Forecast page show?**
History, the forecast line, the 90% interval band and dotted lines on festival days for a chosen item and horizon, plus a table you can download as CSV.

**Q5. What does the Overview page show?**
Key numbers (items tracked, days of data, model MAE, latest data date), total expected demand per item for the next N days, and the next festivals with their dates.

**Q6. How does multi-page navigation work?**
`st.navigation` and `st.Page` in `app.py` group pages into Dashboard, Data and Analysis sections. Each page is a file in `frontend/views/`. We named the folder `views` and not `pages` because Streamlit auto-detects a folder named `pages` and would show a second, duplicate menu.

**Q7. How do you avoid calling the API too often?**
Read calls are cached for 2 minutes (`st.cache_data`), and the cache is cleared after any change such as an upload, daily entry or retrain.

**Q8. What if the backend is not running?**
Every page calls `require_backend()` first. It shows a clear message ("start the backend") instead of a stack trace, and the sidebar shows connection status.

**Q9. How does the Daily sales entry page support time series entry?**
It has three tabs: one day (a table of all items for a date), several days (a date-by-item grid for filling a week or month at once) and the sales log. It validates dates and shows what was saved.

**Q10. What is the Seasonal insights heat map?**
Rows are items and columns are months. Each cell is that item's average daily demand in that month divided by its own yearly average, so 1.0 is normal and 3.5 means 3.5 times normal. It shows mango in April to June and flowers in wedding months.

**Q11. How did you make it usable for a shop owner?**
Plain words (kg, "order", "wastage risk"), defaults chosen to be sensible (Marigold forecast, Rose planner), help text on the safety level and shelf life, and colour used consistently (teal for forecast, orange for festivals).

# 7. Questions for the whole team

**Q1. What is the novelty or contribution of the project?**
Demand forecasting specifically for event-driven perishables, using wedding and festival proximity as features, joined with a stock planner and wastage warning, and a working system where a shop can feed in its own data and retrain.

**Q2. Show the model comparison (Phase 2 benchmark).**

| Model | MAE (kg) |
|---|---|
| Naive item average | 34.06 |
| Same day last year | 15.97 |
| Prophet, dates only | 13.51 |
| Prophet + wedding and festival regressors | 8.43 |
| XGBoost default | 8.13 |
| XGBoost tuned | 8.15 |
| XGBoost with lags (next day only) | 8.08 |

**Q3. What are the limitations?**
Synthetic data; approximate festival dates after 2024; shelf lives are assumptions; demand equals recorded sales; no weather, price or promotion features; CSV storage; no login.

**Q4. What is future work?**
Real shop data, weather and price features, an LSTM or hybrid model, a database, authentication, scheduled retraining, mobile app, and SMS or WhatsApp order alerts.

**Q5. Why is the model good even though accuracy is "only" 85%?**
MAPE of 15% includes unavoidable random noise in the data. Against the naive baseline the error is cut by 76%, and in kg terms the error is about 8 kg per item per day.

**Q6. How do you know the model is not overfitting?**
The reported error is measured on 90 days it never saw, split by time. The interval check and baselines on the same window confirm it.

**Q7. How does the system handle a new item?**
After upload it appears in the data. Once it has at least 120 days it is included in training, and appears in forecasts after the next retrain.

**Q8. What does each team member's work depend on from the others?**
Member 3's clean data feeds Member 2's model. Member 2's model is served by Member 1's API. Member 4's dashboard calls Member 1's API. Member 1 tests all of it.

# 8. Quick run checklist for the demo day

1. Activate the environment, run `python start.py`.
2. Open http://localhost:8501 and http://localhost:8000/docs in two tabs.
3. Have a small CSV ready for upload (download the template from the Upload page).
4. If anything is changed during the demo, press "Reset to original data" on the Upload page.
5. Run `python -m pytest` in a terminal to show 25 tests passing.
6. If the dashboard shows "backend not reachable", the API is not running: run `python -m uvicorn backend.main:app --port 8000`.

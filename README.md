
# DemandSense 📊
### Seasonal Demand Forecasting for Event-Driven Perishable Goods

DemandSense is a machine learning-based demand forecasting application designed to predict future demand for **flowers, vegetables, and fruits**.

The project uses historical sales data, seasonal patterns, wedding seasons, festival information, and calendar features to forecast product demand and help businesses make better inventory and stock-planning decisions.

---

## 🚀 Project Overview

DemandSense helps businesses answer an important question:

**How much stock will we need in the coming days?**

Traditional forecasting methods often rely on historical averages. However, demand for perishable goods can change significantly during festivals, weddings, weekends, and seasonal periods.

DemandSense addresses this problem using machine learning and time-series forecasting techniques.

## ✨ Key Features

- **Demand Forecasting:** Predict demand for the next 7, 14, and 30 days.
- **Interactive Dashboard:** Visualize expected demand and product-wise statistics.
- **Stock Planner:** Support inventory planning using forecasted demand.
- **Daily Sales Entry:** Enter daily product sales records.
- **Data Upload:** Upload sales datasets for forecasting and analysis.
- **Model Performance:** Compare different forecasting approaches using MAE and MAPE.
- **Seasonal Insights:** Understand demand patterns associated with festivals and seasons.
- **Event-Based Forecasting:** Incorporate wedding seasons and festival-related features.

## 🧠 Machine Learning Models

The project explores multiple forecasting approaches:

| Model | Description |
|---|---|
| Naive Item Average | Uses historical average demand |
| Same Day Last Year | Uses demand from 364 days earlier |
| Prophet | Models trend and seasonality |
| Prophet + Event Regressors | Incorporates wedding and festival information |
| XGBoost Regressor | Uses gradient boosting with decision trees |
| XGBoost + TimeSeriesSplit | Evaluates tuned configurations using chronological validation |
| XGBoost + Lag Features | Uses previous demand for next-day forecasting |

### Model Performance

The following results are from the project's displayed evaluation:

| Forecasting Method | MAE |
|---|---:|
| Naive Item Average | 34.06 kg |
| Same Day Last Year | 15.97 kg |
| Prophet (Dates Only) | 13.51 kg |
| Prophet + Wedding & Festival Regressors | 8.43 kg |
| XGBoost (Default) | 8.13 kg |
| XGBoost (TimeSeriesSplit) | 8.15 kg |
| XGBoost + Lags (Next-Day Only) | 8.08 kg |

**Live model performance:**
- MAE: 8.14 kg
- MAPE: 15.0%
- Improvement over simple-average baseline: Approximately 76%

*Note: The lag-based experiment is a next-day forecasting result and is not directly comparable with the other 90-day evaluation results.*

## 🛠️ Technologies Used

**Programming Language**
- Python

**Machine Learning & Data Analysis**
- XGBoost
- Prophet
- Scikit-learn
- Pandas
- NumPy

**Visualization & Dashboard**
- Streamlit
- Plotly

**Development Tools**
- Visual Studio Code
- Git
- GitHub

## 📁 Project Structure

```text
DemandSense_Final_Project/
│
├── .streamlit/
│
├── backend/
│
├── data/
│
├── data_pipeline/
│
├── docs/
│
├── frontend/
│   ├── views/
│   ├── __init__.py
│   ├── api_client.py
│   ├── app.py
│   └── ui.py
│
├── ml/
│   ├── features.py
│   ├── predict.py
│   └── train.py
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── tests/
│
├── config.py
├── pytest.ini
├── requirements.txt
├── requirements-optional.txt
└── start.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd DemandSense_Final_Project
```

### 3. Create a virtual environment

```bash
python -m venv env
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\env\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the application

From the main project directory:

```bash
python start.py
```

### 7. Open the dashboard

Visit:

```text
http://localhost:8501
```

## 📈 Model Evaluation

The project evaluates forecasting models using:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted demand.

**Mean Absolute Percentage Error (MAPE)**

Measures the average absolute prediction error as a percentage.

**Time-Series Validation**

Uses chronological validation to avoid training on future observations when evaluating earlier periods.

## 🎯 Project Objectives

1. Predict future demand for perishable products.
2. Identify seasonal and event-driven demand patterns.
3. Reduce forecasting errors using machine learning.
4. Improve inventory and stock-planning decisions.
5. Compare different forecasting algorithms.
6. Minimize potential overstocking and understocking.

## 🔮 Future Enhancements

- Integration with real-time sales data.
- Automated data collection and preprocessing.
- More advanced seasonal forecasting.
- Product-wise inventory optimization.
- Deployment on a cloud platform.
- Integration with business inventory systems.


## 📌 Conclusion

DemandSense combines machine learning, time-series forecasting, and interactive data visualization to support demand planning for perishable goods.

By incorporating historical demand, calendar patterns, wedding seasons, and festivals, the project aims to provide more reliable demand predictions and help businesses make informed inventory decisions.

---


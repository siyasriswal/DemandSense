"""
DemandSense dashboard.

Start:  streamlit run frontend/app.py      (the backend must be running, see start.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import frontend.api_client as api
from frontend.views import about, daily_entry, forecast, insights, overview, performance, planner, upload

st.set_page_config(page_title="DemandSense", page_icon=":material/local_florist:", layout="wide")

nav = st.navigation({
    "Dashboard": [
        st.Page(overview.render, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(forecast.render, title="Forecast", icon=":material/query_stats:", url_path="forecast"),
        st.Page(planner.render, title="Stock planner", icon=":material/inventory_2:", url_path="planner"),
    ],
    "Data": [
        st.Page(daily_entry.render, title="Daily sales entry", icon=":material/edit_calendar:", url_path="daily-entry"),
        st.Page(upload.render, title="Upload data", icon=":material/upload_file:", url_path="upload"),
    ],
    "Analysis": [
        st.Page(performance.render, title="Model performance", icon=":material/speed:", url_path="performance"),
        st.Page(insights.render, title="Seasonal insights", icon=":material/calendar_month:", url_path="insights"),
        st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
    ],
})

with st.sidebar:
    try:
        info = api.model_info()
        st.caption(f"Backend connected\n\nModel trained {info['trained_at']}\n\nData up to {info['data_end']}")
    except api.ApiError:
        st.caption("Backend not connected")

nav.run()

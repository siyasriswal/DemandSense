"""Thin wrapper around the backend API. The dashboard never reads files or models directly, it only calls these."""
import requests
import streamlit as st

from config import API_URL


class ApiError(Exception):
    pass


def _handle(resp):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(detail if isinstance(detail, str) else str(detail))
    return resp.json()


def get(path, **params):
    try:
        return _handle(requests.get(f"{API_URL}{path}", params=params, timeout=60))
    except requests.ConnectionError:
        raise ApiError("backend_down")


def post(path, json=None, files=None, data=None):
    try:
        return _handle(requests.post(f"{API_URL}{path}", json=json, files=files, data=data, timeout=180))
    except requests.ConnectionError:
        raise ApiError("backend_down")


def get_text(path):
    return requests.get(f"{API_URL}{path}", timeout=30).text


# Cached reads. Cleared by clear_cache() whenever data or the model changes.
@st.cache_data(ttl=120, show_spinner=False)
def items():
    return get("/items")


@st.cache_data(ttl=120, show_spinner=False)
def forecast(item, days):
    return get("/forecast", item=item, days=days)


@st.cache_data(ttl=120, show_spinner=False)
def forecast_summary(days):
    return get("/forecast/summary", days=days)


@st.cache_data(ttl=120, show_spinner=False)
def history(item, days):
    return get("/history", item=item, days=days)


@st.cache_data(ttl=120, show_spinner=False)
def metrics():
    return get("/metrics")


@st.cache_data(ttl=120, show_spinner=False)
def backtest(item):
    return get("/backtest", item=item)


@st.cache_data(ttl=120, show_spinner=False)
def model_info():
    return get("/model/info")


@st.cache_data(ttl=120, show_spinner=False)
def seasonality():
    return get("/insights/seasonality")


@st.cache_data(ttl=120, show_spinner=False)
def upcoming(n=4):
    return get("/calendar/upcoming", n=n)


@st.cache_data(ttl=120, show_spinner=False)
def dataset_summary():
    return get("/dataset/summary")


def clear_cache():
    st.cache_data.clear()

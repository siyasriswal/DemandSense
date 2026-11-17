"""Colours, small layout helpers and chart styling shared by every page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import frontend.api_client as api

TEAL = "#0F766E"
ORANGE = "#E8590C"
GREY = "#94A3B8"
BLUE = "#2563EB"
RISK_COLOR = {"Low": "#15803D", "Medium": "#B45309", "High": "#B91C1C"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1 {font-weight: 700; letter-spacing: -0.5px;}
div[data-testid="stMetric"] {background: #F1F5F4; border-radius: 10px; padding: 14px 16px;}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #52606D;}
.chip {display:inline-block; padding: 4px 12px; margin: 0 8px 6px 0; border-radius: 999px;
       background: #FFF4E6; color: #9A3412; font-size: 0.85rem; border: 1px solid #FDBA74;}
.risk {display:inline-block; padding: 3px 14px; border-radius: 999px; color: white; font-weight: 600;}
.small-note {color: #52606D; font-size: 0.86rem;}
</style>
"""


def pretty(item):
    return item.replace("_", " ")


def page_header(title, subtitle):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def show_chart(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:           # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def style(fig, height=420, title=None):
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=13, color="#1F2933"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor="#CBD2D9")
    fig.update_yaxes(gridcolor="#E4E7EB", zeroline=False, title_text="kg per day")
    return fig


def item_picker(label="Item", key=None, default=None):
    rows = [r for r in api.items() if r["trained"]]
    names = [r["item"] for r in rows]
    index = names.index(default) if default in names else 0
    return st.selectbox(label, names, index=index, format_func=pretty, key=key)


def festival_markers(fig, forecast_rows):
    """Draw a dotted line on each festival day inside the forecast."""
    for r in forecast_rows:
        if r.get("festival") and r["festival_score"] >= 1.0:
            fig.add_shape(type="line", x0=r["date"], x1=r["date"], y0=0, y1=1, yref="paper",
                          line=dict(color="#B45309", width=1.5, dash="dot"))
            fig.add_annotation(x=r["date"], y=1, yref="paper", text=r["festival"], showarrow=False,
                               yanchor="bottom", font=dict(size=11, color="#B45309"))
    return fig


def to_frame(rows):
    df = pd.DataFrame(rows)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    return df


def require_backend():
    """Stop the page with a clear message if the API is not running."""
    try:
        api.get("/health")
    except api.ApiError:
        st.error("The backend API is not running.")
        st.code("python start.py            # starts backend and dashboard together\n"
                "# or, in a separate terminal:\n"
                "uvicorn backend.main:app --port 8000", language="bash")
        st.stop()

import plotly.graph_objects as go
import streamlit as st

import frontend.api_client as api
from frontend import ui


def render():
    ui.page_header("Demand forecast", "Pick an item and see how much of it will be needed in the coming days")
    ui.require_backend()

    left, mid, right = st.columns([1.2, 1.5, 1.2])
    with left:
        item = ui.item_picker("Item", key="fc_item", default="Marigold")
    with mid:
        days = st.slider("Days to forecast", 7, 90, 30)
    with right:
        history_days = st.selectbox("Past data to show", [60, 120, 365], index=1, format_func=lambda d: f"Last {d} days")

    fc = api.forecast(item, days)
    hist = ui.to_frame(api.history(item, history_days))
    f = ui.to_frame(fc["forecast"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["demand_kg"], name="Actual demand", line=dict(color=ui.TEAL, width=2)))
    fig.add_trace(go.Scatter(x=f["date"], y=f["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["date"], y=f["lower"], fill="tonexty", fillcolor="rgba(232,89,12,0.15)",
                             line=dict(width=0), name="Likely range (90%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["date"], y=f["forecast"], name="Forecast", line=dict(color=ui.ORANGE, width=2.5)))
    ui.festival_markers(fig, fc["forecast"])
    ui.style(fig, height=470)
    ui.show_chart(fig)

    k1, k2, k3, k4 = st.columns(4)
    peak = f.loc[f["forecast"].idxmax()]
    k1.metric(f"Total, next {days} days", f"{fc['total_kg']:,.0f} kg")
    k2.metric("Average per day", f"{fc['total_kg'] / days:,.1f} kg")
    k3.metric("Busiest day", peak["date"].strftime("%d %b"), f"{peak['forecast']:,.0f} kg", delta_color="off")
    k4.metric("Festival days in period", int((f["festival_score"] >= 1).sum()))

    reasons = []
    if f["wedding"].any():
        share = f["wedding"].mean() * 100
        reasons.append(f"{share:.0f}% of these days fall in wedding season, which lifts demand for this item.")
    names = sorted({x for x in f["festival"].dropna()})
    if names:
        reasons.append("Festival effect expected around: " + ", ".join(names) + ".")
    if not reasons:
        reasons.append("No wedding season or festival in this period, so demand follows the normal weekly pattern.")
    st.info(" ".join(reasons))

    with st.expander("Day-by-day numbers"):
        t = f[["date", "forecast", "lower", "upper", "wedding", "festival"]].copy()
        t["date"] = t["date"].dt.strftime("%a, %d %b %Y")
        t["wedding"] = t["wedding"].map({1: "Yes", 0: ""})
        t["festival"] = t["festival"].fillna("")
        t.columns = ["Date", "Forecast (kg)", "Low (kg)", "High (kg)", "Wedding season", "Festival"]
        ui.show_table(t)
        csv = f.assign(item=item)[["date", "item", "forecast", "lower", "upper"]].to_csv(index=False)
        st.download_button("Download forecast as CSV", csv, file_name=f"forecast_{item}_{days}d.csv", mime="text/csv")
    st.caption(f"Model trained {fc['model_trained_at']}. The shaded band shows where 90% of past forecast errors fell.")

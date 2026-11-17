import plotly.graph_objects as go
import streamlit as st

import frontend.api_client as api
from frontend import ui


def render():
    ui.page_header("DemandSense", "Seasonal demand forecasting for flowers, vegetables and fruits")
    ui.require_backend()

    summary = api.dataset_summary()
    m = api.metrics()["live"]
    horizon = st.segmented_control("Forecast period", [7, 14, 30], default=7, format_func=lambda d: f"Next {d} days") or 7
    fc = api.forecast_summary(horizon)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items tracked", summary["items"])
    c2.metric("Days of data", f"{summary['rows'] // max(summary['items'], 1):,}", help="Average number of days per item")
    c3.metric("Forecast error (MAE)", f"{m['xgb_mae']:.1f} kg",
              delta=f"{-(m['baseline_mae'] - m['xgb_mae']):.1f} kg vs simple average", delta_color="inverse",
              help=f"Average gap between forecast and actual over the last {m['test_days']} days of data")
    c4.metric("Latest data", summary["date_max"])

    upcoming = api.upcoming(4)
    if upcoming:
        st.markdown("**Coming up:** " + "".join(
            f"<span class='chip'>{u['festival']} &middot; {u['date']} ({u['days_away']} days)</span>" for u in upcoming),
            unsafe_allow_html=True)

    st.subheader(f"Expected demand, next {horizon} days (from {fc['start']})")
    df = ui.to_frame(fc["items"]).sort_values("total_kg")
    colors = df["category"].map({"Flowers": ui.ORANGE, "Vegetables": ui.TEAL, "Fruits": ui.BLUE}).fillna(ui.GREY)
    fig = go.Figure(go.Bar(x=df["total_kg"], y=df["item"].map(ui.pretty), orientation="h", marker_color=colors,
                           text=df["total_kg"].map(lambda v: f"{v:,.0f} kg"), textposition="outside",
                           hovertemplate="%{y}: %{x:,.0f} kg<extra></extra>"))
    ui.style(fig, height=430)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(title_text=f"kg over {horizon} days", showgrid=True, gridcolor="#E4E7EB",
                     range=[0, float(df["total_kg"].max()) * 1.18])
    fig.update_yaxes(title_text="")
    ui.show_chart(fig)
    st.markdown("<span class='small-note'>Orange = flowers, teal = vegetables, blue = fruits.</span>", unsafe_allow_html=True)

    st.subheader("Item details")
    table = df.sort_values("total_kg", ascending=False)[
        ["item", "category", "total_kg", "avg_kg_per_day", "peak_day", "peak_kg", "recent_total_kg", "change_pct"]].copy()
    table["item"] = table["item"].map(ui.pretty)
    table.columns = ["Item", "Category", f"Forecast ({horizon} d, kg)", "Avg kg/day", "Peak day", "Peak kg",
                     f"Last {horizon} d actual (kg)", "Change %"]
    ui.show_table(table)
    st.caption("Change % compares the forecast with the same number of most recent days in the data.")

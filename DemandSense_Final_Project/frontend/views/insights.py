import plotly.graph_objects as go
import streamlit as st

import frontend.api_client as api
from frontend import ui

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def render():
    ui.page_header("Seasonal insights", "What drives demand for each item")
    ui.require_backend()
    data = api.seasonality()

    st.subheader("Average daily demand by month")
    items = [m["item"] for m in data["monthly"]]
    z = [m["values"] for m in data["monthly"]]
    # scale each row to its own average so small and large items can be compared
    rel = []
    for row in z:
        vals = [v for v in row if v is not None]
        mean = sum(vals) / len(vals) if vals else 1
        rel.append([None if v is None else round(v / mean, 2) for v in row])
    fig = go.Figure(go.Heatmap(z=rel, x=MONTHS, y=[ui.pretty(i) for i in items], colorscale="YlOrRd",
                               customdata=z, hovertemplate="%{y}, %{x}<br>%{customdata:.1f} kg/day (%{z:.2f}x its average)<extra></extra>",
                               colorbar=dict(title="x average")))
    ui.style(fig, height=470)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text="", autorange="reversed")
    ui.show_chart(fig)
    st.caption("Darker means demand is higher than that item's own yearly average. Rose and Jasmine glow in wedding months, "
               "Mango only from April to June.")

    st.subheader("How much weddings and festivals lift demand")
    lifts = [x for x in data["lifts"] if x["wedding_lift"] is not None]
    lifts.sort(key=lambda x: -(x["wedding_lift"] or 0))
    fig = go.Figure()
    fig.add_bar(x=[ui.pretty(x["item"]) for x in lifts], y=[x["wedding_lift"] for x in lifts], name="Wedding season", marker_color=ui.TEAL)
    fig.add_bar(x=[ui.pretty(x["item"]) for x in lifts], y=[x["festival_lift"] for x in lifts], name="Festival days", marker_color=ui.ORANGE)
    ui.style(fig, height=400)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="demand vs a normal day (x)")
    fig.add_hline(y=1, line_dash="dot", line_color=ui.GREY)
    ui.show_chart(fig)
    st.caption("A value of 3 means demand is three times a normal day. Wedding season = November to February and April to June. "
               "Festival days = within two days of Diwali, Navratri, Ganesh Chaturthi, Holi or Eid.")

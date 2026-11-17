import plotly.graph_objects as go
import streamlit as st

import frontend.api_client as api
from frontend import ui


def render():
    ui.page_header("Stock planner", "Turn the forecast into a buying decision and a wastage warning")
    ui.require_backend()

    rows = {r["item"]: r for r in api.items() if r["trained"]}
    c1, c2, c3 = st.columns(3)
    with c1:
        item = ui.item_picker("Item", key="pl_item", default="Rose")
        stock = st.number_input("Stock on hand today (kg)", min_value=0.0, value=100.0, step=10.0)
    with c2:
        days = st.slider("Planning period (days)", 3, 30, 7)
        level = st.select_slider("Safety level", [0.80, 0.90, 0.95, 0.99], value=0.95,
                                 format_func=lambda v: f"{int(v * 100)}%",
                                 help="Higher means fewer stock-outs but more stock kept as a cushion.")
    with c3:
        shelf = st.number_input("Shelf life (days)", min_value=1, max_value=60, value=int(rows[item]["shelf_life_days"]),
                                key=f"shelf_{item}", help="How many days this item stays sellable. Assumed value, change it if you know better.")

    if not st.button("Calculate", type="primary"):
        st.info("Enter your stock and press Calculate.")
        return
    try:
        res = api.post("/planner", json={"item": item, "stock_kg": stock, "days": days, "service_level": level,
                                         "shelf_life_days": int(shelf)})
    except api.ApiError as exc:
        st.error(str(exc))
        return

    a, b, c, d = st.columns(4)
    a.metric("Order now", f"{res['recommended_order_kg']:,.0f} kg")
    b.metric(f"Target stock ({res['cycle_days']} d)", f"{res['target_stock_kg']:,.0f} kg")
    c.metric("Stock lasts", f"{res['stock_cover_days']} days" if res["stock_cover_days"] else f"{days}+ days")
    with d:
        st.markdown("**Wastage risk**")
        st.markdown(f"<span class='risk' style='background:{ui.RISK_COLOR[res['wastage_risk']]}'>{res['wastage_risk']}</span>",
                    unsafe_allow_html=True)
        st.caption(f"{res['expected_spoilage_kg']:,.0f} kg may expire ({res['spoilage_pct']:.0f}%)")

    for n in res["notes"]:
        st.markdown(f"- {n}")

    daily = ui.to_frame(res["daily"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["forecast"], name="Forecast demand", marker_color=ui.ORANGE, opacity=0.75))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["remaining_stock"], name="Stock left (no new orders)",
                             line=dict(color=ui.TEAL, width=3)))
    ui.style(fig, height=380, title="Stock left if you do not reorder")
    ui.show_chart(fig)

    with st.expander("How this is calculated"):
        st.markdown(
            f"""
- **Forecast** comes from the trained XGBoost model for each of the next days.
- **Batch size:** fresh items are bought for about one shelf life at a time. For {ui.pretty(item)} that is
  **{res['cycle_days']} day(s)**, with expected demand of **{res['cycle_demand_kg']:,.0f} kg**.
- **Safety stock** = z x error x square root of days = **{res['safety_stock_kg']:,.0f} kg**, where z is
  {res['service_level']:.0%} service level and the error is the typical size of past forecast mistakes for this item.
- **Order now** = target stock minus stock on hand (never below zero).
- **Wastage** = stock on hand that the forecast says will not be sold within the shelf life.
- Shelf life values are assumptions and can be edited above.
""")

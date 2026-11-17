import plotly.graph_objects as go
import streamlit as st

import frontend.api_client as api
from frontend import ui


def render():
    ui.page_header("Model performance", "How accurate the forecasts are, measured on days the model never saw")
    ui.require_backend()

    data = api.metrics()
    live, bench = data["live"], data["phase2_benchmark"]
    a, b, c, d = st.columns(4)
    a.metric("Forecast error (MAE)", f"{live['xgb_mae']:.2f} kg")
    b.metric("Percentage error (MAPE)", f"{live['xgb_mape']:.1f}%")
    c.metric("Simple-average error", f"{live['baseline_mae']:.2f} kg")
    d.metric("Improvement", f"{(1 - live['xgb_mae'] / live['baseline_mae']) * 100:.0f}%")
    st.caption(f"Test period: last {live['test_days']} days of data ({live['test_window'][0]} to {live['test_window'][1]}), "
               f"{live['n_test']:,} item-days. Trained {live['trained_at']}.")

    tab1, tab2, tab3, tab4 = st.tabs(["MAE improvement", "Per item", "Predicted vs actual", "What the model learned"])

    with tab1:
        if data["is_original_data"] and bench:
            st.markdown("**Phase 2 comparison on the original dataset.** Every model is tested on the same 90 days. "
                        "Lower is better.")
            names = [r["Model"] for r in bench]
            maes = [r["MAE (kg)"] for r in bench]
            colors = [ui.GREY, ui.GREY, "#7C3AED", "#7C3AED", ui.ORANGE, ui.ORANGE, ui.TEAL][:len(names)]
            fig = go.Figure(go.Bar(x=maes[::-1], y=names[::-1], orientation="h", marker_color=colors[::-1],
                                   text=[f"{v:.2f} kg" for v in maes[::-1]], textposition="outside"))
            ui.style(fig, height=380)
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title_text="MAE (kg)", showgrid=True, gridcolor="#E4E7EB", range=[0, max(maes) * 1.15])
            fig.update_yaxes(title_text="")
            ui.show_chart(fig)
            st.caption("The last row is a next-day forecast (it uses yesterday's real demand), so it is not a 90-day result. "
                       "Prophet models were run in the Phase 2 notebook; the live model below is XGBoost.")
        st.markdown("**Live model** (recalculated every time the model is retrained)")
        names = [r["model"] for r in live["ladder"]]
        maes = [r["mae"] for r in live["ladder"]]
        fig = go.Figure(go.Bar(x=maes[::-1], y=names[::-1], orientation="h", marker_color=[ui.TEAL, ui.GREY, ui.GREY][::-1][:len(names)],
                               text=[f"{v:.2f} kg" for v in maes[::-1]], textposition="outside"))
        ui.style(fig, height=250)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="MAE (kg)", showgrid=True, gridcolor="#E4E7EB", range=[0, max(maes) * 1.15])
        fig.update_yaxes(title_text="")
        ui.show_chart(fig)
        ui.show_table(__import__("pandas").DataFrame(live["ladder"])[["model", "mae", "mape", "improvement_pct"]].rename(
            columns={"model": "Model", "mae": "MAE (kg)", "mape": "MAPE (%)", "improvement_pct": "Improvement vs simple average (%)"}))

    with tab2:
        df = ui.to_frame(live["per_item"]).sort_values("model_mae")
        fig = go.Figure()
        fig.add_bar(x=df["item"].map(ui.pretty), y=df["naive_mae"], name="Simple average", marker_color=ui.GREY)
        fig.add_bar(x=df["item"].map(ui.pretty), y=df["model_mae"], name="DemandSense model", marker_color=ui.TEAL)
        ui.style(fig, height=400)
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="MAE (kg)")
        ui.show_chart(fig)
        t = df[["item", "category", "naive_mae", "model_mae", "model_mape", "improvement_pct"]].copy()
        t["item"] = t["item"].map(ui.pretty)
        t.columns = ["Item", "Category", "Simple average MAE", "Model MAE", "Model MAPE (%)", "Improvement (%)"]
        ui.show_table(t)
        st.caption("MAPE is high for items with very low demand in some months (for example Mango outside April to June), "
                   "because a small miss is a large percentage of a small number.")

    with tab3:
        item = ui.item_picker("Item", key="bt_item", default="Marigold")
        bt = ui.to_frame(api.backtest(item))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"], name="Actual", line=dict(color=ui.TEAL, width=2), mode="lines+markers",
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["predicted"], name="Model", line=dict(color=ui.ORANGE, width=2.5)))
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["naive"], name="Simple average", line=dict(color=ui.GREY, dash="dash")))
        ui.style(fig, height=430)
        ui.show_chart(fig)
        st.caption("These are the held-out days. The model was trained only on data before this period.")

    with tab4:
        info = api.model_info()
        imp = sorted(info["feature_importance"].items(), key=lambda kv: kv[1])
        label = {"is_wedding_season": "Wedding season", "item_encoded": "Which item", "festival_proximity_score": "Festival nearby",
                 "doy_sin": "Time of year (sin)", "doy_cos": "Time of year (cos)", "dow_num": "Day of week"}
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[label.get(k, k) for k, _ in imp], orientation="h", marker_color=ui.ORANGE))
        ui.style(fig, height=320)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="Importance", showgrid=True, gridcolor="#E4E7EB")
        fig.update_yaxes(title_text="")
        ui.show_chart(fig)
        st.markdown("The model relies mostly on **whether it is wedding season** and **which item** it is looking at. "
                    "Festival score matters less overall because it is non-zero on only a few days a year, but on those "
                    "days it changes demand a lot for items like Marigold.")

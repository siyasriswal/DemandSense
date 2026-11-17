from datetime import date, timedelta

import pandas as pd
import streamlit as st

import frontend.api_client as api
from frontend import ui


def _flash():
    msg = st.session_state.pop("daily_flash", None)
    if msg:
        st.success(msg)


def render():
    ui.page_header("Daily sales entry", "Record what was sold each day. The model learns from it the next time it is retrained")
    ui.require_backend()
    _flash()

    summary = api.dataset_summary()
    last = pd.Timestamp(summary["date_max"]).date()
    rows = [r for r in api.items() if r["forecastable"]]
    st.markdown(f"Latest day in the data: **{last:%d %b %Y}**. Entering a new day adds one row per item to the dataset; "
                "entering a day that already exists overwrites it.")

    tab1, tab2, tab3 = st.tabs(["One day", "Several days (time series)", "Sales log"])

    # ------------------------------------------------------------ one day
    with tab1:
        default_day = min(last + timedelta(days=1), date.today())
        day = st.date_input("Date", value=default_day, max_value=date.today(), key="daily_date")
        counter = st.session_state.setdefault("daily_counter", 0)
        table = pd.DataFrame({"Item": [ui.pretty(r["item"]) for r in rows],
                              "Usual kg per day": [r["avg_kg"] for r in rows],
                              "Sold today (kg)": [None] * len(rows)})
        edited = st.data_editor(table, hide_index=True, disabled=["Item", "Usual kg per day"], key=f"daily_edit_{counter}",
                                column_config={"Sold today (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.1f")})
        note = st.text_input("Note (optional)", placeholder="e.g. wedding order, market closed after 2 pm")
        retrain = st.checkbox("Retrain the model after saving", value=True,
                              help="Takes a few seconds. Untick to add several days first and retrain once.")
        if st.button("Save this day", type="primary"):
            entries = [{"item": r["item"], "kg": float(v)} for r, v in zip(rows, edited["Sold today (kg)"]) if pd.notna(v)]
            if not entries:
                st.warning("Type the kg sold for at least one item.")
            else:
                try:
                    res = api.post("/sales/daily", json={"date": str(day), "entries": entries, "note": note, "retrain": retrain})
                    api.clear_cache()
                    msg = f"Saved {res['saved']} item(s) for {day:%d %b %Y}."
                    if res.get("training"):
                        msg += f" Model retrained: forecast error is now {res['training']['xgb_mae']:.1f} kg."
                    st.session_state["daily_flash"] = msg
                    st.session_state["daily_counter"] = counter + 1
                    st.rerun()
                except api.ApiError as exc:
                    st.error(str(exc))

    # ------------------------------------------------------------ several days
    with tab2:
        st.markdown("Add a run of days at once, for example the last week, one row per date and item. "
                    "You can paste rows straight from Excel.")
        names = [r["item"] for r in rows]
        start = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "Item": pd.Series(dtype="object"),
                              "Sold (kg)": pd.Series(dtype="float")})
        grid = st.data_editor(start, num_rows="dynamic", hide_index=True, key="series_edit",
                              column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                             "Item": st.column_config.SelectboxColumn(options=names),
                                             "Sold (kg)": st.column_config.NumberColumn(min_value=0.0, format="%.1f")})
        retrain2 = st.checkbox("Retrain the model after saving", value=True, key="series_retrain")
        if st.button("Save all rows", type="primary", key="series_save"):
            grid = grid.dropna()
            if grid.empty:
                st.warning("Add at least one complete row (date, item and kg).")
            else:
                grid = grid.assign(day=pd.to_datetime(grid["Date"]).dt.date)
                days = sorted(grid["day"].unique())
                try:
                    saved = 0
                    for i, d in enumerate(days):
                        part = grid[grid["day"] == d]
                        res = api.post("/sales/daily", json={"date": str(d), "retrain": retrain2 and i == len(days) - 1,
                                                             "entries": [{"item": r["Item"], "kg": float(r["Sold (kg)"])} for _, r in part.iterrows()]})
                        saved += res["saved"]
                    api.clear_cache()
                    st.session_state["daily_flash"] = f"Saved {saved} row(s) across {len(days)} day(s)."
                    st.rerun()
                except api.ApiError as exc:
                    st.error(str(exc))

    # ------------------------------------------------------------ log
    with tab3:
        log = pd.DataFrame(api.get("/sales/log", limit=200))
        if log.empty:
            st.info("Nothing has been entered from the dashboard yet.")
        else:
            log["item"] = log["item"].map(ui.pretty)
            log.columns = ["Date", "Item", "Sold (kg)", "Entered at", "Note"]
            ui.show_table(log)
            st.download_button("Download log as CSV", log.to_csv(index=False), file_name="sales_log.csv", mime="text/csv")

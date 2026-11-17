import io

import pandas as pd
import streamlit as st

import frontend.api_client as api
from frontend import ui


def render():
    ui.page_header("Upload data", "Load a full sales history from a CSV or Excel file. The model is retrained on it")
    ui.require_backend()

    summary = api.dataset_summary()
    a, b, c, d = st.columns(4)
    a.metric("Rows in dataset", f"{summary['rows']:,}")
    b.metric("Items", summary["items"])
    c.metric("From", summary["date_min"])
    d.metric("To", summary["date_max"])
    st.caption("Using the original project dataset (synthetic)." if summary["is_original"]
               else "Using a custom dataset (original data plus your uploads or entries).")
    with st.expander("Items in the dataset"):
        t = pd.DataFrame(summary["per_item"])[["item", "category", "rows", "first_date", "last_date", "avg_kg", "forecastable"]]
        t["item"] = t["item"].map(ui.pretty)
        t.columns = ["Item", "Category", "Days of data", "First day", "Last day", "Average kg/day", "Can forecast"]
        ui.show_table(t)

    st.subheader("1. Prepare your file")
    st.markdown("One row per **date** and **item** with the quantity sold. Extra columns are ignored, and column names such as "
                "`product`, `quantity`, `sales` or `usage` are recognised.")
    st.code("date,item,demand_units_kg\n2025-01-01,Rose,52.5\n2025-01-01,Marigold,41.0\n2025-01-02,Rose,49.8", language="csv")
    st.download_button("Download a blank template", api.get_text("/dataset/template"), file_name="demandsense_template.csv",
                       mime="text/csv")

    st.subheader("2. Choose the file")
    file = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    mode = st.radio("What should happen to the existing data?",
                    ["append", "replace"], horizontal=True,
                    format_func=lambda m: "Add to existing data (same date and item is overwritten)" if m == "append"
                    else "Replace everything with this file")
    retrain = st.checkbox("Retrain the model after importing", value=True)

    if file is not None:
        try:
            raw = file.getvalue()
            preview = (pd.read_excel(io.BytesIO(raw)) if file.name.lower().endswith(("xlsx", "xls"))
                       else pd.read_csv(io.BytesIO(raw), sep=None, engine="python"))
            st.markdown(f"**Preview:** {len(preview):,} rows, columns: {', '.join(map(str, preview.columns))}")
            ui.show_table(preview.head(8))
        except Exception as exc:
            st.error(f"This file could not be read: {exc}")
            file = None

    if st.button("Import data", type="primary", disabled=file is None):
        with st.spinner("Checking the file and retraining the model..."):
            try:
                res = api.post("/dataset/upload", files={"file": (file.name, file.getvalue())},
                               data={"mode": mode, "retrain": "true" if retrain else "false"})
            except api.ApiError as exc:
                st.error(str(exc))
                return
        api.clear_cache()
        rep = res["report"]
        st.success(f"Imported {rep['rows_ok']:,} of {rep['rows_in']:,} rows ({rep['date_min']} to {rep['date_max']}).")
        if rep["new_items"]:
            st.info("New items found: " + ", ".join(ui.pretty(i) for i in rep["new_items"]))
        if rep["dropped"]:
            st.warning("Rows skipped: " + ", ".join(f"{v} ({k.replace('_', ' ')})" for k, v in rep["dropped"].items()))
        for w in rep["warnings"]:
            st.warning(w)
        if res.get("training"):
            t = res["training"]
            st.success(f"Model retrained. Forecast error on the last {t['test_days']} days: {t['xgb_mae']:.1f} kg "
                       f"(simple average: {t['baseline_mae']:.1f} kg).")
        if res.get("training_error"):
            st.error(res["training_error"])

    st.divider()
    with st.expander("Start over with the original dataset"):
        st.markdown("This removes everything you uploaded or typed in and goes back to the project dataset.")
        sure = st.checkbox("Yes, remove my uploaded and entered data")
        if st.button("Reset to original data", disabled=not sure):
            with st.spinner("Resetting and retraining..."):
                api.post("/dataset/reset")
            api.clear_cache()
            st.success("Back to the original dataset.")
            st.rerun()

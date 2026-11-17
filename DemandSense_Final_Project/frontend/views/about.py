import streamlit as st

from frontend import ui

DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#475569"];
  edge [color="#64748B", fontname="Helvetica", fontsize=9];
  data  [label="CSV data\\nuploads and\\ndaily entries", fillcolor="#E0F2F1"];
  store [label="Data layer\\nvalidate, add\\nfestival features", fillcolor="#E0F2F1"];
  ml    [label="ML layer\\nXGBoost model\\ntrain and forecast", fillcolor="#FFF4E6"];
  api   [label="FastAPI backend\\nREST endpoints\\n(port 8000)", fillcolor="#E0E7FF"];
  ui    [label="Streamlit dashboard\\n(port 8501)", fillcolor="#FCE7F3"];
  data -> store; store -> ml; ml -> api; store -> api; api -> ui [label="JSON over HTTP"];
}
"""


def render():
    ui.page_header("About the project", "Seasonal Demand Forecasting for Event-Driven Perishable Goods (Flowers, Catering Vegetables and Fruits)")

    st.subheader("What it does")
    st.markdown(
        "Flowers, vegetables and fruits sell in sharp waves around weddings and festivals. Stock too little and sales are lost, "
        "stock too much and it spoils. DemandSense predicts daily demand (kg) for 12 items, lets the user add real sales every day "
        "or upload a full history, and turns the forecast into a buying suggestion with a wastage warning.")

    st.subheader("How the parts connect")
    try:
        st.graphviz_chart(DOT)
    except Exception:
        st.code("CSV data -> Data layer -> ML layer -> FastAPI backend -> Streamlit dashboard")

    st.subheader("Technology")
    ui.show_table(__import__("pandas").DataFrame([
        ["Python", "Language for every part"],
        ["pandas, NumPy", "Reading, cleaning and reshaping the data"],
        ["scikit-learn", "Label encoding of items"],
        ["XGBoost", "The forecasting model (many small decision trees, each correcting the last)"],
        ["Prophet", "Used in the Phase 2 notebook as a comparison model"],
        ["FastAPI and Uvicorn", "Backend REST API and its web server"],
        ["Streamlit", "Dashboard front end"],
        ["Plotly", "Interactive charts"],
        ["pytest", "Automated tests"],
    ], columns=["Technology", "Used for"]))

    st.subheader("Team POKEMONS (MCA AI/DS, Section A)")
    ui.show_table(__import__("pandas").DataFrame([
        ["Siya Sriswal (Team Lead)", "Backend API, integration, testing, documentation"],
        ["Priyanshu Jimiwal", "Machine learning: features, models, evaluation, forecasting"],
        ["Ashish Negi", "Data: dataset generation, festival calendar, validation, uploads"],
        ["Priyansu Nautiyal", "Front end: dashboard pages, charts, user experience"],
    ], columns=["Member", "Responsibility"]))

    st.subheader("Limits to keep in mind")
    st.markdown(
        "- The project dataset is **synthetic** (generated from documented rules for weddings, festivals and weekends). "
        "Accuracy on real market data would be lower.\n"
        "- Festival dates for 2025 to 2027 are approximate.\n"
        "- The forecast uses the calendar only, not yesterday's sales, so it can look weeks ahead. A model that also used "
        "yesterday's sales scored slightly better in a next-day test but cannot forecast far ahead.\n"
        "- Shelf-life values in the stock planner are assumptions.")

"""BrightMart Retail - interactive dashboard (Streamlit).

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "brightmart_bundle.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
CLEAN_PATH = ROOT / "data" / "BrightMart_ML_Dataset_Cleaned.csv"
FIG = ROOT / "outputs" / "figures"

st.set_page_config(page_title="BrightMart Analytics", page_icon="🛒", layout="wide")

ACTIONS = {
    "Premium Spenders": "Loyalty rewards, early access and premium bundles. Best group to protect.",
    "Mainstream Shoppers": "Targeted discount coupons and bundle offers to raise purchase frequency.",
    "Budget Shoppers": "Low-cost channels (SMS / app notifications) and entry-level offers; avoid expensive campaigns.",
}


def build_artifacts():
    """(Re)create data, figures and models by running the full pipeline."""
    from src.analysis import main
    with st.spinner("Building models and figures (first run only)..."):
        main()


@st.cache_resource
def load_bundle():
    try:
        return joblib.load(BUNDLE_PATH)
    except Exception:          # missing file or scikit-learn version mismatch
        build_artifacts()
        return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_PATH)


if not (BUNDLE_PATH.exists() and METRICS_PATH.exists() and CLEAN_PATH.exists()):
    build_artifacts()

bundle, metrics, data = load_bundle(), load_metrics(), load_data()
dc, lr, seg = metrics["data_cleaning"], metrics["linear_regression"], metrics["segmentation"]

st.title("🛒 BrightMart Retail - Sales, Segmentation and Campaign Response")
st.caption("MBA (Data Science & AI) mini-project. The data is synthetic, so results illustrate "
           "the method and are not real BrightMart figures.")

tab_over, tab_sales, tab_cust, tab_models, tab_data = st.tabs(
    ["Overview", "Sales predictor", "Customer predictor", "Model comparison", "Data"])

# ---------------------------------------------------------------- Overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean customers", dc["clean_rows"], f"{dc['duplicates_removed']} duplicates removed")
    c2.metric("Sales model R²", f"{lr['r2']:.3f}")
    c3.metric("Sales model MAE", f"₹{lr['mae']:.1f}k")
    c4.metric("Segments (K-Means)", seg["k"], f"silhouette {seg['silhouette']:.2f}")

    st.subheader("Customer segments")
    st.dataframe(pd.DataFrame(seg["profiles"]).rename(columns={
        "Segment_Label": "Segment", "customers": "Customers", "avg_income": "Avg income (₹k)",
        "avg_spending": "Avg spending score", "avg_purchases": "Avg purchases",
        "response_rate": "Response rate"}), hide_index=True)
    col_a, col_b = st.columns(2)
    col_a.image(str(FIG / "fig6_kmeans_clusters.png"), caption="K-Means clusters")
    col_b.image(str(FIG / "fig1_sales_distribution_and_scatter.png"), caption="Sales and Ad Spend")

# ---------------------------------------------------------------- Sales predictor
with tab_sales:
    lo, hi = bundle["ad_spend_range"]
    st.subheader("Predict sales from advertising spend")
    budget = st.slider("Advertising budget (₹ thousand)", float(round(lo)), float(round(hi)), 30.0, 0.5)
    pred = float(bundle["linear_model"].predict(pd.DataFrame({"Ad_Spend": [budget]}))[0])
    c1, c2 = st.columns(2)
    c1.metric("Predicted sales", f"₹{pred:,.1f}k")
    c2.metric("Expected error (MAE)", f"± ₹{metrics['linear_regression']['mae']:.1f}k")
    st.caption(f"Equation: Sales = {lr['intercept']:.2f} + {lr['slope']:.2f} × Ad_Spend. "
               f"Valid only inside the tested range (₹{lo:.0f}k to ₹{hi:.0f}k). "
               "Shows association, not proof that advertising causes sales.")
    grid = pd.DataFrame({"Ad_Spend": [round(lo) + i for i in range(int(hi - lo) + 1)]})
    grid["Predicted sales"] = bundle["linear_model"].predict(grid)
    st.line_chart(grid.set_index("Ad_Spend"))

# ---------------------------------------------------------------- Customer predictor
with tab_cust:
    st.subheader("Score a customer")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 80, 35)
    income = c2.number_input("Annual income (₹ thousand)", 100, 1200, 540, step=10)
    spend = c3.slider("Spending score", 1, 100, 55)
    c4, c5 = st.columns(2)
    freq = c4.slider("Purchase frequency", 1, 12, 5)
    disc = c5.selectbox("Discount usage", ["Low", "Medium", "High"], index=1)

    cl = int(bundle["kmeans"].predict(bundle["scaler"].transform(
        pd.DataFrame({"Annual_Income": [income], "Spending_Score": [spend]})))[0])
    label = bundle["cluster_names"][cl]
    row = pd.DataFrame([[age, income, freq, bundle["discount_map"][disc]]],
                       columns=bundle["clf_features"])
    p_rf = float(bundle["random_forest"].predict_proba(row)[0][1])
    p_dt = float(bundle["decision_tree"].predict_proba(row)[0][1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Segment", label)
    m2.metric("Response probability (Random Forest)", f"{p_rf:.0%}")
    m3.metric("Response probability (Decision Tree)", f"{p_dt:.0%}")
    st.info(f"**Suggested action:** {ACTIONS[label]}")
    st.caption("Use probabilities to prioritise outreach, not to exclude customers: recall is low "
               "because only about a quarter of customers respond.")

# ---------------------------------------------------------------- Model comparison
with tab_models:
    st.subheader("Decision Tree vs Random Forest (test set)")
    clf = metrics["classification"]
    rows = []
    for key, name in [("decision_tree", "Decision Tree"), ("random_forest", "Random Forest")]:
        m = clf[key]
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"],
                     "TN/FP/FN/TP": "/".join(str(m["confusion"][k]) for k in ["TN", "FP", "FN", "TP"])})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    st.warning(f"A model that predicts \"No\" for everyone scores "
               f"{clf['majority_baseline_accuracy']:.1%} accuracy on this test set, so accuracy alone "
               "is misleading. Compare precision, recall and F1 for the \"Yes\" class.")
    a, b, c = st.columns(3)
    a.image(str(FIG / "fig7a_dt_confusion.png"))
    b.image(str(FIG / "fig7b_rf_confusion.png"))
    c.image(str(FIG / "fig7c_rf_feature_importance.png"))
    st.image(str(FIG / "fig4_decision_tree.png"), caption="Decision Tree (max_depth = 3)")

# ---------------------------------------------------------------- Data
with tab_data:
    st.subheader("Cleaned dataset")
    st.dataframe(data, hide_index=True)
    st.download_button("Download cleaned CSV", data.to_csv(index=False).encode(),
                       "BrightMart_ML_Dataset_Cleaned.csv", "text/csv")
    st.subheader("Data-cleaning summary")
    st.json(metrics["data_cleaning"])

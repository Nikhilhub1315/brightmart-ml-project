"""BrightMart Retail - Sales, Customer Segmentation and Campaign Response.

Runs the whole workflow from the mini-project report:
  Task A  data preparation + EDA
  Task B  Linear Regression (Ad_Spend -> Sales)
  Task C  Decision Tree (Campaign_Response)
  Task D  K-Means customer segmentation
  Task E  Random Forest + model comparison

Usage:  python src/analysis.py
If data/BrightMart_ML_Dataset.csv is missing it is generated first. To reproduce the
numbers of your own report, put your original CSV in data/ before running.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:                                   # works as script and as module
    from generate_data import RAW_PATH, generate
except ImportError:
    from src.generate_data import RAW_PATH, generate

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"
CLEAN_PATH = ROOT / "data" / "BrightMart_ML_Dataset_Cleaned.csv"
SEED = 42
DISCOUNT_MAP = {"Low": 0, "Medium": 1, "High": 2}
CLF_FEATURES = ["Age", "Annual_Income", "Purchase_Frequency", "Discount_Usage_Enc"]


# ----------------------------------------------------------------------------------
# Task A - data preparation
# ----------------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(RAW_PATH, index=False)
        print(f"Raw dataset not found - generated {RAW_PATH.name}")
    return pd.read_csv(RAW_PATH)


def clean(df: pd.DataFrame):
    report = {"raw_rows": int(len(df))}
    df = df.copy()

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Invalid values -> missing, then impute
    invalid_age = (df["Age"] < 18) | (df["Age"] > 100)
    invalid_sales = df["Sales"] < 0
    invalid_score = (df["Spending_Score"] < 1) | (df["Spending_Score"] > 100)
    report["invalid_age"] = int(invalid_age.sum())
    report["invalid_sales"] = int(invalid_sales.sum())
    report["invalid_spending_score"] = int(invalid_score.sum())
    df.loc[invalid_age, "Age"] = np.nan
    df.loc[invalid_sales, "Sales"] = np.nan
    df.loc[invalid_score, "Spending_Score"] = np.nan

    report["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v}

    for col in ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency",
                "Ad_Spend", "Sales"]:
        df[col] = df[col].fillna(df[col].median())
    df["Discount_Usage"] = df["Discount_Usage"].fillna(df["Discount_Usage"].mode()[0])

    report["clean_rows"] = int(len(df))
    report["missing_after"] = int(df.isna().sum().sum())
    df["Discount_Usage_Enc"] = df["Discount_Usage"].map(DISCOUNT_MAP)
    df["Response_Flag"] = (df["Campaign_Response"] == "Yes").astype(int)
    return df, report


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eda(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["Sales"], kde=True, ax=ax[0], color="#4C8BC2")
    ax[0].set(title="Distribution of Sales", xlabel="Sales (Rs thousands)")
    sns.scatterplot(data=df, x="Ad_Spend", y="Sales", ax=ax[1], color="#D2691E")
    ax[1].set(title="Ad Spend vs Sales", xlabel="Ad Spend (Rs thousands)",
              ylabel="Sales (Rs thousands)")
    save(fig, "fig1_sales_distribution_and_scatter.png")

    num = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Ad_Spend", "Sales"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    save(fig, "fig2_correlation_matrix.png")


# ----------------------------------------------------------------------------------
# Task B - linear regression
# ----------------------------------------------------------------------------------
def linear_regression(df):
    X, y = df[["Ad_Spend"]], df["Sales"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)
    res = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
        "prediction_30k": float(model.predict(pd.DataFrame({"Ad_Spend": [30]}))[0]),
        "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(Xtr, ytr, s=14, alpha=.7, label="Training data", color="#4C8BC2")
    ax.scatter(Xte, yte, s=18, label="Test data", color="#D2691E")
    line = np.linspace(float(X['Ad_Spend'].min()), float(X['Ad_Spend'].max()), 100)
    ax.plot(line, model.predict(pd.DataFrame({"Ad_Spend": line})), "k", lw=2,
            label="Fitted regression line")
    ax.set(title="Linear Regression: Sales vs Ad Spend",
           xlabel="Ad Spend (Rs thousands)", ylabel="Sales (Rs thousands)")
    ax.legend()
    save(fig, "fig3_linear_regression.png")
    return model, res


# ----------------------------------------------------------------------------------
# Task C / E - classifiers
# ----------------------------------------------------------------------------------
def clf_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def confusion_plot(y_true, y_pred, title, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set(title=title, xlabel="Predicted label", ylabel="True label")
    save(fig, name)


def classifiers(df):
    X, y = df[CLF_FEATURES], df["Response_Flag"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr)
    t_pred = tree.predict(Xte)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED).fit(Xtr, ytr)
    r_pred = rf.predict(Xte)

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(tree, feature_names=CLF_FEATURES, class_names=["No", "Yes"], filled=True,
              fontsize=8, ax=ax)
    ax.set_title("Decision Tree (max_depth=3) - Campaign Response")
    save(fig, "fig4_decision_tree.png")

    confusion_plot(yte, t_pred, "Decision Tree - Confusion Matrix", "fig7a_dt_confusion.png")
    confusion_plot(yte, r_pred, "Random Forest - Confusion Matrix", "fig7b_rf_confusion.png")

    imp = pd.Series(rf.feature_importances_, index=CLF_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    imp.plot.barh(ax=ax, color="#2E75B6")
    ax.set_title("Random Forest - Feature Importance")
    save(fig, "fig7c_rf_feature_importance.png")

    dt_m, rf_m = clf_metrics(yte, t_pred), clf_metrics(yte, r_pred)
    comp = pd.DataFrame({"Decision Tree": {k: dt_m[k] for k in ["accuracy", "precision", "recall", "f1"]},
                         "Random Forest": {k: rf_m[k] for k in ["accuracy", "precision", "recall", "f1"]}})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp.plot.bar(ax=ax, color=["#C8591A", "#2E7DC8"], rot=0)
    ax.set(title="Decision Tree vs Random Forest (test set)", ylim=(0, 1))
    save(fig, "fig8_model_comparison.png")

    res = {
        "test_rows": int(len(yte)), "test_responders": int(yte.sum()),
        "majority_baseline_accuracy": float(1 - yte.mean()),
        "decision_tree": {**dt_m, "feature_importance": dict(zip(CLF_FEATURES, map(float, tree.feature_importances_)))},
        "random_forest": {**rf_m, "feature_importance": dict(zip(CLF_FEATURES, map(float, rf.feature_importances_)))},
    }
    return tree, rf, res


# ----------------------------------------------------------------------------------
# Task D - k-means
# ----------------------------------------------------------------------------------
def segmentation(df):
    feats = ["Annual_Income", "Spending_Score"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[feats])
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs).inertia_
               for k in range(1, 11)]
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xs)
    sil = float(silhouette_score(Xs, km.labels_))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 11), inertia, "o-", color="#2E75B6")
    ax.axvline(3, color="grey", ls="--", lw=1)
    ax.set(title="Elbow Method", xlabel="Number of clusters (K)", ylabel="Inertia (WCSS)")
    save(fig, "fig5_elbow.png")

    df = df.copy()
    df["Cluster"] = km.labels_
    order = df.groupby("Cluster")["Annual_Income"].mean().sort_values().index.tolist()
    names = dict(zip(order, ["Budget Shoppers", "Mainstream Shoppers", "Premium Spenders"]))
    df["Segment_Label"] = df["Cluster"].map(names)

    centers = scaler.inverse_transform(km.cluster_centers_)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score", hue="Segment_Label",
                    palette="Set1", ax=ax)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=220, c="black", label="Centroids")
    ax.set(title="K-Means Clusters (K=3)", xlabel="Annual Income (Rs thousands)")
    ax.legend()
    save(fig, "fig6_kmeans_clusters.png")

    prof = (df.groupby("Segment_Label")
              .agg(customers=("Customer_ID", "count"),
                   avg_income=("Annual_Income", "mean"),
                   avg_spending=("Spending_Score", "mean"),
                   avg_purchases=("Purchase_Frequency", "mean"),
                   response_rate=("Response_Flag", "mean"))
              .round(2).reindex(["Premium Spenders", "Mainstream Shoppers", "Budget Shoppers"]))
    res = {"k": 3, "silhouette": sil, "inertia": [round(float(i), 1) for i in inertia],
           "profiles": prof.reset_index().to_dict(orient="records")}
    return km, scaler, names, df, res


# ----------------------------------------------------------------------------------
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    df, clean_report = clean(raw)
    df.drop(columns=["Discount_Usage_Enc", "Response_Flag"]).to_csv(CLEAN_PATH, index=False)
    eda(df)

    lin, lin_res = linear_regression(df)
    tree, rf, clf_res = classifiers(df)
    km, scaler, names, df_seg, seg_res = segmentation(df)
    clean_report["class_balance"] = {k: int(v) for k, v in df["Campaign_Response"].value_counts().items()}
    clean_report["corr_adspend_sales"] = float(df["Ad_Spend"].corr(df["Sales"]))

    metrics = {"data_cleaning": clean_report, "linear_regression": lin_res,
               "classification": clf_res, "segmentation": seg_res}
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({"linear_model": lin, "decision_tree": tree, "random_forest": rf,
                 "kmeans": km, "scaler": scaler, "cluster_names": {int(k): v for k, v in names.items()},
                 "clf_features": CLF_FEATURES, "discount_map": DISCOUNT_MAP,
                 "ad_spend_range": [float(df["Ad_Spend"].min()), float(df["Ad_Spend"].max())]},
                MODELS / "brightmart_bundle.joblib")

    # ---- console summary ----
    print("\n=== BrightMart pipeline finished ===")
    print(f"Rows: {clean_report['raw_rows']} raw -> {clean_report['clean_rows']} clean | "
          f"duplicates removed: {clean_report['duplicates_removed']}")
    print(f"Linear Regression: slope={lin_res['slope']:.2f} intercept={lin_res['intercept']:.2f} "
          f"MAE={lin_res['mae']:.2f} R2={lin_res['r2']:.3f} | 30k budget -> {lin_res['prediction_30k']:.1f}")
    for key, label in [("decision_tree", "Decision Tree"), ("random_forest", "Random Forest")]:
        m = clf_res[key]
        print(f"{label:14s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} cm={m['confusion']}")
    print(f"K-Means K=3 silhouette={seg_res['silhouette']:.2f}")
    print(pd.DataFrame(seg_res["profiles"]).to_string(index=False))
    print("\nSaved: cleaned CSV, figures, outputs/metrics.json, models/brightmart_bundle.joblib")


if __name__ == "__main__":
    main()

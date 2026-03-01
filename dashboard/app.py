import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st


DATA_PATH = "data/train_features.csv"
TEST_PATH = "data/test_features.csv"


@st.cache_data
def load_data():
    train = pd.read_csv(DATA_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("models/rf_burnout.pkl"):
        models["rf_burnout"] = joblib.load("models/rf_burnout.pkl")
    if os.path.exists("models/logreg_dropout.pkl"):
        models["logreg_dropout"] = joblib.load("models/logreg_dropout.pkl")
    return models


def add_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for easier filtering and display."""
    d = df.copy()
    d["gender_label"] = d["gender_M"].map({1: "Male", 0: "Female"})
    if "income_level_Low" in d.columns and "income_level_Medium" in d.columns:
        def income(row):
            if row["income_level_Low"] == 1:
                return "Low"
            if row["income_level_Medium"] == 1:
                return "Medium"
            return "High"
        d["income_label"] = d.apply(income, axis=1)
    else:
        d["income_label"] = "All"
    nat_cols = [c for c in d.columns if c.startswith("nationality_")]
    if nat_cols:
        ref = "CN"  # reference level when all dummies are 0
        def nationality(row):
            for c in sorted(nat_cols):
                if row.get(c, 0) == 1:
                    return c.replace("nationality_", "")
            return ref
        d["nationality_label"] = d.apply(nationality, axis=1)
    else:
        d["nationality_label"] = "All"
    return d


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    if filters.get("burnout_levels"):
        out = out[out["burnout_level"].isin(filters["burnout_levels"])]
    if filters.get("dropout_status") is not None and len(filters["dropout_status"]) < 2:
        out = out[out["dropout_status"].isin(filters["dropout_status"])]
    if "age_min" in filters and filters["age_min"] is not None:
        out = out[out["age"] >= filters["age_min"]]
    if "age_max" in filters and filters["age_max"] is not None:
        out = out[out["age"] <= filters["age_max"]]
    if filters.get("gender") and filters["gender"] != "All":
        if filters["gender"] == "Male":
            out = out[out["gender_M"] == 1]
        else:
            out = out[out["gender_M"] == 0]
    if filters.get("income") and filters["income"] != "All":
        out = out[out["income_label"] == filters["income"]]
    if filters.get("nationality") and filters["nationality"] != "All":
        out = out[out["nationality_label"] == filters["nationality"]]
    return out


def main():
    st.set_page_config(
        page_title="Student Burnout & Dropout Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not os.path.exists(DATA_PATH):
        st.error("Data not found. Run the pipeline first (data_generation → preprocessing → feature_engineering).")
        return

    train, test = load_data()
    train = add_filter_columns(train)
    models = load_models()

    # --------------- Sidebar filters ---------------
    st.sidebar.title("Filters")
    st.sidebar.caption("Filter the main dashboard. Clear selections to show all.")

    burnout_opts = ["Low", "Medium", "High"]
    burnout_selected = st.sidebar.multiselect("Burnout level", burnout_opts, default=burnout_opts)
    dropout_opts = [0, 1]
    dropout_selected = st.sidebar.multiselect("Dropout status", [0, 1], default=[0, 1], format_func=lambda x: "Dropped" if x == 1 else "Retained")
    age_min, age_max = int(train["age"].min()), int(train["age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    gender_choice = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])
    income_choice = st.sidebar.selectbox("Income", ["All", "Low", "Medium", "High"])
    nat_opts = ["All"] + sorted(train["nationality_label"].unique().tolist())
    nationality_choice = st.sidebar.selectbox("Nationality", nat_opts)

    filters = {
        "burnout_levels": burnout_selected,
        "dropout_status": dropout_selected,
        "age_min": age_range[0],
        "age_max": age_range[1],
        "gender": gender_choice,
        "income": income_choice,
        "nationality": nationality_choice,
    }
    train_f = apply_filters(train, filters)

    # --------------- Main: single-page dashboard ---------------
    st.title("Student Burnout & Dropout Analytics")
    st.caption(f"Showing **{len(train_f):,}** of **{len(train):,}** students (use sidebar to filter)")

    # --------------- Row 1: KPIs ---------------
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Students", f"{len(train_f):,}")
    with k2:
        dr = train_f["dropout_status"].mean() * 100 if len(train_f) else 0
        st.metric("Dropout rate", f"{dr:.1f}%")
    with k3:
        st.metric("Avg burnout score", f"{train_f['burnout_score'].mean():.1f}" if len(train_f) else "—")
    with k4:
        hb = (train_f["burnout_level"] == "High").mean() * 100 if len(train_f) else 0
        st.metric("High burnout %", f"{hb:.1f}%")
    with k5:
        st.metric("Avg engagement", f"{train_f['engagement_index'].mean():.1f}" if len(train_f) else "—")
    with k6:
        anomaly_path = "data/anomalies.csv"
        if os.path.exists(anomaly_path):
            adf = pd.read_csv(anomaly_path)
            in_f = train_f.index.intersection(adf.index) if "index" in adf.columns else len(adf)
            try:
                n_anom = len(adf) if not train_f.empty else 0
            except Exception:
                n_anom = len(adf)
            st.metric("Anomalies (total)", f"{n_anom}")
        else:
            st.metric("Anomalies", "—")

    st.markdown("---")

    # --------------- Row 2: Burnout distribution + Dropout risk ---------------
    r2a, r2b = st.columns(2)
    with r2a:
        st.subheader("Burnout level distribution")
        if len(train_f) > 0:
            burn_df = train_f["burnout_level"].value_counts().reset_index()
            burn_df.columns = ["burnout_level", "count"]
            c1 = alt.Chart(burn_df).mark_bar().encode(
                x=alt.X("burnout_level:N", sort=["Low", "Medium", "High"], title="Burnout level"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("burnout_level:N", scale=alt.Scale(range=["#2ecc71", "#f39c12", "#e74c3c"])),
                tooltip=["burnout_level", "count"],
            )
            st.altair_chart(c1, use_container_width=True)
        else:
            st.info("No data after filters.")

    with r2b:
        st.subheader("Dropout risk (test set)")
        if "logreg_dropout" in models and len(test) > 0:
            exclude = ("burnout_level", "burnout_level_code", "dropout_status", "time_to_dropout", "gender_label", "income_label", "nationality_label")
            feature_cols = [c for c in test.columns if c not in exclude]
            X_test = test[feature_cols]
            prob = models["logreg_dropout"].predict_proba(X_test)[:, 1]
            df_prob = pd.DataFrame({"dropout_prob": prob, "dropout_status": test["dropout_status"].astype(str)})
            c2 = alt.Chart(df_prob).mark_bar(opacity=0.7).encode(
                x=alt.X("dropout_prob:Q", bin=alt.Bin(maxbins=25), title="Predicted dropout probability"),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("dropout_status:N", title="Actual", scale=alt.Scale(range=["#3498db", "#e74c3c"])),
                tooltip=[alt.Tooltip("dropout_prob:Q", bin=True), "count():Q"],
            )
            st.altair_chart(c2, use_container_width=True)
        else:
            st.warning("Dropout model or test data not available.")

    # --------------- Row 3: Clusters + SHAP ---------------
    r3a, r3b = st.columns(2)
    with r3a:
        st.subheader("Student clusters (engagement vs motivation)")
        clusters_path = "data/clusters.csv"
        if os.path.exists(clusters_path):
            cdf = pd.read_csv(clusters_path)
            if "engagement_index" in cdf.columns and "motivation_proxy" in cdf.columns:
                c3 = alt.Chart(cdf).mark_circle(size=50, opacity=0.7).encode(
                    x=alt.X("engagement_index:Q", title="Engagement index"),
                    y=alt.Y("motivation_proxy:Q", title="Motivation proxy"),
                    color=alt.Color("cluster:N", title="Cluster"),
                    tooltip=["engagement_index", "motivation_proxy", "cluster"],
                ).interactive()
                st.altair_chart(c3, use_container_width=True)
            else:
                st.info("Cluster data missing expected columns.")
        else:
            st.info("Run clustering.py to generate clusters.")

    with r3b:
        st.subheader("Top features for burnout (SHAP)")
        shap_path = "plots/shap_burnout_feature_importance.png"
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True)
        else:
            st.info("Run explainability.py to generate SHAP plot.")

    # --------------- Row 4: Survival + Causal DAG ---------------
    r4a, r4b = st.columns(2)
    with r4a:
        st.subheader("Survival curve (dropout over time)")
        km_path = "plots/km_overall.png"
        if os.path.exists(km_path):
            st.image(km_path, use_container_width=True)
        else:
            st.info("Run survival_analysis.py to generate curve.")

    with r4b:
        st.subheader("Causal structure (DAG)")
        dag_path = "plots/causal_dag.png"
        if os.path.exists(dag_path):
            st.image(dag_path, use_container_width=True)
        else:
            st.info("Run causal_graphs.py to generate DAG.")

    # --------------- Row 5: Engagement & anomaly table ---------------
    st.subheader("Engagement and stress (filtered)")
    if len(train_f) > 0 and "engagement_index" in train_f.columns:
        r5a, r5b = st.columns(2)
        with r5a:
            eng_df = train_f[["engagement_index", "burnout_level"]].copy()
            c_eng = alt.Chart(eng_df).mark_boxplot(size=30).encode(
                x=alt.X("burnout_level:N", sort=["Low", "Medium", "High"], title="Burnout level"),
                y=alt.Y("engagement_index:Q", title="Engagement index"),
                color="burnout_level:N",
            )
            st.altair_chart(c_eng, use_container_width=True)
        with r5b:
            if "sentiment_stress" in train_f.columns:
                stress_df = train_f[["sentiment_stress", "burnout_level"]].copy()
                c_stress = alt.Chart(stress_df).mark_boxplot(size=30).encode(
                    x=alt.X("burnout_level:N", sort=["Low", "Medium", "High"], title="Burnout level"),
                    y=alt.Y("sentiment_stress:Q", title="Sentiment-adjusted stress"),
                    color="burnout_level:N",
                )
                st.altair_chart(c_stress, use_container_width=True)
            else:
                st.write("Sentiment stress not in data.")
    else:
        st.write("No filtered data for engagement/stress charts.")

    st.subheader("Anomalous students (Isolation Forest)")
    if os.path.exists("data/anomalies.csv"):
        adf = pd.read_csv("data/anomalies.csv")
        display_cols = [c for c in adf.columns if c in ["age", "GPA", "attendance", "engagement_index", "burnout_level", "dropout_status", "anomaly_score", "is_anomaly"]]
        if not display_cols:
            display_cols = adf.columns[:10].tolist()
        st.dataframe(adf[display_cols].head(100), use_container_width=True, height=220)
    else:
        st.info("Run anomaly_detection.py to flag anomalies.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Behavioural Analytics for Student Burnout & Dropout Risk")


if __name__ == "__main__":
    main()

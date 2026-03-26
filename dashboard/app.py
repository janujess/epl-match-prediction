import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="EPL Model Monitor", layout="wide")

st.title("EPL Match Prediction — Developer Dashboard")

results_path = Path("models/results_summary.csv")
model_path = Path("models/xgb_weighted.pkl")


st.header("Model Performance")

if results_path.exists():

    results_df = pd.read_csv(results_path)

    st.dataframe(results_df, use_container_width=True)

    best_model = results_df.sort_values("F1_macro", ascending=False).iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best Model", best_model["Model"])
    col2.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    col3.metric("Precision Macro", f"{best_model['Precision_macro']:.4f}")
    col4.metric("F1 Macro", f"{best_model['F1_macro']:.4f}")

    st.subheader("Model Comparison")

    fig, ax = plt.subplots()

    ax.bar(results_df["Model"], results_df["F1_macro"])

    ax.set_ylabel("F1 Macro")
    ax.set_title("Model Comparison")

    plt.xticks(rotation=45)

    st.pyplot(fig)

else:
    st.warning("Run training pipeline first to generate results_summary.csv")


st.header("Feature Importance")

if model_path.exists():

    model = joblib.load(model_path)
    features = joblib.load("models/feature_columns.pkl")

    importances = model.feature_importances_

    fi = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    st.dataframe(fi)

    fig2, ax2 = plt.subplots()

    ax2.barh(fi["feature"], fi["importance"])
    ax2.invert_yaxis()

    st.pyplot(fig2)

else:
    st.warning("Model not found. Train pipeline first.")


st.header("Model Artifacts")

if model_path.exists():

    st.success("Model Loaded Successfully")

    file_info = {
        "Model file": model_path.name,
        "Size (MB)": round(model_path.stat().st_size / 1e6, 2)
    }

    st.json(file_info)

else:
    st.warning("No trained model found.")

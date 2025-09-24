import pandas as pd
import streamlit as st
from utils.churn_utils import analyze_churn

from utils.churn_utils import train_churn_model, plot_churn_distribution, load_churn_predictions

def run_churn_analysis():
    st.title("Churn Analysis")

    uploaded_file = st.file_uploader("Upload churn data CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded file loaded.")
    else:
        try:
            df = pd.read_csv("data/churn_labels.csv")
            st.info("ℹ️ Using default dataset.")
        except FileNotFoundError:
            st.error("❌ No file uploaded and default dataset not found.")
            return

    # --- Make sure 'churn' column is present
    if 'churn' not in df.columns:
        st.error("❌ Dataset must contain a 'churn' column.")
        return

    # --- Now trigger actual analysis below this line
    st.subheader("📊 Churn Distribution")
    fig = plot_churn_distribution(df)
    st.pyplot(fig)

    st.subheader("📈 Training Churn Prediction Model")
    model, report = train_churn_model(df)
    accuracy = report["accuracy"]
    st.success(f"✅ Model trained. Accuracy: {accuracy:.2%}")

    st.subheader("📉 Example Predictions")
    predictions = load_churn_predictions(model, df)
    st.write(predictions.head())

#def run_churn_analysis():
#    st.title("Churn Analysis")
#    uploaded_file = st.file_uploader("Upload churn data CSV", type=["csv"])
#    if uploaded_file:
#        data = pd.read_csv(uploaded_file)
#        st.write("Data Preview:", data.head())
#        churn_metrics = analyze_churn(data)
#        st.write("Churn Metrics:", churn_metrics)

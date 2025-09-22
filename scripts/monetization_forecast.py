import pandas as pd
import streamlit as st
from utils.monetization_forecast_utils import forecast_revenue

def run_monetization_forecast():
    st.title("Monetization Forecast")
    uploaded_file = st.file_uploader("Upload revenue data CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        forecast = forecast_revenue(df)
        st.write("Forecast Results:", forecast)

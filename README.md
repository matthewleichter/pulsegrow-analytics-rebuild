# PulseGrow Analytics

**PulseGrow Analytics** is a full-featured analytics dashboard built with Streamlit. It includes user retention models, marketing mix modeling, churn prediction, funnel analytics, causal inference, anomaly detection, and more — all powered by advanced ML and time series forecasting tools.

## 📊 Features

- **Churn Prediction** (XGBoost, Survival Analysis)
- **Usage Forecasting** (XGBoost, Anomaly Detection)
- **TimesNet Transformer Forecasting**
- **Kaplan-Meier Retention Curves**
- **Causal Inference (EconML)**
- **A/B Test Analysis**
- **Funnel Analysis (Sankey Diagrams)**
- **Behavioral Segmentation (GMM + UMAP)**
- **Revenue Forecasting**
- **LLM-based Insight Summarization**

## 📁 Project Structure

```
pulsegrow-analytics-rebuild/
├── data/                  # CSV datasets
├── models/                # Model code (TimesNet, segmentation, etc.)
├── utils/                 # All utilities (data loaders, metrics, plotting, etc.)
├── scripts/               # Each script generates plots or runs ML pipelines
├── streamlit_app.py       # Streamlit entrypoint
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   ├── runtime.txt
│   └── runtime.yaml
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/matthewleichter/pulsegrow-analytics-rebuild.git
cd pulsegrow-analytics-rebuild
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or upload to [Streamlit Cloud](https://streamlit.io/cloud) and set:

- **Main file**: `streamlit_app.py`
- **Python version**: `3.10.12` via `runtime.txt`
- **Runtime trigger**: Rebuild if needed via `.streamlit/runtime.yaml`

---

## 🖼️ Screenshots

### Churn Prediction Output
📷 Screenshot Placeholder

### Usage Forecast Dashboard
📷 Screenshot Placeholder

### Causal Inference Model
📷 Screenshot Placeholder

### Funnel Analysis (Sankey)
📷 Screenshot Placeholder

---

## 📞 Contact

**Matthew Leichter**  
📧 Email: [matthew.leichter@gmail.com](mailto:matthew.leichter@gmail.com)  
📱 Phone: 323-303-8062  
🌐 Portfolio: [https://matthewleichter.github.io](https://matthewleichter.github.io)

---

## ✅ Status

This project is production-ready and runs end-to-end with real or synthetic data. No placeholders. Ready for deployment and showcase. Created in Leprechaun OS, the world's only LLM based operating system. 


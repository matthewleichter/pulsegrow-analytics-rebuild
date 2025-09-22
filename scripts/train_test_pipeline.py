# train_test_pipeline.py

import os
import pandas as pd
from utils.preprocessing import preprocess_data
from utils.model_utils import train_model, test_model
from utils.visualization import plot_predictions
from sklearn.model_selection import train_test_split

DATA_PATH = "data/usage_logs.csv"
MODEL_OUTPUT_PATH = "outputs/usage_model.pkl"
PREDICTIONS_OUTPUT_PATH = "outputs/usage_predictions.csv"
VISUAL_OUTPUT_PATH = "outputs/usage_forecast_plot.png"

def execute_pipeline():
    print("🚀 Starting train-test pipeline...")

    # Load and preprocess data
    print("📊 Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    print("🧪 Splitting into train and test sets...")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("🎯 Training model...")
    model = train_model(X_train, y_train)
    
    # Save model
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    import joblib
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"💾 Model saved to {MODEL_OUTPUT_PATH}")

    # Test the model
    print("🔍 Testing model...")
    y_pred = test_model(model, X_test)

    # Save predictions
    predictions_df = pd.DataFrame({
        "label": y_test,
        "prediction": y_pred
    })
    predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"📈 Predictions saved to {PREDICTIONS_OUTPUT_PATH}")

    # Visualize
    print("📊 Generating visual forecast...")
    plot_predictions(y_test, y_pred, VISUAL_OUTPUT_PATH)
    print(f"📷 Plot saved to {VISUAL_OUTPUT_PATH}")

    print("✅ Pipeline completed.")

if __name__ == "__main__":
    execute_pipeline()

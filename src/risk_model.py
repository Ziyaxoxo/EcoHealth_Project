import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib

# 1. Load Data
def load_and_prep_data(filepath):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    # Create synthetic 'Risk_Category' if not in dataset
    if 'Risk_Category' not in df.columns:
        df['Risk_Category'] = df['AQI'].apply(lambda x: 'High Risk' if x > 200 else 'Low/Moderate Risk')
    
    return df

# 2. Train Classifier
def train_risk_classifier(df):
    # Features: Pollutants only (Matches your app input)
    features = ['PM2.5', 'PM10', 'NO2', 'SO2']
    
    X = df[features]
    y = df['Risk_Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")
    return model, features

# 3. Anomaly Detection
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05)
    features = ['AQI', 'PM2.5'] 
    df['Anomaly'] = model.fit_predict(df[features])
    return df

if __name__ == "__main__":
    # Ensure this matches your renamed file in data/raw/
    df = load_and_prep_data('data/raw/aqi_health_data.csv')
    
    # Train and Save Model
    print("Training model...")
    model, feature_names = train_risk_classifier(df)
    joblib.dump(model, 'src/risk_model.pkl')
    
    # Run Anomaly Detection
    print("Detecting anomalies...")
    df = detect_anomalies(df)

    # Create folder if it doesn't exist and Save Data
    print("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/tableau_ready_data.csv', index=False)
    
    print("✅ System Training Complete.")
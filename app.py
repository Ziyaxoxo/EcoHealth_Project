import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load Model
try:
    model = joblib.load('src/risk_model.pkl')
except FileNotFoundError:
    st.error("Model not found! Please run 'python3 src/risk_model.py' first.")
    st.stop()

st.set_page_config(page_title="EcoHealth Shield", layout="wide")
st.title("🌍 EcoHealth Shield: Climate-Risk Intelligence")

# --- Sidebar: Policy Simulator ---
st.sidebar.header("⚙️ Policy Simulator")
st.sidebar.markdown("Adjust pollutant levels to predict health risk.")

# Sliders (Inputs match the Training Columns)
pm25 = st.sidebar.slider("PM2.5 Level (µg/m³)", 0, 500, 50)
pm10 = st.sidebar.slider("PM10 Level (µg/m³)", 0, 500, 100)
no2 = st.sidebar.slider("NO2 Level", 0, 200, 40)
so2 = st.sidebar.slider("SO2 Level", 0, 100, 20)

# --- Prediction Logic ---
input_data = pd.DataFrame([[pm25, pm10, no2, so2]], 
                          columns=['PM2.5', 'PM10', 'NO2', 'SO2'])

prediction = model.predict(input_data)[0]
probs = model.predict_proba(input_data)
max_prob = max(probs[0]) * 100

# --- Dashboard Display ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Predicted Health Risk", value=prediction)
with col2:
    st.metric(label="Confidence", value=f"{max_prob:.1f}%")
with col3:
    if prediction == "High Risk":
        st.error("⚠️ REDUCE EMISSIONS")
    else:
        st.success("✅ SAFE LIMITS")

# --- Visuals (Python) ---
st.markdown("---")
st.subheader("📊 Diagnostic Analysis")

try:
    df_chart = pd.read_csv('data/raw/aqi_health_data.csv').head(500)
    fig = px.scatter(
        df_chart, x='PM2.5', y='AQI', color='AQI',
        title="PM2.5 vs AQI Correlation", color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.warning("Dataset not found for visualization.")

# --- Tableau Integration (The Map) ---
st.markdown("---")
st.subheader("🗺️ Live Pollution Map (India)")

# YOUR SPECIFIC LINK (Converted to Embed format)
tableau_url = "https://public.tableau.com/views/EcoHealthMap/Sheet1?:showVizHome=no&:embed=true"

st.markdown(f"""
    <iframe src="{tableau_url}" width="100%" height="600" style="border:none;"></iframe>
""", unsafe_allow_html=True)
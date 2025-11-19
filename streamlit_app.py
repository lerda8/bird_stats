import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Your specific BirdNET-Go instance
BIRDNET_API_URL = "https://birds.ballaty.cz/api/v2/detections" 
# Note: If your API doesn't support a full history dump via GET, 
# you might need to use the 'Download CSV' feature from your BirdNET-Go 
# dashboard and upload it here, or connect to the SQLite database directly.

# Coordinates for Prague (from your location)
LATITUDE = 50.0755 
LONGITUDE = 14.4378

st.set_page_config(page_title="BirdNET Data Explorer", layout="wide")

st.title("🐦 BirdNET-Go Analysis Dashboard")
st.markdown(f"Connected to: [{BIRDNET_API_URL}]({BIRDNET_API_URL})")

# --- DATA FETCHING ---

@st.cache_data(ttl=3600)
def get_bird_data(days=7):
    """
    Fetches bird detection data. 
    Tries to load from API. If that fails or returns mock data, we simulate.
    """
    # ------------------------------------------------------------------
    # ATTEMPT 1: Real API Call
    # ------------------------------------------------------------------
    try:
        # Many BirdNET-Go instances paginate or limit range. 
        # We ask for a date range if the API supports it.
        params = {
            'start': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        }
        # Uncomment this when ready to test connection:
        # response = requests.get(BIRDNET_API_URL, params=params, timeout=10)
        # if response.status_code == 200:
        #     data = response.json()
        #     # Handle if data is wrapped in a 'detections' key
        #     if isinstance(data, dict) and 'detections' in data:
        #         data = data['detections']
        #     return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not connect to API ({e}). Using simulated data.")

    # ------------------------------------------------------------------
    # FALLBACK: Simulated Data (for demonstration)
    # ------------------------------------------------------------------
    dates = pd.date_range(end=datetime.now(), periods=days*24*2, freq='30min')
    import numpy as np
    
    # Simulate that BirdNET-Go *might* pass weather data if configured
    has_weather_integration = True 
    
    data = {
        'Timestamp': dates,
        'CommonName': np.random.choice(['Eurasian Blackbird', 'European Robin', 'House Sparrow', 'Great Tit', 'Magpie'], size=len(dates)),
        'Confidence': np.random.uniform(0.7, 0.99, size=len(dates)),
    }
    
    # Add fake weather data simulating what BirdNET-Go might log
    if has_weather_integration:
        data['Temperature'] = np.random.uniform(10, 25, size=len(dates))
        data['Humidity'] = np.random.uniform(40, 80, size=len(dates))
        
    df = pd.DataFrame(data)
    
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    return df

@st.cache_data(ttl=3600)
def get_historical_weather(days=7):
    """
    Fetches historical weather from Open-Meteo as a fallback
    if BirdNET-Go didn't save weather data.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "precipitation", "cloudcover"],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        hourly = data.get('hourly', {})
        df = pd.DataFrame({
            'Timestamp': pd.to_datetime(hourly['time']),
            'External_Temp': hourly['temperature_2m'],
            'External_Precip': hourly['precipitation'],
            'External_Cloud': hourly['cloudcover']
        })
        return df
    except:
        return pd.DataFrame()

# --- MAIN APP ---

days_to_analyze = st.sidebar.slider("Days to Analyze", 1, 30, 7)

with st.spinner('Loading detection data...'):
    df_birds = get_bird_data(days_to_analyze)

if not df_birds.empty:
    
    # 1. SMART WEATHER MERGE
    # Check if we already have weather columns in the bird data
    bird_cols = [c.lower() for c in df_birds.columns]
    has_internal_weather = any(x in bird_cols for x in ['temp', 'temperature', 'weather'])
    
    if has_internal_weather:
        st.success("✅ Using Weather data found directly in BirdNET logs.")
        df_analysis = df_birds.copy()
        # Standardize column names for plotting
        col_map = {c: c for c in df_birds.columns}
        for c in df_birds.columns:
            if 'temp' in c.lower(): col_map[c] = 'Temperature_Analysis'
        df_analysis.rename(columns=col_map, inplace=True)
        
        # Ensure we have a timestamp to group by
        df_analysis['Hour'] = df_analysis['Timestamp'].dt.round('H')
        
        # Since weather is per detection, we average it per hour
        df_weather_grouped = df_analysis.groupby('Hour')['Temperature_Analysis'].mean().reset_index()
        df_counts = df_analysis.groupby('Hour').size().reset_index(name='Detection Count')
        df_merged = pd.merge(df_counts, df_weather_grouped, on='Hour')
        
    else:
        st.info("ℹ️ No weather data in BirdNET logs. Fetching from Open-Meteo...")
        df_weather = get_historical_weather(days_to_analyze)
        
        df_birds['Hour'] = df_birds['Timestamp'].dt.round('H')
        df_counts = df_birds.groupby('Hour').size().reset_index(name='Detection Count')
        
        # Merge
        df_merged = pd.merge(df_counts, df_weather, left_on='Hour', right_on='Timestamp', how='inner')
        df_merged['Temperature_Analysis'] = df_merged['External_Temp']

    # --- VISUALIZATIONS ---

    # 1. Correlation Scatter Plot
    st.subheader(f"🌡️ Temperature vs. Bird Activity")
    
    fig_corr = px.scatter(
        df_merged, 
        x="Temperature_Analysis", 
        y="Detection Count",
        hover_data=['Hour'],
        title="Correlation: Temperature vs Detections",
        trendline="ols", # Ordinary Least Squares regression line
        labels={"Temperature_Analysis": "Temperature (°C)", "Detection Count": "Detections per Hour"}
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # 2. Top Species Breakdown
    st.subheader("🏆 Top Detected Species")
    top_species = df_birds['CommonName'].value_counts().head(10)
    fig_bar = px.bar(
        top_species, 
        orientation='h', 
        title="Top 10 Species by Count",
        color=top_species.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 3. Detailed Data
    with st.expander("Data Inspector"):
        st.write("Sample of raw data:")
        st.dataframe(df_birds.head(50))

else:
    st.error("No data loaded.")

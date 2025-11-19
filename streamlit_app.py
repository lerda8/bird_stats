import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# --- KONFIGURACE ---
# URL vaší BirdNET-Go instance
BIRDNET_API_URL = "https://birds.ballaty.cz/api/v2/detections" 

# Souřadnice (Praha) - pro počasí
LATITUDE = 50.0755 
LONGITUDE = 14.4378

# Nastavení stránky
st.set_page_config(page_title="BirdNET Analýza", layout="wide")

st.title("🐦 BirdNET-Go Analytický Dashboard")
st.markdown(f"Zdroj dat: [{BIRDNET_API_URL}]({BIRDNET_API_URL})")

# --- NAČÍTÁNÍ DAT ---

@st.cache_data(ttl=3600)
def get_bird_data(days=7):
    """
    Stáhne data o detekcích ptáků.
    Parsuje JSON strukturu: {"data": [{"beginTime": "...", "commonName": "...", ...}]}
    """
    # ------------------------------------------------------------------
    # 1. Pokus o stažení z API
    # ------------------------------------------------------------------
    try:
        params = {
            'start': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        }
        
        headers = {'User-Agent': 'StreamlitBirdNET/1.0'}
        
        # Timeout nastaven na 15 sekund
        response = requests.get(BIRDNET_API_URL, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            json_resp = response.json()
            
            if isinstance(json_resp, dict) and 'data' in json_resp:
                raw_data = json_resp['data']
                df = pd.DataFrame(raw_data)
                
                # Přejmenování sloupců pro interní logiku
                rename_map = {
                    'beginTime': 'Timestamp',
                    'commonName': 'CommonName',
                    'confidence': 'Confidence'
                }
                df = df.rename(columns=rename_map)
                
                # Převod času na datetime objekt
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601')
                
                # Převod spolehlivosti na číslo
                if 'Confidence' in df.columns:
                    df['Confidence'] = pd.to_numeric(df['Confidence'])
                
                return df
            else:
                st.warning("API připojeno, ale v odpovědi chybí klíč 'data'.")
        else:
            st.warning(f"Chyba API, status kód: {response.status_code}")

    except Exception as e:
        st.warning(f"Nelze se připojit k API ({e}). Používám simulovaná data pro ukázku.")

    # ------------------------------------------------------------------
    # 2. Simulovaná data (pokud API selže)
    # ------------------------------------------------------------------
    dates = pd.date_range(end=datetime.now(), periods=days*24*2, freq='30min')
    
    data = {
        'Timestamp': dates,
        'CommonName': np.random.choice(['Sýkora modřinka', 'Vrabec polní', 'Kos černý', 'Sýkora koňadra', 'Strakapoud velký'], size=len(dates)),
        'Confidence': np.random.uniform(0.7, 0.99, size=len(dates)),
    }
    
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

@st.cache_data(ttl=3600)
def get_historical_weather(days=7):
    """
    Stáhne historické počasí z Open-Meteo (fallback).
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

# --- HLAVNÍ APLIKACE ---

# Boční panel
days_to_analyze = st.sidebar.slider("Počet dní k analýze", 1, 30, 7)

with st.spinner('Načítám data o detekcích...'):
    df_birds = get_bird_data(days_to_analyze)

if not df_birds.empty:
    
    # 1. SLOUČENÍ S POČASÍM
    bird_cols = [c.lower() for c in df_birds.columns]
    has_internal_weather = any(x in bird_cols for x in ['temp', 'temperature', 'weather'])
    
    if has_internal_weather:
        st.success("✅ Používám data o počasí přímo z logů BirdNET.")
        df_analysis = df_birds.copy()
        # Standardizace názvů
        col_map = {c: c for c in df_birds.columns}
        for c in df_birds.columns:
            if 'temp' in c.lower(): col_map[c] = 'Temperature_Analysis'
        df_analysis.rename(columns=col_map, inplace=True)
        
        df_analysis['Hour'] = df_analysis['Timestamp'].dt.round('h')
        df_weather_grouped = df_analysis.groupby('Hour')['Temperature_Analysis'].mean().reset_index()
        df_counts = df_analysis.groupby('Hour').size().reset_index(name='Detection Count')
        df_merged = pd.merge(df_counts, df_weather_grouped, on='Hour')
        
    else:
        # Tento blok proběhne, pokud v JSONu není počasí (váš případ)
        st.info("ℹ️ V logu chybí počasí. Stahuji data z Open-Meteo...")
        df_weather = get_historical_weather(days_to_analyze)
        
        if not df_weather.empty:
            df_birds['Hour'] = df_birds['Timestamp'].dt.round('h')
            df_counts = df_birds.groupby('Hour').size().reset_index(name='Detection Count')
            
            # Sloučení podle hodiny
            df_merged = pd.merge(df_counts, df_weather, left_on='Hour', right_on='Timestamp', how='inner')
            df_merged['Temperature_Analysis'] = df_merged['External_Temp']
        else:
            st.error("Nepodařilo se stáhnout data o počasí.")
            df_merged = pd.DataFrame()

    # --- VIZUALIZACE ---
    
    if not df_merged.empty:
        # 1. Korelace (Scatter Plot)
        st.subheader(f"🌡️ Závislost aktivity na teplotě")
        
        fig_corr = px.scatter(
            df_merged, 
            x="Temperature_Analysis", 
            y="Detection Count",
            hover_data=['Hour'],
            title="Korelace: Teplota vs Počet detekcí",
            trendline="ols", 
            labels={
                "Temperature_Analysis": "Teplota (°C)", 
                "Detection Count": "Počet detekcí za hodinu"
            }
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # 2. Časová osa (Timeline)
        st.subheader("📅 Aktivita v čase")
        fig_timeline = go.Figure()
        
        # Počty ptáků
        fig_timeline.add_trace(go.Bar(
            x=df_merged['Hour'], 
            y=df_merged['Detection Count'], 
            name='Počet detekcí',
            marker_color='#1f77b4'
        ))
        
        # Čára teploty
        fig_timeline.add_trace(go.Scatter(
            x=df_merged['Hour'], 
            y=df_merged['Temperature_Analysis'], 
            name='Teplota (°C)',
            yaxis='y2',
            line=dict(color='#ff7f0e', width=3)
        ))

        fig_timeline.update_layout(
            title="Detekce a teplota v průběhu času",
            yaxis=dict(title="Počet detekcí"),
            yaxis2=dict(title="Teplota (°C)", overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # 3. Nejčastější druhy
    st.subheader("🏆 Nejčastěji detekované druhy")
    if 'CommonName' in df_birds.columns:
        top_species = df_birds['CommonName'].value_counts().head(10)
        fig_bar = px.bar(
            top_species, 
            orientation='h', 
            title="Top 10 druhů podle počtu detekcí",
            labels={"index": "Druh", "value": "Počet"},
            color=top_species.values,
            color_continuous_scale='Viridis'
        )
        # Skrytí legendy barev, pokud není potřeba
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 4. Tabulka dat
    with st.expander("🔍 Prohlížeč detailních dat"):
        st.write("Ukázka stažených dat (prvních 50 záznamů):")
        st.dataframe(df_birds.head(50))

else:
    st.error("Žádná data nebyla načtena. Zkontrolujte API.")

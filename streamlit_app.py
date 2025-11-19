import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
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
def get_bird_data(start_date, end_date):
    """
    Stáhne data o detekcích ptáků pro zadaný rozsah dat.
    """
    # ------------------------------------------------------------------
    # 1. Pokus o stažení z API
    # ------------------------------------------------------------------
    try:
        params = {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
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
                
                # --- OPRAVA TIMEZONE ---
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
                    df['Timestamp'] = df['Timestamp'].dt.tz_convert('Europe/Prague')
                    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
                
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
    # 2. Simulovaná data (Fallback)
    # ------------------------------------------------------------------
    # Vygenerujeme data pro zadaný rozsah
    delta = end_date - start_date
    days = delta.days + 1
    dates = pd.date_range(start=start_date, periods=days*24*2, freq='30min')
    
    data = {
        'Timestamp': dates,
        'CommonName': np.random.choice(['Sýkora modřinka', 'Vrabec polní', 'Kos černý', 'Sýkora koňadra', 'Strakapoud velký'], size=len(dates)),
        'Confidence': np.random.uniform(0.7, 0.99, size=len(dates)),
    }
    
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

@st.cache_data(ttl=3600)
def get_historical_weather(start_date, end_date):
    """
    Stáhne historické počasí z Open-Meteo pro zadaný rozsah.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
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

# 1. VÝBĚR DATA (NOVÉ)
st.sidebar.header("Filtrování")
today = datetime.now().date()
default_start = today - timedelta(days=7)

# Widget pro výběr rozsahu dat
date_range = st.sidebar.date_input(
    "Vyberte časové období",
    value=(default_start, today),
    max_value=today
)

# Ověření, že máme start i konec data
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    
    with st.spinner(f'Načítám data od {start_d} do {end_d}...'):
        df_birds = get_bird_data(start_d, end_d)

    if not df_birds.empty:
        # Filtrujeme dataframe ještě lokálně pro jistotu (pokud by API vrátilo víc)
        mask = (df_birds['Timestamp'].dt.date >= start_d) & (df_birds['Timestamp'].dt.date <= end_d)
        df_birds = df_birds.loc[mask]
        
        # --- ZPRACOVÁNÍ POČASÍ ---
        bird_cols = [c.lower() for c in df_birds.columns]
        has_internal_weather = any(x in bird_cols for x in ['temp', 'temperature', 'weather'])
        
        if has_internal_weather:
            df_analysis = df_birds.copy()
            col_map = {c: c for c in df_birds.columns}
            for c in df_birds.columns:
                if 'temp' in c.lower(): col_map[c] = 'Temperature_Analysis'
            df_analysis.rename(columns=col_map, inplace=True)
            
            df_analysis['Hour'] = df_analysis['Timestamp'].dt.floor('h')
            df_weather_grouped = df_analysis.groupby('Hour')['Temperature_Analysis'].mean().reset_index()
            df_counts = df_analysis.groupby('Hour').size().reset_index(name='Detection Count')
            df_merged = pd.merge(df_counts, df_weather_grouped, on='Hour')
            
        else:
            # Stahujeme počasí pro zvolený rozsah
            df_weather = get_historical_weather(start_d, end_d)
            
            if not df_weather.empty:
                if df_weather['Timestamp'].dt.tz is not None:
                    df_weather['Timestamp'] = df_weather['Timestamp'].dt.tz_localize(None)

                df_birds['Hour'] = df_birds['Timestamp'].dt.floor('h')
                df_counts = df_birds.groupby('Hour').size().reset_index(name='Detection Count')
                
                df_merged = pd.merge(df_counts, df_weather, left_on='Hour', right_on='Timestamp', how='inner')
                df_merged['Temperature_Analysis'] = df_merged['External_Temp']
            else:
                df_merged = pd.DataFrame()

        # --- VIZUALIZACE (NOVÁ) ---
        
        if not df_merged.empty:
            # 1. Hlavní graf: Kombinace Sloupců (Ptáci) a Čáry (Teplota)
            st.subheader(f"🌡️ Vztah mezi počtem ptáků a teplotou")
            
            fig_combo = go.Figure()
            
            # Sloupce: Počet ptáků
            fig_combo.add_trace(go.Bar(
                x=df_merged['Hour'],
                y=df_merged['Detection Count'],
                name='Počet ptáků',
                marker_color='rgba(55, 128, 191, 0.7)', # Modrá s průhledností
                yaxis='y'
            ))
            
            # Čára: Teplota
            fig_combo.add_trace(go.Scatter(
                x=df_merged['Hour'],
                y=df_merged['Temperature_Analysis'],
                name='Teplota (°C)',
                mode='lines', # Pouze čára, bez bodů
                line=dict(color='firebrick', width=3),
                yaxis='y2' # Mapování na druhou osu Y
            ))

            # Nastavení layoutu pro dvě osy
            fig_combo.update_layout(
                title="Vývoj v čase: Detekce vs. Teplota",
                xaxis=dict(title="Čas"),
                yaxis=dict(
                    title="Počet detekcí",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")
                ),
                yaxis2=dict(
                    title="Teplota (°C)",
                    titlefont=dict(color="firebrick"),
                    tickfont=dict(color="firebrick"),
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0, y=1.1, orientation='h'),
                hovermode='x unified' # Společný tooltip pro obě hodnoty
            )
            
            st.plotly_chart(fig_combo, use_container_width=True)

            # 2. Korelační graf (Scatter) - ponecháváme jako doplňkový
            with st.expander("Zobrazit detailní korelaci (Scatter Plot)"):
                fig_corr = px.scatter(
                    df_merged, 
                    x="Temperature_Analysis", 
                    y="Detection Count",
                    title="Scatter Plot: Teplota vs Detekce",
                    trendline="ols",
                    labels={"Temperature_Analysis": "Teplota", "Detection Count": "Počet detekcí"}
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # 3. Top Druhy (Beze změny)
        st.subheader("🏆 Statistiky druhů")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'CommonName' in df_birds.columns:
                top_species = df_birds['CommonName'].value_counts().head(15)
                fig_bar = px.bar(
                    top_species, 
                    orientation='h', 
                    title="Nejčastější druhy",
                    labels={"index": "Druh", "value": "Počet"},
                    color=top_species.values,
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.metric("Celkem detekcí", len(df_birds))
            st.metric("Unikátních druhů", df_birds['CommonName'].nunique())
            if not df_merged.empty:
                st.metric("Průměrná teplota", f"{df_merged['Temperature_Analysis'].mean():.1f} °C")

        # 4. Tabulka
        with st.expander("🔍 Prohlížeč detailních dat"):
            st.dataframe(df_birds)

    else:
        st.info("V tomto časovém rozmezí nebyla nalezena žádná data.")

else:
    st.info("Pro zobrazení dat prosím vyberte počáteční i koncové datum v levém menu.")

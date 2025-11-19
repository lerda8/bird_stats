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
        # Převedení datumů na řetězce ve formátu YYYY-MM-DD
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        params = {
            'start': start_str,
            'end': end_str,
            # ZVÝŠENÍ LIMITU: API pravděpodobně omezuje výsledky na 100.
            # Tímto parametrem zajistíme, že dostaneme více dat.
            'limit': 10000 
        }
        
        headers = {'User-Agent': 'StreamlitBirdNET/1.0'}
        
        st.info(f"Stahuji data z API pro rozsah: {start_str} až {end_str}")
        
        response = requests.get(BIRDNET_API_URL, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            json_resp = response.json()
            
            if isinstance(json_resp, dict) and 'data' in json_resp:
                raw_data = json_resp['data']
                
                if not raw_data:
                    st.warning("API je v pořádku, ale pro zvolený rozsah nevrátilo žádné detekce.")
                    return pd.DataFrame() # Vrací prázdný DataFrame
                
                df = pd.DataFrame(raw_data)
                
                # Přejmenování sloupců pro interní logiku a přehlednost
                rename_map = {
                    'beginTime': 'Timestamp',
                    'commonName': 'CommonName',
                    'scientificName': 'ScientificName', # Nově přidáno
                    'source': 'Source', # Nově přidáno
                    'confidence': 'Confidence'
                }
                df = df.rename(columns=rename_map)
                
                # --- OPRAVA TIMEZONE ---
                if 'Timestamp' in df.columns:
                    # 1. Načíst jako UTC (protože API vrací 'Z' na konci)
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
                    # 2. Převést na čas v Praze a zrušit timezónu
                    df['Timestamp'] = df['Timestamp'].dt.tz_convert('Europe/Prague').dt.tz_localize(None)
                
                if 'Confidence' in df.columns:
                    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
                
                return df
            else:
                st.error("API připojeno, ale v odpovědi chybí klíč 'data' nebo je formát neočekávaný.")
                return pd.DataFrame()
        else:
            st.error(f"Chyba API, status kód: {response.status_code}. Nelze stáhnout data.")
            return pd.DataFrame()

    except requests.exceptions.Timeout:
        st.error("Vypršel časový limit při pokusu o připojení k API. Používám simulovaná data pro ukázku.")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Chyba připojení k API ({e}). Používám simulovaná data pro ukázku.")

    # ------------------------------------------------------------------
    # 2. Simulovaná data (Fallback)
    # ------------------------------------------------------------------
    # Používáme simulovaná data jen jako poslední záchranu, pokud API selže.
    delta = end_date - start_date
    days = delta.days + 1
    dates = pd.date_range(start=start_date, periods=days*24*2, freq='30min')
    
    data = {
        'Timestamp': dates,
        'CommonName': np.random.choice(['Sýkora modřinka', 'Vrabec polní', 'Kos černý', 'Sýkora koňadra', 'Strakapoud velký'], size=len(dates)),
        'ScientificName': np.random.choice(['Cyanistes caeruleus', 'Passer montanus', 'Turdus merula'], size=len(dates)),
        'Source': np.random.choice(['Mic1', 'Mic2', 'RTSP'], size=len(dates)),
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

# 1. VÝBĚR DATA
st.sidebar.header("Filtrování")
today = datetime.now().date()
default_start = today - timedelta(days=7)

# Widget pro výběr rozsahu dat
date_range = st.sidebar.date_input(
    "Vyberte časové období",
    value=(default_start, today),
    max_value=today
)

# Ověření a nastavení datumu
start_d = None
end_d = None

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d = date_range[0]
    end_d = date_range[1]
elif isinstance(date_range, date):
    # Pokud je vybráno jen jedno datum, bereme ho jako START a Konec je DNES (oprava)
    start_d = date_range
    end_d = today
elif isinstance(date_range, list) and len(date_range) == 2:
    start_d = date_range[0]
    end_d = date_range[1]

# --- ZPRACOVÁNÍ DAT ---

if start_d and end_d:
    
    with st.spinner(f'Načítám a zpracovávám data od {start_d} do {end_d}...'):
        df_birds = get_bird_data(start_d, end_d)

    if not df_birds.empty:
        # Filtrujeme dataframe ještě lokálně pro jistotu (pokud by API vrátilo víc)
        mask = (df_birds['Timestamp'].dt.date >= start_d) & (df_birds['Timestamp'].dt.date <= end_d)
        df_birds = df_birds.loc[mask]
        
        # --- ZPRACOVÁNÍ POČASÍ ---
        
        # Zkontrolujeme, jestli API data obsahují sloupce s počasím (např. 'temp', 'temperature')
        bird_cols = [c.lower() for c in df_birds.columns]
        has_internal_weather = any(x in bird_cols for x in ['temp', 'temperature', 'weather'])
        
        if has_internal_weather:
            st.success("✅ Používám data o počasí přímo z logů BirdNET.")
            df_analysis = df_birds.copy()
            col_map = {c: c for c in df_birds.columns}
            for c in df_birds.columns:
                if 'temp' in c.lower(): col_map[c] = 'Temperature_Analysis'
            df_analysis.rename(columns=col_map, inplace=True)
            
            df_analysis['Hour'] = df_analysis['Timestamp'].dt.floor('h')
            df_weather_grouped = df_analysis.groupby('Hour')['Temperature_Analysis'].mean().reset_index()
            df_counts = df_analysis.groupby('Hour').size().reset_index(name='Detection Count')
            
            # OPRAVA: Použít LEFT MERGE, aby se zachovaly všechny řádky detekcí
            df_merged = pd.merge(df_counts, df_weather_grouped, on='Hour', how='left')
            
        else:
            st.info("ℹ️ V logu chybí počasí. Stahuji historická data z Open-Meteo...")
            df_weather = get_historical_weather(start_d, end_d)
            
            if not df_weather.empty:
                df_birds['Hour'] = df_birds['Timestamp'].dt.floor('h')
                df_counts = df_birds.groupby('Hour').size().reset_index(name='Detection Count')
                
                # Sloučení dat
                # OPRAVA: Použít LEFT MERGE, aby se zachovaly všechny řádky detekcí
                df_merged = pd.merge(df_counts, df_weather, left_on='Hour', right_on='Timestamp', how='left')
                df_merged['Temperature_Analysis'] = df_merged['External_Temp']
            else:
                st.error("Nepodařilo se stáhnout data o počasí.")
                df_merged = pd.DataFrame()

        # --- VIZUALIZACE ---
        
        if not df_merged.empty:
            # 1. Hlavní graf: Kombinace Sloupců (Ptáci) a Čáry (Teplota)
            st.subheader(f"🌡️ Vztah mezi počtem ptáků a teplotou ({start_d} - {end_d})")
            
            fig_combo = go.Figure()
            
            # Sloupce: Počet ptáků
            fig_combo.add_trace(go.Bar(
                x=df_merged['Hour'],
                y=df_merged['Detection Count'],
                name='Počet ptáků',
                marker_color='rgba(55, 128, 191, 0.8)', # Modrá s průhledností
                yaxis='y'
            ))
            
            # Čára: Teplota
            fig_combo.add_trace(go.Scatter(
                x=df_merged['Hour'],
                y=df_merged['Temperature_Analysis'],
                name='Teplota (°C)',
                mode='lines',
                line=dict(color='firebrick', width=3),
                yaxis='y2'
            ))

            # Nastavení layoutu pro dvě osy
            fig_combo.update_layout(
                title="Vývoj v čase: Detekce vs. Teplota",
                xaxis=dict(title="Čas"),
                yaxis=dict(
                    title=dict(
                        text="Počet detekcí",
                        font=dict(color="#1f77b4")
                    ),
                    tickfont=dict(color="#1f77b4")
                ),
                yaxis2=dict(
                    title=dict(
                        text="Teplota (°C)",
                        font=dict(color="firebrick")
                    ),
                    tickfont=dict(color="firebrick"),
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0, y=1.1, orientation='h'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_combo, use_container_width=True)

            # 2. Korelační graf (Scatter) - doplňkový
            with st.expander("Zobrazit detailní korelaci (Scatter Plot)"):
                fig_corr = px.scatter(
                    df_merged.dropna(subset=['Temperature_Analysis']), # Odstranit NaN pro trendline
                    x="Temperature_Analysis", 
                    y="Detection Count",
                    title="Scatter Plot: Teplota vs Detekce",
                    trendline="ols",
                    labels={"Temperature_Analysis": "Teplota", "Detection Count": "Počet detekcí"}
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # 3. Top Druhy
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
                # Použít jen data s počasím pro průměr
                temp_avg = df_merged['Temperature_Analysis'].dropna().mean()
                if not pd.isna(temp_avg):
                    st.metric("Průměrná teplota", f"{temp_avg:.1f} °C")

        # 4. Tabulka
        with st.expander("🔍 Prohlížeč detailních dat"):
            # Zobrazujeme více relevantních sloupců
            st.dataframe(df_birds[['Timestamp', 'CommonName', 'ScientificName', 'Confidence', 'Source']].sort_values('Timestamp', ascending=False))

    else:
        st.info(f"V tomto časovém rozmezí ({start_d} - {end_d}) nebyla nalezena žádná data.")

else:
    st.info("Pro zobrazení dat prosím vyberte počáteční i koncové datum v levém menu.")

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# ==============================
# 1. Page Configuration
# ==============================
st.set_page_config(
    page_title="Backyard Bird Analytics",
    page_icon="🦜",
    layout="wide"
)

# ==============================
# 2. Data Loading
# ==============================
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect("birds.db")
        df = pd.read_sql_query("SELECT * FROM detections", conn)
        conn.close()
    except Exception as e:
        st.error(f"❌ Could not load data from database: {e}")
        return pd.DataFrame()

    if not {"Date", "Time", "Com_Name", "Sci_Name", "Confidence"}.issubset(df.columns):
        st.error("❌ Missing required columns in the database.")
        return pd.DataFrame()

    # Process time columns
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"])
    df["Hour"] = df["DateTime"].dt.hour
    df["DayName"] = df["DateTime"].dt.day_name()
    df["Week"] = df["DateTime"].dt.isocalendar().week.astype(int)
    return df


# ==============================
# 3. Main App
# ==============================
df = load_data()

st.title("🦜 Backyard Bird Activity Dashboard")
st.markdown("Visualize bird activity, trends, and detection confidence from your backyard.")

if df.empty:
    st.stop()

# ==============================
# 4. Sidebar Filters
# ==============================
st.sidebar.header("🔍 Filters")

min_date, max_date = df["Date"].min(), df["Date"].max()

date_range = st.sidebar.date_input(
    "Date Range", (min_date, max_date), min_value=min_date, max_value=max_date
)
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.05)
species_list = sorted(df["Com_Name"].unique())
selected_species = st.sidebar.multiselect("Filter by Species", species_list)

# Apply filters
filtered = df.copy()
if len(date_range) == 2:
    filtered = filtered[(filtered["Date"] >= date_range[0]) & (filtered["Date"] <= date_range[1])]
filtered = filtered[filtered["Confidence"] >= min_conf]
if selected_species:
    filtered = filtered[filtered["Com_Name"].isin(selected_species)]

if filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ==============================
# 5. Navigation
# ==============================
page = st.sidebar.radio("Select View", ["Overview", "Species Insights"])

# ==============================
# 6. Overview Page
# ==============================
if page == "Overview":
    st.header("📈 General Activity Trends")

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections", len(filtered))
    col2.metric("Unique Species", filtered["Com_Name"].nunique())
    col3.metric("Avg Confidence", f"{filtered['Confidence'].mean():.2f}")
    col4.metric("Days Recorded", filtered["Date"].nunique())

    st.divider()

    # --- Daily Detections ---
    st.subheader("📅 Daily Detection Trend")
    daily_counts = filtered.groupby("Date").size()
    st.line_chart(daily_counts, use_container_width=True)
    st.caption("Number of detections per day")

    # --- Hourly and Weekly Activity ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("⏰ Hourly Activity")
        hourly = filtered.groupby("Hour").size().reindex(range(24), fill_value=0)
        st.bar_chart(hourly, use_container_width=True)
        st.caption("Activity pattern throughout the day")

    with col_b:
        st.subheader("📆 Day of Week Activity")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly = filtered["DayName"].value_counts().reindex(day_order, fill_value=0)
        st.area_chart(weekly, use_container_width=True)
        st.caption("Detections by day of week")

    st.divider()

    # --- Confidence Distribution ---
    st.subheader("🎯 Confidence Score Distribution")
    conf_bins = pd.cut(filtered["Confidence"], bins=20)
    conf_counts = conf_bins.value_counts().sort_index()
    conf_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in conf_counts.index]
    conf_df = pd.DataFrame({"Confidence Range": conf_labels, "Count": conf_counts.values}).set_index("Confidence Range")
    st.bar_chart(conf_df, use_container_width=True)
    st.caption("Distribution of detection confidence levels")

# ==============================
# 7. Species Insights Page
# ==============================
else:
    st.header("🐦 Species Insights")

    # --- Top Species ---
    st.subheader("🏆 Most Frequent Visitors")
    top_species = filtered["Com_Name"].value_counts().head(10).sort_values(ascending=True)
    st.bar_chart(top_species, use_container_width=True)
    st.caption("Top 10 most frequently detected species")

    # --- Species Statistics Table ---
    st.subheader("📊 Species Statistics")
    stats = (
        filtered.groupby("Com_Name")
        .agg(
            Count=("Com_Name", "count"),
            Avg_Confidence=("Confidence", "mean"),
            Min_Confidence=("Confidence", "min"),
            Max_Confidence=("Confidence", "max"),
        )
        .round(2)
        .sort_values("Count", ascending=False)
    )
    st.dataframe(stats, use_container_width=True, height=400)
    st.caption("Aggregated detection statistics per species")

    st.divider()

    # --- Raw Data Table ---
    st.subheader("📋 Raw Detection Data")
    display_cols = ["Date", "Time", "Com_Name", "Sci_Name", "Confidence", "DayName", "Hour"]
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)

    # --- Download CSV ---
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"bird_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# ==============================
# End
# ==============================
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Bird Analytics Dashboard")

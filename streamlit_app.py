import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
from pathlib import Path

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
DATABASE_FILE = "birds.duckdb"
BOOL_COLUMNS = [
    "verified",
    "locked",
    "isNewSpecies",
    "isNewThisYear",
    "isNewThisSeason",
]
NUMERIC_COLUMNS = ["daysSinceFirstSeen", "daysThisYear", "daysThisSeason"]


@st.cache_data
def load_data(db_file: str = DATABASE_FILE):
    db_path = Path(db_file)
    if not db_path.exists():
        st.error(f"❌ Database file '{db_file}' not found in project directory.")
        return pd.DataFrame()

    try:
        with duckdb.connect(database=str(db_path), read_only=True) as conn:
            df = conn.execute("SELECT * FROM detections").fetchdf()
    except Exception as e:
        st.error(f"❌ Could not load data from DuckDB database: {e}")
        return pd.DataFrame()

    column_mapping = {
        "date": "Date",
        "time": "Time",
        "commonName": "Com_Name",
        "scientificName": "Sci_Name",
        "confidence": "Confidence",
    }
    df = df.rename(columns=column_mapping)

    required_columns = set(column_mapping.values())
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        st.error(f"❌ Missing required columns in the database: {missing}")
        return pd.DataFrame()

    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Confidence"])

    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
            )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    date_str = df["Date"].astype(str)
    df["DateTime"] = pd.to_datetime(date_str + " " + df["Time"].astype(str), errors="coerce")
    df = df.dropna(subset=["DateTime"])

    df["Date"] = df["DateTime"].dt.date
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

    species_summary = (
        filtered.groupby("Com_Name")
        .agg(
            Count=("Com_Name", "count"),
            Avg_Confidence=("Confidence", "mean"),
            Min_Confidence=("Confidence", "min"),
            Max_Confidence=("Confidence", "max"),
            First_Seen=("Date", "min"),
            Last_Seen=("Date", "max"),
        )
        .round({"Avg_Confidence": 2, "Min_Confidence": 2, "Max_Confidence": 2})
        .sort_values("Count", ascending=False)
    )
    species_summary["Active_Days"] = (
        species_summary["Last_Seen"] - species_summary["First_Seen"]
    ).dt.days + 1

    top_species_name = species_summary.index[0]
    top_species_count = int(species_summary.iloc[0]["Count"])
    total_verified = int(filtered["verified"].sum()) if "verified" in filtered.columns else None
    new_species_count = int(filtered["isNewSpecies"].sum()) if "isNewSpecies" in filtered.columns else 0
    new_season_count = (
        int(filtered["isNewThisSeason"].sum()) if "isNewThisSeason" in filtered.columns else 0
    )

    st.subheader("Snapshot")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Top Visitor", top_species_name, f"{top_species_count} detections")
    metric_cols[1].metric("Unique Species", species_summary.shape[0])
    if total_verified is not None:
        metric_cols[2].metric("Verified Detections", total_verified)
    else:
        metric_cols[2].metric("Avg Confidence", f"{filtered['Confidence'].mean():.2f}")
    metric_cols[3].metric("New This Season", new_season_count or new_species_count)

    st.divider()

    tab_leaderboard, tab_focus, tab_discoveries = st.tabs(
        ["Leaderboard", "Species Focus", "Discoveries"]
    )

    with tab_leaderboard:
        st.subheader("🏆 Leaderboard")
        top_n = species_summary.head(15).iloc[::-1]
        st.bar_chart(top_n["Count"], use_container_width=True)
        st.caption("Detection counts for the top species (descending order).")

        leaderboard_df = (
            species_summary.reset_index()
            .rename(
                columns={
                    "Com_Name": "Species",
                    "Count": "Detections",
                    "Avg_Confidence": "Avg Confidence",
                    "First_Seen": "First Seen",
                    "Last_Seen": "Last Seen",
                    "Active_Days": "Active Days",
                }
            )
        )
        st.dataframe(leaderboard_df, use_container_width=True, height=400)

    with tab_focus:
        st.subheader("🔬 Species Deep Dive")
        species_option = st.selectbox("Choose a species to explore", species_summary.index.tolist())
        focus_df = filtered[filtered["Com_Name"] == species_option]

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Detections", len(focus_df))
        col_b.metric("Avg Confidence", f"{focus_df['Confidence'].mean():.2f}")
        first_seen = focus_df["Date"].min()
        last_seen = focus_df["Date"].max()
        span = (pd.to_datetime(last_seen) - pd.to_datetime(first_seen)).days + 1
        col_c.metric("Active Days", span)
        if "verified" in focus_df.columns:
            verified_rate = focus_df["verified"].mean() * 100
            col_d.metric("Verified %", f"{verified_rate:.0f}%")
        else:
            morning_rate = (focus_df["Hour"].between(5, 11).mean() * 100)
            col_d.metric("Morning Activity", f"{morning_rate:.0f}%")

        st.markdown("**Daily detections**")
        focus_daily = focus_df.groupby("Date").size()
        st.line_chart(focus_daily, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Hourly pattern**")
            focus_hourly = focus_df.groupby("Hour").size().reindex(range(24), fill_value=0)
            st.bar_chart(focus_hourly, use_container_width=True)
        with col_right:
            st.markdown("**Confidence distribution**")
            focus_conf = focus_df["Confidence"]
            conf_hist = pd.DataFrame({"Confidence": focus_conf}).sort_values("Confidence")
            st.area_chart(conf_hist.set_index("Confidence"), use_container_width=True)

        st.markdown("**Latest detections**")
        recent_cols = ["Date", "Time", "Confidence", "DayName", "Hour", "source"]
        existing_cols = [c for c in recent_cols if c in focus_df.columns]
        st.dataframe(
            focus_df.sort_values("DateTime", ascending=False)[existing_cols].head(25),
            use_container_width=True,
            height=350,
        )

        csv_focus = focus_df.to_csv(index=False)
        st.download_button(
            f"📥 Download {species_option} data",
            data=csv_focus,
            file_name=f"{species_option.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with tab_discoveries:
        st.subheader("🆕 Discoveries & Seasonality")
        if "isNewSpecies" in filtered.columns and filtered["isNewSpecies"].any():
            new_species_df = filtered[filtered["isNewSpecies"]]
            col_x, col_y, col_z = st.columns(3)
            col_x.metric("Lifetime new species", new_species_df["Com_Name"].nunique())
            if "daysSinceFirstSeen" in new_species_df.columns:
                col_y.metric(
                    "Avg days since first seen",
                    f"{new_species_df['daysSinceFirstSeen'].mean():.0f}",
                )
            if "isNewThisYear" in new_species_df.columns:
                col_z.metric(
                    "New this year",
                    int(new_species_df["isNewThisYear"].sum()),
                )

            new_species_table = (
                new_species_df.groupby("Com_Name")
                .agg(
                    First_Seen=("Date", "min"),
                    Last_Seen=("Date", "max"),
                    Total_Detections=("Com_Name", "count"),
                )
                .sort_values("First_Seen")
            )
            st.dataframe(new_species_table, use_container_width=True, height=350)
        else:
            st.info("No 'new species' flags were detected in the selected range.")

        if "currentSeason" in filtered.columns:
            seasonal = filtered.groupby(["currentSeason", "Com_Name"]).size().reset_index(name="Count")
            season_pivot = seasonal.pivot_table(
                index="currentSeason", columns="Com_Name", values="Count", fill_value=0
            )
            st.markdown("**Seasonal activity heatmap**")
            st.dataframe(season_pivot, use_container_width=True, height=350)
        else:
            st.caption("Add 'currentSeason' to the dataset to unlock seasonal insights.")

# ==============================
# End
# ==============================
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Bird Analytics Dashboard")

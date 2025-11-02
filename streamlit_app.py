import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Bird Identification Analytics",
    page_icon="🦜",
    layout="wide"
)

# --- 2. Database Connection and Data Loading ---
@st.cache_data
def load_data():
    """
    Load data from SQLite database ('birds.db') and preprocess time features.
    Returns an empty DataFrame if the database connection fails.
    """
    
    try:
        # Establish connection and load all data from the 'detections' table
        conn = sqlite3.connect('birds.db')
        query = "SELECT * FROM detections"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except sqlite3.OperationalError as e:
        st.error(f"Database Error: Could not connect to 'birds.db' or load the 'detections' table. Details: {e}")
        # Return an empty DataFrame on failure
        return pd.DataFrame({
            'Date': [], 'Time': [], 'Com_Name': [], 'Sci_Name': [], 'Confidence': [], 
            'DateTime': [], 'Hour': [], 'DayName': [], 'Week': []
        })

    # Ensure all required columns are present before proceeding
    required_cols = ['Date', 'Time', 'Com_Name', 'Sci_Name', 'Confidence']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Database table 'detections' is missing required columns: {', '.join(missing_cols)}")
        return pd.DataFrame() 

    # --- Data Preprocessing ---
    # Ensure Date is a date object and create a combined datetime object
    try:
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])

        # Preprocess time features
        df['Hour'] = df['DateTime'].dt.hour
        df['DayName'] = df['DateTime'].dt.day_name()
        # isocalendar().week returns the ISO week number
        df['Week'] = df['DateTime'].dt.isocalendar().week.astype(int) 
    except Exception as e:
        st.error(f"Error during data preprocessing. Check that 'Date' and 'Time' columns are in valid formats. Details: {e}")
        return pd.DataFrame()
    
    return df

# --- 3. Main Application Logic ---
try:
    df = load_data()
    
    # --- Title and Description ---
    st.title("🦜 Backyard Bird Activity Dashboard")
    st.markdown("Analyze detection trends, species frequency, and daily/hourly activity patterns.")
    st.markdown("---")
    
    # --- 4. Sidebar Filters ---
    st.sidebar.header("Filter Detections")
    
    # Check if data loaded successfully before getting min/max dates
    if df.empty:
        # Stop execution if data loading failed (handled by load_data's error message)
        st.stop()
        
    min_date_global = df['Date'].min()
    max_date_global = df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date_global, max_date_global),
        min_value=min_date_global,
        max_value=max_date_global
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence Score",
        min_value=0.0,
        max_value=1.0,
        value=0.7, 
        step=0.05,
        format="%.2f"
    )
    
    all_species = sorted(df['Com_Name'].unique())
    selected_species = st.sidebar.multiselect(
        "Filter by Species",
        options=all_species,
        default=[]
    )
    
    # --- 5. Apply Filters ---
    filtered_df = df.copy()
    
    # Date filtering
    if len(date_range) == 2:
        start_date = date_range[0]
        end_date = date_range[1]
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) &
            (filtered_df['Date'] <= end_date)
        ]
        
    # Confidence filtering
    filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence]
    
    # Species filtering
    if selected_species:
        filtered_df = filtered_df[filtered_df['Com_Name'].isin(selected_species)]

    # Handle case where no data is left after filtering
    if filtered_df.empty:
        st.warning("No detections found for the selected filters.")
        st.stop()
        
    # --- 6. Tab Structure ---
    tab1, tab2 = st.tabs(["📊 Dashboard Overview", "📈 Detailed Statistics & Data"])

    with tab1:
        
        st.subheader("Key Performance Indicators (KPIs)")
        
        # --- Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        
        total_detections = len(filtered_df)
        unique_species = filtered_df['Com_Name'].nunique()
        avg_confidence = filtered_df['Confidence'].mean()
        detection_days = filtered_df['Date'].nunique()

        # Calculate a simple delta for total detections (comparison against the 7-day period prior to the current selection)
        # Note: pd.Timedelta requires a full datetime object or conversion, which is fine here since df['Date'] contains Python date objects which can be subtracted from pd.Timedelta results after conversion.
        current_min_date = pd.to_datetime(filtered_df['Date'].min())
        comparison_end = current_min_date - pd.Timedelta(days=1) if detection_days > 0 else pd.to_datetime(df['Date'].min())
        comparison_start = comparison_end - pd.Timedelta(days=7)
        
        
        previous_7_days_count = df[
            (pd.to_datetime(df['Date']) > comparison_start) & 
            (pd.to_datetime(df['Date']) <= comparison_end)
        ].shape[0]
        
        delta_str = None
        if previous_7_days_count > 0 and total_detections > 0:
            change = ((total_detections - previous_7_days_count) / previous_7_days_count) * 100
            delta_str = f"{change:+.1f}% vs prior 7 days"

        with col1:
            st.metric(
                "Total Detections", 
                total_detections,
                delta=delta_str if delta_str else None,
            )
        
        with col2:
            st.metric("Unique Species", unique_species)
        
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            detection_rate = total_detections / (detection_days or 1)
            st.metric("Avg Detections/Day", f"{detection_rate:.1f}", help="Average detections over the number of days with activity.")

        st.markdown("---")
        
        # --- Chart Row 1: Species Frequency & Time Trends ---
        col_species, col_trend = st.columns([1, 1.5])

        with col_species:
            st.subheader("Top 10 Most Frequent Visitors")
            
            # Calculate and display top 10 species (sorted ascending for better bar chart readability)
            species_counts = filtered_df['Com_Name'].value_counts().head(10).sort_values(ascending=True)
            
            # Using st.bar_chart for a horizontal presentation (index is the species name)
            st.bar_chart(species_counts, use_container_width=True)
            st.caption("Count of detections for the top 10 species.")
            
        with col_trend:
            st.subheader("Daily Detection Trend")
            
            # Prepare daily data
            daily_counts = filtered_df.groupby('Date').size()
            
            # Use st.line_chart for visualizing trend over time
            st.line_chart(daily_counts, use_container_width=True)
            st.caption("Total detections per day over the selected date range.")
            
        st.markdown("---")

        # --- Chart Row 2: Temporal Patterns ---
        col_hourly, col_weekly = st.columns(2)
        
        with col_hourly:
            st.subheader("Hourly Activity Pattern")
            
            # Group by hour and count, ensuring all 24 hours are present
            hourly_counts = filtered_df.groupby('Hour').size().reindex(range(24), fill_value=0)
            
            st.bar_chart(hourly_counts, use_container_width=True)
            st.caption("Distribution of detections across 24 hours (0=midnight).")
            
        with col_weekly:
            st.subheader("Activity by Day of Week")
            
            # Map DayName to a numerical order for correct sorting
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = filtered_df['DayName'].value_counts().reindex(day_order, fill_value=0)
            
            # Using st.area_chart for a softer look on the weekly cycle
            st.area_chart(day_counts, use_container_width=True)
            st.caption("Total detections grouped by day of the week.")

        st.markdown("---")
        
        # --- Chart Row 3: Confidence Distribution ---
        st.subheader("Confidence Distribution")
        
        # Prepare data for confidence histogram
        # Create bins and counts
        confidence_bins = pd.cut(filtered_df['Confidence'], bins=20, include_lowest=True, precision=2)
        confidence_counts = confidence_bins.value_counts().sort_index()
        
        # Convert the intervals to strings for a better X-axis label
        chart_data = confidence_counts.rename(lambda x: f"{x.left:.2f}-{x.right:.2f}")

        st.bar_chart(chart_data, use_container_width=True)
        st.caption("Frequency of detections across different confidence score ranges.")


    with tab2:
        
        # --- Species Statistics Table ---
        st.subheader("Species Performance Metrics")
        
        species_stats = filtered_df.groupby('Com_Name').agg(
            Count=('Com_Name', 'count'),
            Avg_Confidence=('Confidence', 'mean'),
            Min_Confidence=('Confidence', 'min'),
            Max_Confidence=('Confidence', 'max')
        ).round(2).sort_values('Count', ascending=False)
        
        # Reset index to make Com_Name a column and rename
        species_stats = species_stats.reset_index().rename(columns={'Com_Name': 'Common Name'})
        
        st.dataframe(
            species_stats, 
            use_container_width=True, 
            height=400,
            column_order=['Common Name', 'Count', 'Avg_Confidence', 'Min_Confidence', 'Max_Confidence']
        )
        st.caption("Aggregated statistics for each unique species detected.")
        
        st.markdown("---")
        
        # --- Detailed Data Table with Native Search/Sort ---
        st.subheader("Raw Detections Data")
        
        display_cols = ['Date', 'Time', 'Com_Name', 'Sci_Name', 'Confidence', 'DayName', 'Week', 'Hour']
        
        # Use native st.dataframe features for search and sort
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            height=500
        )
        st.caption("Full list of detections based on current filters. Use the search bar in the table header to filter.")
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"bird_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key='download_csv_button'
        )

# --- 7. Final Error Handling ---
except Exception as e:
    # This catches general errors, while load_data catches database connection errors
    st.error(f"An unexpected application error occurred: {str(e)}")
    st.info("Please ensure the data structure is correct (Date, Time, Com_Name, Sci_Name, Confidence columns).")

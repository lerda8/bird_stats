import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Bird Identification Analytics",
    page_icon="🦜",
    layout="wide"
)

# Database connection
@st.cache_data
def load_data():
    """Load data from SQLite database"""
    conn = sqlite3.connect('birds.db')
    query = "SELECT * FROM detections"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

# Load data
try:
    df = load_data()
    
    # Title and description
    st.title("🦜 Bird Identification Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Confidence threshold
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f"
    )
    
    # Species filter
    all_species = sorted(df['Com_Name'].unique())
    selected_species = st.sidebar.multiselect(
        "Filter by Species",
        options=all_species,
        default=[]
    )
    
    # Apply filters
    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
        ]
    filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence]
    if selected_species:
        filtered_df = filtered_df[filtered_df['Com_Name'].isin(selected_species)]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Identifications", len(filtered_df))
    
    with col2:
        st.metric("Unique Species", filtered_df['Com_Name'].nunique())
    
    with col3:
        avg_confidence = filtered_df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col4:
        date_span = (filtered_df['Date'].max() - filtered_df['Date'].min()).days
        st.metric("Date Span (days)", date_span)
    
    st.markdown("---")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top species chart
        st.subheader("Top 10 Most Identified Species")
        species_counts = filtered_df['Com_Name'].value_counts().head(10)
        st.bar_chart(species_counts)
    
    with col2:
        # Confidence distribution
        st.subheader("Confidence Distribution")
        # Create histogram with numeric bins
        confidence_hist, bin_edges = pd.cut(filtered_df['Confidence'], bins=20, retbins=True)
        confidence_counts = confidence_hist.value_counts().sort_index()
        # Convert interval index to midpoints for plotting
        bin_midpoints = [(interval.left + interval.right) / 2 for interval in confidence_counts.index]
        chart_data = pd.Series(confidence_counts.values, index=bin_midpoints)
        st.bar_chart(chart_data)
    
    # Time series analysis
    st.subheader("Identifications Over Time")
    daily_counts = filtered_df.groupby('Date').size()
    st.line_chart(daily_counts)
    
    # Weekly and hourly analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Identifications by Week")
        week_counts = filtered_df.groupby('Week').size()
        st.bar_chart(week_counts)
    
    with col2:
        st.subheader("Hourly Activity Pattern")
        # Extract hour from time column
        filtered_df['hour'] = pd.to_datetime(filtered_df['Time'], format='%H:%M:%S').dt.hour
        hourly_counts = filtered_df.groupby('hour').size()
        st.line_chart(hourly_counts)
    
    # Detailed data table
    st.markdown("---")
    st.subheader("Detailed Data")
    
    # Add search functionality
    search_term = st.text_input("Search species (common or scientific name)")
    if search_term:
        display_df = filtered_df[
            (filtered_df['Com_Name'].str.contains(search_term, case=False, na=False)) |
            (filtered_df['Sci_Name'].str.contains(search_term, case=False, na=False))
        ]
    else:
        display_df = filtered_df
    
    # Sort options
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=['Date', 'Confidence', 'Com_Name', 'Week'],
            index=0
        )
    with col2:
        sort_order = st.radio("Order", options=['Desc', 'Asc'], horizontal=True)
    
    display_df = display_df.sort_values(
        by=sort_by,
        ascending=(sort_order == 'Asc')
    )
    
    # Display table
    st.dataframe(
        display_df[['Date', 'Time', 'Com_Name', 'Sci_Name', 'Confidence', 'Week']],
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name=f"bird_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Species statistics
    st.markdown("---")
    st.subheader("Species Statistics")
    
    species_stats = filtered_df.groupby('Com_Name').agg({
        'Confidence': ['mean', 'min', 'max', 'count'],
        'Sci_Name': 'first'
    }).round(2)
    
    species_stats.columns = ['Avg Confidence', 'Min Confidence', 'Max Confidence', 'Count', 'Scientific Name']
    species_stats = species_stats.sort_values('Count', ascending=False)
    species_stats = species_stats.reset_index()
    species_stats.columns = ['Common Name', 'Avg Confidence', 'Min Confidence', 'Max Confidence', 'Count', 'Scientific Name']
    
    st.dataframe(species_stats, use_container_width=True, height=400)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure 'birds.db' exists in the same directory and contains a 'detections' table with the required columns.")

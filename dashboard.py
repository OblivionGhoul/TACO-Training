import streamlit as st
import pandas as pd
import altair as alt
import os

# 1. Page Configuration
st.set_page_config(page_title="Auto-Bot Mission Control", layout="wide")

# 2. Data Loading Logic
def load_historical_data():
    csv_path = "trash_classification_results.csv" 
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Fallback if file is missing
        return pd.DataFrame(columns=["image_id", "file_name", "trash_types", "num_objects"])

df = load_historical_data()

# 3. Sidebar for Global Controls (Matching Screenshot 1)
with st.sidebar:
    st.title("🤖 Bot Status")
    st.status("System Active", state="complete")
    st.info("Model: Auto-Bot v2.1")
    if st.button('🔄 Refresh Live Data'):
        st.rerun()

st.title("Auto-Bot Command Center")

# 4. Define Tabs
tab1, tab2 = st.tabs(["📊 Mission Metrics", "⚙️ System Health"])

# --- TAB 1: MISSION METRICS ---
with tab1:
    st.header("Mission Performance & Trash Classification")
    
    # A. Top Table (The Log)
    if not df.empty:
        log_df = df.rename(columns={
            "image_id": "ID",
            "file_name": "IMAGE SOURCE",
            "trash_types": "DETECTIONS",
            "num_objects": "TOTAL COUNT"
        }).sort_values(by="ID", ascending=False)
        
        st.subheader("Trash Classification Data")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # B. Middle Section: Visual Breakdowns (Matching Screenshot 1)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Trash Classification Breakdown")
        if not df.empty:
            all_types = df['trash_types'].str.split(',').explode().str.strip()
            type_counts = all_types.value_counts().reset_index()
            type_counts.columns = ['TRASH_TYPE', 'COUNT']
            st.bar_chart(data=type_counts, x='TRASH_TYPE', y='COUNT', color="#519dd9")

    with col_right:
        st.subheader("Sorting Accuracy")
        # Hardcoded to match your goal screenshot
        accuracy_data = pd.DataFrame([
            {"STATUS": "Success", "COUNT": 45},
            {"STATUS": "Failed Pickup", "COUNT": 5},
            {"STATUS": "Sorting Error", "COUNT": 3}
        ])
        
        acc_chart = alt.Chart(accuracy_data).mark_bar().encode(
            x=alt.X('COUNT:Q', title='Frequency'),
            y=alt.Y('STATUS:N', sort='-x', title=None),
            color=alt.value("#519dd9")
        ).properties(height=300)
        st.altair_chart(acc_chart, use_container_width=True)

# --- TAB 2: SYSTEM HEALTH (Matching Screenshot 2) ---
with tab2:
    st.header("Hardware Diagnostics & Maintenance")
    
    h_col1, h_col2 = st.columns([1, 1.5])
    
    with h_col1:
        st.subheader("Real-time Sensor Status")
        health_data = [
            {"COMPONENT": "Lidar", "STATUS": "🟢 NOMINAL", "LATENCY": "12ms"},
            {"COMPONENT": "Camera (RGB)", "STATUS": "🟢 NOMINAL", "LATENCY": "45ms"},
            {"COMPONENT": "Vacuum Motor", "STATUS": "🟡 WARNING", "LATENCY": "N/A"},
            {"COMPONENT": "Ultrasonic", "STATUS": "🔴 DISCONNECTED", "LATENCY": "ERR"}
        ]
        st.table(health_data)
        
    with h_col2:
        st.subheader("Battery Level (10h Trend)")
        battery_df = pd.DataFrame({
            'Time': ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00'],
            'Level (%)': [100, 88, 72, 55, 38, 21]
        })
        st.area_chart(data=battery_df, x='Time', y='Level (%)')

    st.divider()
    st.warning("⚠️ Maintenance Required: Check Ultrasonic sensor connection.")

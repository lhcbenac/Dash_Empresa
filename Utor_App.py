import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# --- CONFIG ---
st.set_page_config(
    page_title="Utor Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🚀 Utor Analytics")
page = st.sidebar.radio("Navigation", [
    "📤 Upload",
    "📊 Executive Dashboard",
    "🌍 Macro View",
    "👤 Assessor View",
    "💰 Profit Analysis"
])

# --- SESSION STATE ---
if "df_all" not in st.session_state:
    st.session_state["df_all"] = None

# --- HELPERS ---
def parse_chave_to_date(chave):
    try:
        month, year = chave.split('_')
        return datetime(int(year), int(month), 1)
    except:
        return None

def format_currency(value):
    return f"R$ {value:,.2f}"

def calculate_growth_rate(current, previous):
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def create_gauge_chart(value, title, max_val=None):
    if max_val is None:
        max_val = value * 1.5

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.5], 'color': "lightgray"},
                {'range': [max_val*0.5, max_val*0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# --- PAGES ---

if page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload & Data Management</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your Utor Excel file (.xlsx)",
            type=["xlsx"]
        )

    with col2:
        st.info("📋 **Required Columns:**\n- Chave\n- AssessorReal\n- Pix_Assessor\n- Lucro_Empresa (optional)")

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        all_sheets = xls.sheet_names
        expected_cols = {"Chave", "AssessorReal", "Pix_Assessor"}
        data = []
        skipped_sheets = []

        progress = st.progress(0)
        for idx, sheet in enumerate(all_sheets):
            df = pd.read_excel(xls, sheet_name=sheet)
            if not expected_cols.issubset(df.columns):
                skipped_sheets.append(sheet)
                continue

            available_cols = ["Chave", "AssessorReal", "Pix_Assessor"]
            if "Lucro_Empresa" in df.columns:
                available_cols.append("Lucro_Empresa")
            if "Comissão" in df.columns:
                available_cols.append("Comissão")

            df = df[available_cols]
            df["Distribuidor"] = sheet
            df['Chave_Date'] = df['Chave'].apply(parse_chave_to_date)
            df['Month_Year'] = df['Chave_Date'].dt.strftime('%Y-%m')
            data.append(df)
            progress.progress((idx + 1) / len(all_sheets))

        if not data:
            st.error("❌ No valid sheets found.")
        else:
            df_all = pd.concat(data, ignore_index=True)
            st.session_state["df_all"] = df_all

            st.success("✅ Data loaded successfully!")

            st.markdown("### 📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(df_all))
            with col2:
                st.metric("Unique Assessors", df_all["AssessorReal"].nunique())
            with col3:
                st.metric("Unique Distributors", df_all["Distribuidor"].nunique())
            with col4:
                total_pix = df_all["Pix_Assessor"].sum()
                st.metric("Total Pix", format_currency(total_pix))

            st.markdown("### 🔍 Missing Data Check")
            missing = df_all.isnull().sum()
            st.dataframe(missing[missing > 0])

            st.markdown("### 👀 Sample Data")
            st.dataframe(df_all.head())

elif page == "📊 Executive Dashboard":
    st.markdown('<h1 class="main-header">📊 Executive Dashboard</h1>', unsafe_allow_html=True)
    if st.session_state["df_all"] is not None:
        df = st.session_state["df_all"]
        st.dataframe(df.head())
    else:
        st.warning("📤 Please upload data first.")

elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Macro View</h1>', unsafe_allow_html=True)
    st.info("Add Macro View content here.")

elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Assessor View</h1>', unsafe_allow_html=True)
    st.info("Add Assessor View content here.")

elif page == "💰 Profit Analysis":
    st.markdown('<h1 class="main-header">💰 Profit Analysis</h1>', unsafe_allow_html=True)
    st.info("Add Profit Analysis content here.")


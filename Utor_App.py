import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar

# --- CONFIG ---
st.set_page_config(
    page_title="Utor Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for better styling
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
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 Utor Analytics")
page = st.sidebar.radio("Navigation", [
    "📤 Upload", 
    "📊 Executive Dashboard", 
    "🌍 Macro View", 
    "👤 Assessor View",
    "💰 Profit Analysis"
])

# --- SESSION STORAGE ---
if "df_utor" not in st.session_state:
    st.session_state["df_utor"] = None
if "skipped_sheets" not in st.session_state:
    st.session_state["skipped_sheets"] = []
if "uploaded_file_data" not in st.session_state:
    st.session_state["uploaded_file_data"] = None

# --- HELPER FUNCTIONS ---
def parse_chave_to_date(chave):
    """Convert Chave format (MM_YYYY) to datetime"""
    try:
        month, year = chave.split('_')
        return datetime(int(year), int(month), 1)
    except:
        return None

def format_currency(value):
    """Format currency with proper formatting"""
    return f"R$ {value:,.2f}"

def calculate_growth_rate(current, previous):
    """Calculate growth rate between two values"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

# --- UPLOAD PAGE ---
if page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload & Data Management</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your Utor_Detalhado.xlsx file", 
            type=["xlsx"],
            help="File must contain sheets with the required columns"
        )
    
    with col2:
        st.info("📋 **Required Columns:**\n- Chave\n- AssessorReal\n- Pix_Assessor\n- Lucro_Empresa (optional)")
    
    if uploaded_file:
        # Store the uploaded file data in session state
        st.session_state["uploaded_file_data"] = uploaded_file
        
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            all_sheets = xls.sheet_names
            
            # Define expected columns
            required_cols = {"Chave", "AssessorReal", "Pix_Assessor"}
            optional_cols = {"Lucro_Empresa", "Categoria", "Comissão", "Tributo_Retido"}
            
            data = []
            skipped_sheets = []
            
            # Process each sheet
            for sheet in all_sheets:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    
                    # Check if sheet has required columns
                    if not required_cols.issubset(df.columns):
                        skipped_sheets.append(sheet)
                        continue
                    
                    # Select available columns
                    available_cols = ["Chave", "AssessorReal", "Pix_Assessor"]
                    
                    # Add optional columns if they exist
                    for col in optional_cols:
                        if col in df.columns:
                            available_cols.append(col)
                    
                    df_selected = df[available_cols].copy()
                    df_selected["Distribuidor"] = sheet
                    
                    # Fill missing optional columns with 0
                    for col in optional_cols:
                        if col not in df_selected.columns:
                            df_selected[col] = 0
                    
                    data.append(df_selected)
                    
                except Exception as e:
                    skipped_sheets.append(f"{sheet} (Error: {str(e)})")
            
            if not data:
                st.error("❌ No valid sheets found. Please check columns.")
            else:
                # Combine all data
                df_utor = pd.concat(data, ignore_index=True)
                
                # Data preprocessing
                df_utor['Chave_Date'] = df_utor['Chave'].apply(parse_chave_to_date)
                df_utor['Month_Year'] = df_utor['Chave_Date'].dt.strftime('%Y-%m')
                
                # Store data in session state
                st.session_state["df_utor"] = df_utor
                st.session_state["skipped_sheets"] = skipped_sheets
                
                st.success("✅ Data successfully loaded and processed!")
                
                # Enhanced data overview
                st.markdown("### 📊 Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{len(df_utor):,}")
                with col2:
                    st.metric("Unique Assessors", df_utor["AssessorReal"].nunique())
                with col3:
                    st.metric("Distributors", df_utor["Distribuidor"].nunique())
                with col4:
                    total_pix = df_utor["Pix_Assessor"].sum()
                    st.metric("Total Pix", format_currency(total_pix))
                
                # Show processed sheets
                st.markdown("### ✅ Processed Sheets")
                processed_sheets = df_utor['Distribuidor'].unique()
                cols = st.columns(min(len(processed_sheets), 4))
                for i, sheet in enumerate(processed_sheets):
                    with cols[i % 4]:
                        sheet_data = df_utor[df_utor['Distribuidor'] == sheet]
                        st.info(f"**{sheet}**\n{len(sheet_data)} transactions")
                
                # Show skipped sheets if any
                if skipped_sheets:
                    with st.expander("⚠️ Skipped Sheets"):
                        for sheet in skipped_sheets:
                            st.write(f"- {sheet}")
                
                # Sample data preview
                st.markdown("### 👀 Sample Data Preview")
                display_cols = ['Chave', 'AssessorReal', 'Distribuidor', 'Pix_Assessor', 'Lucro_Empresa']
                available_display_cols = [col for col in display_cols if col in df_utor.columns]
                sample_data = df_utor[available_display_cols].head(10)
                st.dataframe(sample_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- EXECUTIVE DASHBOARD ---
elif page == "📊 Executive Dashboard":
    st.markdown('<h1 class="main-header">📊 Executive Dashboard</h1>', unsafe_allow_html=True)

    if st.session_state["df_utor"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()

    df = st.session_state["df_utor"]

    # Time Period Filter
    col1 = st.columns(1)[0]
    with col1:
        chave_list = sorted(df["Chave"].dropna().unique())
        selected_chaves = st.multiselect(
            "🕐 Select Time Periods",
            chave_list,
            default=chave_list[-6:] if len(chave_list) >= 6 else chave_list
        )

    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]

        # KPI Cards
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)

        total_pix = df_filtered["Pix_Assessor"].sum()
        total_profit = df_filtered["Lucro_Empresa"].sum()
        avg_transaction = df_filtered["Pix_Assessor"].mean()
        active_assessors = df_filtered["AssessorReal"].nunique()
        active_distributors = df_filtered["Distribuidor"].nunique()

        with col1:
            st.metric("Total Pix", format_currency(total_pix))
        with col2:
            st.metric("Company Profit", format_currency(total_profit))
        with col3:
            st.metric("Avg Transaction", format_currency(avg_transaction))
        with col4:
            st.metric("Active Assessors", active_assessors)
        with col5:
            st.metric("Active Distributors", active_distributors)

        # Charts Row 1
        col1, col2 = st.columns(2)

        with col1:
            # Pix Evolution
            monthly_pix = df_filtered.groupby('Chave')['Pix_Assessor'].sum().reset_index()
            monthly_pix['Chave_Date'] = monthly_pix['Chave'].apply(parse_chave_to_date)
            monthly_pix = monthly_pix.sort_values('Chave_Date')

            fig_pix = px.line(
                monthly_pix,
                x='Chave',
                y='Pix_Assessor',
                title='📈 Pix Evolution',
                markers=True
            )
            fig_pix.update_layout(xaxis_title="Period", yaxis_title="Pix Amount (R$)")
            st.plotly_chart(fig_pix, use_container_width=True)

        with col2:
            # Top Assessors
            top_assessors = df_filtered.groupby('AssessorReal')['Pix_Assessor'].sum().nlargest(10)
            fig_assessors = px.bar(
                x=top_assessors.values,
                y=top_assessors.index,
                orientation='h',
                title='🏆 Top 10 Assessors by Pix',
                color=top_assessors.values,
                color_continuous_scale='Blues'
            )
            fig_assessors.update_layout(xaxis_title="Pix Amount (R$)", yaxis_title="Assessor")
            st.plotly_chart(fig_assessors, use_container_width=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            # Distributor Performance
            distributor_data = df_filtered.groupby('Distribuidor')['Pix_Assessor'].sum()
            fig_dist = px.pie(
                values=distributor_data.values,
                names=distributor_data.index,
                title='🎯 Pix Distribution by Distributor'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Profit Analysis by Period
            profit_data = df_filtered.groupby('Chave').agg({
                'Pix_Assessor': 'sum',
                'Lucro_Empresa': 'sum'
            }).reset_index()
            
            fig_profit = px.bar(
                profit_data,
                x='Chave',
                y='Lucro_Empresa',
                title='💰 Profit by Period',
                color='Lucro_Empresa',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_profit, use_container_width=True)
    else:
        st.info("Please select at least one time period to display the dashboard.")

# --- MACRO VIEW PAGE ---
elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Macro View - Assessor Performance</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_utor"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_utor"]
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        chave_list = sorted(df["Chave"].dropna().unique())
        selected_chaves = st.multiselect(
            "🕐 Select Time Periods",
            chave_list,
            default=chave_list
        )
    
    with col2:
        min_pix = st.number_input("💰 Minimum Pix Filter", min_value=0.0, value=0.0, step=1000.0)
    
    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]
        
        # Summary calculations
        financial_cols = ["Pix_Assessor", "Lucro_Empresa"]
        summary_df = df_filtered.groupby("AssessorReal")[financial_cols].sum().reset_index()
        
        # Add calculated metrics
        summary_df['Transaction_Count'] = df_filtered.groupby("AssessorReal").size().values
        summary_df['Avg_Transaction'] = summary_df['Pix_Assessor'] / summary_df['Transaction_Count']
        
        # Add distributor count per assessor
        distributor_counts = df_filtered.groupby("AssessorReal")["Distribuidor"].nunique().reset_index()
        distributor_counts.columns = ["AssessorReal", "Distributor_Count"]
        summary_df = summary_df.merge(distributor_counts, on="AssessorReal")
        
        # Filter by minimum pix
        summary_df = summary_df[summary_df['Pix_Assessor'] >= min_pix]
        summary_df = summary_df.sort_values("Pix_Assessor", ascending=False)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessors", len(summary_df))
        with col2:
            st.metric("Total Pix", format_currency(summary_df['Pix_Assessor'].sum()))
        with col3:
            st.metric("Avg Pix/Assessor", format_currency(summary_df['Pix_Assessor'].mean()))
        
        # Enhanced table with formatting
        st.markdown("### 📋 Assessor Performance Summary")
        
        # Create pivot table for detailed view
        pivot_df = pd.pivot_table(
            df_filtered,
            index="AssessorReal",
            columns="Distribuidor",
            values="Pix_Assessor",
            aggfunc="sum",
            fill_value=0,
            margins=True,
            margins_name="Total"
        ).reset_index()
        
        st.dataframe(pivot_df.round(2), use_container_width=True, height=400)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers
            top_10 = summary_df.head(10)
            fig_top = px.bar(
                top_10,
                x='Pix_Assessor',
                y='AssessorReal',
                orientation='h',
                title='🏆 Top 10 Performers',
                color='Distributor_Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Scatter plot: Pix vs Transaction Count
            fig_scatter = px.scatter(
                summary_df,
                x='Pix_Assessor',
                y='Transaction_Count',
                size='Avg_Transaction',
                color='Distributor_Count',
                hover_data=['AssessorReal'],
                title='💰 Pix vs Transaction Volume',
                labels={'Pix_Assessor': 'Total Pix (R$)', 'Transaction_Count': 'Number of Transactions'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = pivot_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Pivot Summary (CSV)",
                csv_data,
                f"Macro_Summary_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )
        
        with col2:
            # Full filtered data
            csv_full = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📦 Download Full Data (CSV)",
                csv_full,
                f"Full_Data_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Individual Assessor Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_utor"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_utor"]
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        assessor_list = sorted(df["AssessorReal"].dropna().unique())
        selected_assessor = st.selectbox("👤 Select Assessor", assessor_list)
    
    with col2:
        chave_list = sorted(df["Chave"].dropna().unique())
        selected_chaves = st.multiselect(
            "🕐 Select Time Periods (optional)",
            chave_list,
            default=chave_list
        )
    
    if selected_assessor:
        # Filter data
        df_filtered = df[df["AssessorReal"] == selected_assessor]
        if selected_chaves:
            df_filtered = df_filtered[df_filtered["Chave"].isin(selected_chaves)]
        
        if df_filtered.empty:
            st.warning("No data found for the selected criteria.")
        else:
            # Individual KPIs
            st.markdown(f"### 📊 Performance Overview: {selected_assessor}")
            
            col1, col2, col3, col4 = st.columns(4)

            total_pix = df_filtered["Pix_Assessor"].sum()
            total_transactions = len(df_filtered)
            total_profit = df_filtered["Lucro_Empresa"].sum()
            distributor_count = df_filtered["Distribuidor"].nunique()
            
            with col1:
                st.metric("Total Pix", format_currency(total_pix))
            with col2:
                st.metric("Total Transactions", total_transactions)
            with col3:
                st.metric("Generated Profit", format_currency(total_profit))
            with col4:
                st.metric("Active Distributors", distributor_count)
            
            # Performance breakdown
            st.markdown("### 📋 Performance by Distributor")
            
            # Create pivot table
            pivot_df = pd.pivot_table(
                df_filtered,
                index="Distribuidor",
                columns="Chave",
                values="Pix_Assessor",
                aggfunc="sum",
                fill_value=0,
                margins=True,
                margins_name="Total"
            ).reset_index()
            
            st.dataframe(pivot_df.round(2), use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Distributor performance
                dist_summary = df_filtered.groupby("Distribuidor")["Pix_Assessor"].sum()
                fig_dist = px.bar(
                    x=dist_summary.index,
                    y=dist_summary.values,
                    title=f'📊 {selected_assessor} - Performance by Distributor',
                    labels={'x': 'Distributor', 'y': 'Pix Amount (R$)'},
                    color=dist_summary.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Timeline performance
                timeline_data = df_filtered.groupby("Chave")["Pix_Assessor"].sum().reset_index()
                timeline_data['Chave_Date'] = timeline_data['Chave'].apply(parse_chave_to_date)
                timeline_data = timeline_data.sort_values('Chave_Date')
                
                fig_timeline = px.line(
                    timeline_data,
                    x='Chave',
                    y='Pix_Assessor',
                    title=f'📈 {selected_assessor} - Performance Timeline',
                    markers=True
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Download section
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Pivot table CSV
                csv_pivot = pivot_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Download Pivot Summary",
                    csv_pivot,
                    f"{selected_assessor}_Summary.csv",
                    "text/csv"
                )
            
            with col2:
                # Full data CSV
                csv_full = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📦 Download Full Data",
                    csv_full,
                    f"{selected_assessor}_FullData.csv",
                    "text/csv"
                )
            
            with col3:
                # Performance summary
                performance_summary = pd.DataFrame({
                    'Metric': ['Total Pix', 'Total Transactions', 'Avg Transaction', 'Total Profit', 'Distributors'],
                    'Value': [total_pix, total_transactions, total_pix/total_transactions, total_profit, distributor_count]
                })
                csv_perf = performance_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🎯 Download Performance Summary",
                    csv_perf,
                    f"{selected_assessor}_Performance.csv",
                    "text/csv"
                )

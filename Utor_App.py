import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar
from io import BytesIO

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
    "💰 Profit Analysis",
    "📈 Performance Analytics"
])

# --- SESSION STORAGE ---
if "df_all" not in st.session_state:
    st.session_state["df_all"] = None
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

def create_gauge_chart(value, title, max_val=None):
    """Create a gauge chart for KPIs"""
    if max_val is None:
        max_val = value * 1.5
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
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
        st.info("📋 **Required Columns:**\n- Chave\n- AssessorReal\n- Pix_Assessor\n- Lucro_Empresa (for profit analysis)")
    
    if uploaded_file:
        st.session_state["uploaded_file_data"] = uploaded_file
        
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            all_sheets = xls.sheet_names
            expected_cols = {"Chave", "AssessorReal", "Pix_Assessor"}
            data = []
            skipped_sheets = []
            
            progress_bar = st.progress(0)
            total_sheets = len(all_sheets)
            
            for idx, sheet in enumerate(all_sheets):
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    if not expected_cols.issubset(df.columns):
                        skipped_sheets.append(sheet)
                        continue
                    
                    # Include additional columns if they exist
                    available_cols = ["Chave", "AssessorReal", "Pix_Assessor"]
                    if "Lucro_Empresa" in df.columns:
                        available_cols.append("Lucro_Empresa")
                    if "Comissão" in df.columns:
                        available_cols.append("Comissão")
                    
                    df = df[available_cols]
                    df["Distribuidor"] = sheet
                    
                    # Add date parsing
                    df['Chave_Date'] = df['Chave'].apply(parse_chave_to_date)
                    df['Month_Year'] = df['Chave_Date'].dt.strftime('%Y-%m')
                    
                    data.append(df)
                    progress_bar.progress((idx + 1) / total_sheets)
                except Exception as e:
                    skipped_sheets.append(f"{sheet} (Error: {str(e)})")
            
            if not data:
                st.error("❌ No valid sheets found. Please check columns.")
            else:
                df_all = pd.concat(data, ignore_index=True)
                st.session_state["df_all"] = df_all
                st.session_state["skipped_sheets"] = skipped_sheets
                st.success("✅ Data successfully loaded and processed!")
                
                # Enhanced data overview
                st.markdown("### 📊 Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{len(df_all):,}")
                with col2:
                    st.metric("Unique Assessors", df_all["AssessorReal"].nunique())
                with col3:
                    st.metric("Unique Distributors", df_all["Distribuidor"].nunique())
                with col4:
                    total_pix = df_all["Pix_Assessor"].sum()
                    st.metric("Total Pix Assessor", format_currency(total_pix))
                
                # Data quality checks
                st.markdown("### 🔍 Data Quality Assessment")
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_data = df_all.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning("⚠️ Missing Data Found:")
                        st.dataframe(missing_data[missing_data > 0])
                    else:
                        st.success("✅ No missing data detected")
                
                with col2:
                    # Time periods
                    periods = df_all['Chave'].nunique()
                    st.info(f"📅 **Time Periods:** {periods}")
                    
                    # Top distributors
                    top_distributors = df_all['Distribuidor'].value_counts().head(3)
                    st.info("🏆 **Top Distributors:**\n" + "\n".join([f"• {dist}: {count}" for dist, count in top_distributors.items()]))
                
                # Show skipped sheets
                if skipped_sheets:
                    with st.expander("⚠️ Skipped Sheets"):
                        for s in skipped_sheets:
                            st.write(f"- {s}")
                
                # Sample data preview
                st.markdown("### 👀 Sample Data Preview")
                display_cols = ['Chave', 'AssessorReal', 'Distribuidor', 'Pix_Assessor']
                if 'Lucro_Empresa' in df_all.columns:
                    display_cols.append('Lucro_Empresa')
                    
                sample_data = df_all[display_cols].head(10)
                st.dataframe(sample_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# --- EXECUTIVE DASHBOARD ---
elif page == "📊 Executive Dashboard":
    st.markdown('<h1 class="main-header">📊 Executive Dashboard</h1>', unsafe_allow_html=True)

    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()

    df = st.session_state["df_all"]

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
        avg_pix = df_filtered["Pix_Assessor"].mean()
        active_assessors = df_filtered["AssessorReal"].nunique()
        active_distributors = df_filtered["Distribuidor"].nunique()
        total_transactions = len(df_filtered)

        with col1:
            st.metric("Total Pix Assessor", format_currency(total_pix))
        with col2:
            st.metric("Average Pix", format_currency(avg_pix))
        with col3:
            st.metric("Active Assessors", active_assessors)
        with col4:
            st.metric("Active Distributors", active_distributors)
        with col5:
            st.metric("Total Transactions", total_transactions)

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
                title='📈 Pix Assessor Evolution',
                markers=True
            )
            fig_pix.update_layout(xaxis_title="Period", yaxis_title="Pix Assessor (R$)")
            st.plotly_chart(fig_pix, use_container_width=True)

        with col2:
            # Top Assessors
            top_assessors = df_filtered.groupby('AssessorReal')['Pix_Assessor'].sum().nlargest(10)
            fig_assessors = px.bar(
                x=top_assessors.values,
                y=top_assessors.index,
                orientation='h',
                title='🏆 Top 10 Assessors by Pix'
            )
            fig_assessors.update_layout(xaxis_title="Pix Assessor (R$)", yaxis_title="Assessor")
            st.plotly_chart(fig_assessors, use_container_width=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            # Distributor Distribution
            distributor_dist = df_filtered.groupby('Distribuidor')['Pix_Assessor'].sum()
            fig_pie = px.pie(
                values=distributor_dist.values,
                names=distributor_dist.index,
                title='🎯 Pix Distribution by Distributor'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Transaction Volume by Period
            transaction_count = df_filtered.groupby('Chave').size().reset_index(name='Transaction_Count')
            fig_volume = px.bar(
                transaction_count,
                x='Chave',
                y='Transaction_Count',
                title='📊 Transaction Volume by Period',
                color='Transaction_Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_volume, use_container_width=True)

        # Additional KPIs if Lucro_Empresa exists
        if 'Lucro_Empresa' in df_filtered.columns:
            st.markdown("### 💰 Profit Analysis")
            col1, col2, col3 = st.columns(3)
            
            total_profit = df_filtered["Lucro_Empresa"].sum()
            avg_profit = df_filtered["Lucro_Empresa"].mean()
            
            with col1:
                st.metric("Total Profit", format_currency(total_profit))
            with col2:
                st.metric("Average Profit", format_currency(avg_profit))
            with col3:
                if total_pix > 0:
                    profit_ratio = (total_profit / total_pix) * 100
                    st.metric("Profit/Pix Ratio", f"{profit_ratio:.1f}%")
    else:
        st.info("Please select at least one time period to display the dashboard.")

# --- MACRO VIEW PAGE ---
elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Macro Summary View</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_all"]
    
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
        min_pix = st.number_input("💰 Minimum Pix Filter", min_value=0.0, value=0.0, step=100.0)
    
    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]
        
        # Create pivot table
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
        
        # Filter by minimum pix
        if min_pix > 0:
            pivot_df = pivot_df[pivot_df['Total'] >= min_pix]
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessors", len(pivot_df) - 1)  # -1 for Total row
        with col2:
            total_pix = pivot_df.loc[pivot_df['AssessorReal'] == 'Total', 'Total'].values[0] if 'Total' in pivot_df['AssessorReal'].values else 0
            st.metric("Total Pix", format_currency(total_pix))
        with col3:
            avg_pix = pivot_df[pivot_df['AssessorReal'] != 'Total']['Total'].mean()
            st.metric("Avg Pix/Assessor", format_currency(avg_pix))
        
        st.markdown(f"### 📋 Summary for Period(s): `{', '.join(map(str, selected_chaves))}`")
        
        # Enhanced table display
        st.dataframe(pivot_df.round(2), use_container_width=True, height=400)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers (excluding Total row)
            top_performers = pivot_df[pivot_df['AssessorReal'] != 'Total'].nlargest(10, 'Total')
            fig_top = px.bar(
                top_performers,
                x='Total',
                y='AssessorReal',
                orientation='h',
                title='🏆 Top 10 Performers by Total Pix'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Distributor totals
            distributor_totals = {}
            for col in pivot_df.columns:
                if col not in ['AssessorReal', 'Total']:
                    distributor_totals[col] = pivot_df[col].sum()
            
            fig_dist = px.bar(
                x=list(distributor_totals.keys()),
                y=list(distributor_totals.values()),
                title='📊 Total Pix by Distributor'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Download Summary CSV",
                csv,
                f"Pix_Summary_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )
        
        with col2:
            csv_all = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📦 Download Full Raw Data",
                csv_all,
                f"FullData_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )
    else:
        st.info("Please select at least one time period.")

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Individual Assessor Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_all"]
    
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
        assessor_list = sorted(df["AssessorReal"].dropna().unique())
        selected_assessor = st.selectbox("👤 Select Assessor", assessor_list)
    
    if selected_chaves and selected_assessor:
        df_filtered = df[
            (df["AssessorReal"] == selected_assessor) &
            (df["Chave"].isin(selected_chaves))
        ]
        
        if df_filtered.empty:
            st.warning("No data found for the selected criteria.")
        else:
            # Individual KPIs
            st.markdown(f"### 📊 Performance Overview: {selected_assessor}")
            
            col1, col2, col3, col4 = st.columns(4)

            total_pix = df_filtered["Pix_Assessor"].sum()
            total_transactions = len(df_filtered)
            avg_pix = df_filtered["Pix_Assessor"].mean()
            distributors_count = df_filtered["Distribuidor"].nunique()
            
            with col1:
                st.metric("Total Pix Assessor", format_currency(total_pix))
            with col2:
                st.metric("Total Transactions", total_transactions)
            with col3:
                st.metric("Average Pix", format_currency(avg_pix))
            with col4:
                st.metric("Active Distributors", distributors_count)
            
            # Create pivot table for this assessor
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
            
            st.markdown("### 📋 Performance by Distributor and Period")
            st.dataframe(pivot_df.round(2), use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance over time
                time_performance = df_filtered.groupby('Chave')['Pix_Assessor'].sum().reset_index()
                time_performance['Chave_Date'] = time_performance['Chave'].apply(parse_chave_to_date)
                time_performance = time_performance.sort_values('Chave_Date')
                
                fig_time = px.line(
                    time_performance,
                    x='Chave',
                    y='Pix_Assessor',
                    title=f'📈 {selected_assessor} - Performance Over Time',
                    markers=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Performance by distributor
                dist_performance = df_filtered.groupby('Distribuidor')['Pix_Assessor'].sum().reset_index()
                fig_dist = px.bar(
                    dist_performance,
                    x='Distribuidor',
                    y='Pix_Assessor',
                    title=f'📊 {selected_assessor} - Performance by Distributor'
                )
                fig_dist.update_xaxes(tickangle=45)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Profit analysis if available
            if 'Lucro_Empresa' in df_filtered.columns:
                st.markdown("### 💰 Profit Analysis")
                col1, col2, col3 = st.columns(3)
                
                total_profit = df_filtered["Lucro_Empresa"].sum()
                avg_profit = df_filtered["Lucro_Empresa"].mean()
                
                with col1:
                    st.metric("Total Profit", format_currency(total_profit))
                with col2:
                    st.metric("Average Profit", format_currency(avg_profit))
                with col3:
                    if total_pix > 0:
                        profit_ratio = (total_profit / total_pix) * 100
                        st.metric("Profit/Pix Ratio", f"{profit_ratio:.1f}%")
            
            # Export options
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📊 Download Summary CSV",
                    csv,
                    f"{selected_assessor}_Summary.csv",
                    "text/csv"
                )
            
            with col2:
                csv_all = df_filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📦 Download Full Data",
                    csv_all,
                    f"{selected_assessor}_FullData.csv",
                    "text/csv"
                )
            
            with col3:
                # Excel export with multiple sheets
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    pivot_df.to_excel(writer, sheet_name='Summary', index=False)
                    df_filtered.to_excel(writer, sheet_name='Raw_Data', index=False)
                    time_performance.to_excel(writer, sheet_name='Time_Performance', index=False)
                
                st.download_button(
                    "📋 Download Complete Report (Excel)",
                    buffer.getvalue(),
                    f"{selected_assessor}_Complete_Report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )



            

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
    page_title="Taurus Analytics Dashboard", 
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
    .filter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #495057;
        margin-bottom: 1rem;
        border-bottom: 2px solid #007bff;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 Taurus Analytics")
page = st.sidebar.radio("Navigation", [
    "📤 Upload", 
    "📊 Executive Dashboard", 
    "🌍 Macro View", 
    "👤 Assessor View", 
    "📈 Performance Analytics"
])

# --- SESSION STORAGE ---
if "df_taurus" not in st.session_state:
    st.session_state["df_taurus"] = None

# --- HELPER FUNCTIONS ---
def parse_chave_to_date(chave):
    """Convert Chave format (MM_YYYY) to datetime"""
    try:
        month, year = chave.split('_')
        return datetime(int(year), int(month), 1)
    except:
        return None

def sort_chave_periods(chave_list):
    """Sort Chave periods chronologically"""
    try:
        chave_dates = [(chave, parse_chave_to_date(chave)) for chave in chave_list]
        chave_dates = [(chave, date) for chave, date in chave_dates if date is not None]
        chave_dates.sort(key=lambda x: x[1])
        return [chave for chave, date in chave_dates]
    except:
        return sorted(chave_list)

def format_currency(value):
    """Format currency with proper formatting"""
    if pd.isna(value):
        return "R$ 0,00"
    return f"R$ {value:,.2f}"

def format_percentage(value, decimals=1):
    """Format percentage with proper formatting"""
    if pd.isna(value):
        return "0.0%"
    return f"{value:.{decimals}f}%"

def calculate_growth_rate(current, previous):
    """Calculate growth rate between two values"""
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0
    return ((current - previous) / previous) * 100

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            formatted_value = format_currency(value)
        else:
            formatted_value = f"{value:,.2f}"
    else:
        formatted_value = str(value)
    
    return st.metric(title, formatted_value, delta=delta, delta_color=delta_color)

def create_summary_statistics(df, group_col, financial_cols):
    """Create comprehensive summary statistics"""
    summary = df.groupby(group_col)[financial_cols].agg(['sum', 'mean', 'count']).round(2)
    summary.columns = [f"{col}_{agg}" for col, agg in summary.columns]
    
    # Add derived metrics
    if 'Comissão_sum' in summary.columns and 'Lucro_Empresa_sum' in summary.columns:
        summary['Profit_Margin'] = (summary['Lucro_Empresa_sum'] / summary['Comissão_sum'] * 100).round(2)
    
    return summary.reset_index()

# --- UPLOAD PAGE ---
if page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload & Data Management</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your Taurus Excel file", 
            type=["xlsx"],
            help="File must contain a 'Taurus' sheet with the required columns"
        )
    
    with col2:
        st.info("📋 **Required Columns:**\n- Chave\n- AssessorReal\n- Categoria\n- Comissão\n- Tributo_Retido\n- Pix_Assessor\n- Lucro_Empresa")
    
    if uploaded_file:
        try:
            # Read specifically the 'Taurus' sheet
            df_taurus = pd.read_excel(uploaded_file, sheet_name="Taurus", engine="openpyxl")
            
            # Check if required columns exist
            required_cols = {"Chave", "AssessorReal", "Categoria", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"}
            
            if not required_cols.issubset(df_taurus.columns):
                missing_cols = required_cols - set(df_taurus.columns)
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                # Data preprocessing
                df_taurus['Chave_Date'] = df_taurus['Chave'].apply(parse_chave_to_date)
                df_taurus['Month_Year'] = df_taurus['Chave_Date'].dt.strftime('%Y-%m')
                
                # Store data in session state
                st.session_state["df_taurus"] = df_taurus
                st.success("✅ Data successfully loaded and processed!")
                
                # Enhanced data overview
                st.markdown("### 📊 Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{len(df_taurus):,}")
                with col2:
                    st.metric("Unique Assessors", df_taurus["AssessorReal"].nunique())
                with col3:
                    st.metric("Time Periods", df_taurus["Chave"].nunique())
                with col4:
                    total_revenue = df_taurus["Comissão"].sum()
                    st.metric("Total Revenue", format_currency(total_revenue))
                
                # Data quality checks
                st.markdown("### 🔍 Data Quality Assessment")
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_data = df_taurus.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning("⚠️ Missing Data Found:")
                        st.dataframe(missing_data[missing_data > 0])
                    else:
                        st.success("✅ No missing data detected")
                
                with col2:
                    # Date range
                    chave_periods = sort_chave_periods(df_taurus['Chave'].unique())
                    if chave_periods:
                        st.info(f"📅 **Period Range:** {chave_periods[0]} to {chave_periods[-1]}")
                    
                    # Top categories
                    top_categories = df_taurus['Categoria'].value_counts().head(3)
                    st.info("🏆 **Top Categories:**\n" + "\n".join([f"• {cat}: {count}" for cat, count in top_categories.items()]))
                
                # Sample data with better formatting
                st.markdown("### 👀 Sample Data Preview")
                display_cols = ['Chave', 'AssessorReal', 'Categoria', 'Comissão', 'Pix_Assessor', 'Lucro_Empresa']
                sample_data = df_taurus[display_cols].head(10)
                st.dataframe(sample_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- EXECUTIVE DASHBOARD ---
elif page == "📊 Executive Dashboard":
    st.markdown('<h1 class="main-header">📊 Executive Dashboard</h1>', unsafe_allow_html=True)

    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()

    df = st.session_state["df_taurus"]

    # Time Period Filter with proper sorting
    with st.container():
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Dashboard Filters")
        
        col1 = st.columns(1)[0]
        with col1:
            chave_list = sort_chave_periods(df["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "🕐 Select Time Periods",
                chave_list,
                default=chave_list[-6:] if len(chave_list) >= 6 else chave_list,
                help="Select one or more time periods to analyze"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]

        # KPI Cards
        st.markdown('<div class="section-header">🎯 Key Performance Indicators</div>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)

        total_revenue = df_filtered["Comissão"].sum()
        total_pix = df_filtered["Pix_Assessor"].sum()
        total_profit = df_filtered["Lucro_Empresa"].sum()
        avg_transaction = df_filtered["Comissão"].mean()
        active_assessors = df_filtered["AssessorReal"].nunique()

        with col1:
            create_metric_card("Total Revenue", total_revenue)
        with col2:
            create_metric_card("Total Pix Assessor", total_pix)
        with col3:
            create_metric_card("Company Profit", total_profit)
        with col4:
            create_metric_card("Avg Transaction", avg_transaction)
        with col5:
            st.metric("Active Assessors", active_assessors)

        # Charts Row 1
        st.markdown('<div class="section-header">📈 Revenue Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            # Revenue Evolution
            monthly_revenue = df_filtered.groupby('Chave')['Comissão'].sum().reset_index()
            monthly_revenue['Chave_Date'] = monthly_revenue['Chave'].apply(parse_chave_to_date)
            monthly_revenue = monthly_revenue.sort_values('Chave_Date')

            fig_revenue = px.line(
                monthly_revenue,
                x='Chave',
                y='Comissão',
                title='📈 Revenue Evolution',
                markers=True
            )
            fig_revenue.update_layout(
                xaxis_title="Period", 
                yaxis_title="Revenue (R$)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

        with col2:
            # Top Assessors
            top_assessors = df_filtered.groupby('AssessorReal')['Comissão'].sum().nlargest(10)
            fig_assessors = px.bar(
                x=top_assessors.values,
                y=top_assessors.index,
                orientation='h',
                title='🏆 Top 10 Assessors by Revenue',
                color=top_assessors.values,
                color_continuous_scale='Blues'
            )
            fig_assessors.update_layout(
                xaxis_title="Revenue (R$)", 
                yaxis_title="Assessor",
                showlegend=False
            )
            st.plotly_chart(fig_assessors, use_container_width=True)

        # Charts Row 2
        st.markdown('<div class="section-header">📊 Distribution Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            # Category Distribution
            category_dist = df_filtered.groupby('Categoria')['Comissão'].sum()
            fig_pie = px.pie(
                values=category_dist.values,
                names=category_dist.index,
                title='🎯 Revenue Distribution by Category'
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Profit Margin Analysis
            profit_margin = df_filtered.groupby('Chave').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum'
            }).reset_index()
            profit_margin['Margin_Percent'] = (profit_margin['Lucro_Empresa'] / profit_margin['Comissão']) * 100

            fig_margin = px.bar(
                profit_margin,
                x='Chave',
                y='Margin_Percent',
                title='📊 Profit Margin by Period (%)',
                color='Margin_Percent',
                color_continuous_scale='RdYlGn'
            )
            fig_margin.update_layout(
                xaxis_title="Period",
                yaxis_title="Profit Margin (%)"
            )
            st.plotly_chart(fig_margin, use_container_width=True)
    else:
        st.info("Please select at least one time period to display the dashboard.")

# --- ENHANCED MACRO VIEW PAGE ---
elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Macro View - Assessor Performance</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("⚠️ Please upload the Excel file first using the Upload page.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Enhanced Filters Section
    with st.container():
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Analysis Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sorted time periods
            chave_list = sort_chave_periods(df["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "🕐 Select Time Periods",
                chave_list,
                default=chave_list[-6:] if len(chave_list) >= 6 else chave_list,
                help="Choose specific time periods for analysis"
            )
        
        with col2:
            # Assessor multi-select with search capability
            assessor_list = sorted(df["AssessorReal"].dropna().unique())
            selected_assessors = st.multiselect(
                "👥 Select Assessors",
                assessor_list,
                default=[],
                help="Leave empty to include all assessors, or select specific ones",
                placeholder="Choose assessors..."
            )
            
            # Quick selection buttons
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("📊 Top 20", help="Select top 20 performers"):
                    if selected_chaves:
                        temp_df = df[df["Chave"].isin(selected_chaves)]
                        top_20 = temp_df.groupby("AssessorReal")["Comissão"].sum().nlargest(20).index.tolist()
                        st.session_state.temp_assessors = top_20
                        st.experimental_rerun()
            
            with col2b:
                if st.button("🔄 Clear All", help="Clear assessor selection"):
                    st.session_state.temp_assessors = []
                    st.experimental_rerun()
            
            # Apply temp selection if exists
            if hasattr(st.session_state, 'temp_assessors'):
                selected_assessors = st.session_state.temp_assessors
                delattr(st.session_state, 'temp_assessors')
        
        with col3:
            # Advanced filters
            min_revenue = st.number_input(
                "💰 Minimum Revenue Filter (R$)", 
                min_value=0.0, 
                value=0.0, 
                step=1000.0,
                help="Filter assessors with revenue below this amount"
            )
            
            min_transactions = st.number_input(
                "📊 Minimum Transactions", 
                min_value=0, 
                value=0, 
                step=1,
                help="Filter assessors with fewer transactions"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_chaves:
        st.warning("⚠️ Please select at least one time period to continue.")
        st.stop()
    
    # Apply filters
    df_filtered = df[df["Chave"].isin(selected_chaves)]
    
    if selected_assessors:
        df_filtered = df_filtered[df_filtered["AssessorReal"].isin(selected_assessors)]
    
    # Calculate comprehensive summary
    financial_cols = ["Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
    if "Receita Bruta" in df.columns:
        financial_cols.insert(0, "Receita Bruta")
    
    # Main summary calculations
    summary_df = df_filtered.groupby("AssessorReal")[financial_cols].sum().reset_index()
    
    # Add calculated metrics
    transaction_counts = df_filtered.groupby("AssessorReal").size().reset_index(name='Transaction_Count')
    summary_df = summary_df.merge(transaction_counts, on="AssessorReal")
    
    # Calculate derived metrics
    summary_df['Avg_Transaction'] = summary_df['Comissão'] / summary_df['Transaction_Count']
    summary_df['Profit_Margin'] = (summary_df['Lucro_Empresa'] / summary_df['Comissão']) * 100
    
    # Apply additional filters
    summary_df = summary_df[
        (summary_df['Comissão'] >= min_revenue) & 
        (summary_df['Transaction_Count'] >= min_transactions)
    ]
    
    # Sort by revenue
    summary_df = summary_df.sort_values("Comissão", ascending=False)
    
    if summary_df.empty:
        st.warning("⚠️ No data matches the selected criteria. Please adjust your filters.")
        st.stop()
    
    # Key Metrics Section
    st.markdown('<div class="section-header">📊 Portfolio Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📥 Total Assessors", len(summary_df))
    with col2:
        create_metric_card("💰 Total Revenue", summary_df['Comissão'].sum())
    with col3:
        create_metric_card("📊 Avg Revenue/Assessor", summary_df['Comissão'].mean())
    with col4:
        create_metric_card("💼 Total Profit", summary_df['Lucro_Empresa'].sum())
    with col5:
        avg_margin = summary_df['Profit_Margin'].mean()
        st.metric("📈 Avg Profit Margin", f"{avg_margin:.1f}%")
    
    # Performance Distribution
    st.markdown('<div class="section-header">📈 Performance Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue distribution histogram
        fig_hist = px.histogram(
            summary_df,
            x='Comissão',
            nbins=20,
            title='📊 Revenue Distribution Among Assessors',
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(
            xaxis_title="Revenue (R$)",
            yaxis_title="Number of Assessors"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Performance quadrant analysis
        fig_scatter = px.scatter(
            summary_df,
            x='Comissão',
            y='Profit_Margin',
            size='Transaction_Count',
            hover_data=['AssessorReal'],
            title='💎 Performance Quadrant: Revenue vs Profit Margin',
            labels={
                'Comissão': 'Revenue (R$)', 
                'Profit_Margin': 'Profit Margin (%)',
                'Transaction_Count': 'Transactions'
            },
            color='Profit_Margin',
            color_continuous_scale='RdYlGn'
        )
        
        # Add quadrant lines
        median_revenue = summary_df['Comissão'].median()
        median_margin = summary_df['Profit_Margin'].median()
        
        fig_scatter.add_hline(y=median_margin, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=median_revenue, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top Performers Analysis
    st.markdown('<div class="section-header">🏆 Top Performers Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performers by revenue
        display_count = min(15, len(summary_df))
        top_performers = summary_df.head(display_count)
        
        fig_top = px.bar(
            top_performers,
            x='Comissão',
            y='AssessorReal',
            orientation='h',
            title=f'💰 Top {display_count} Performers by Revenue',
            color='Profit_Margin',
            color_continuous_scale='RdYlGn',
            hover_data=['Transaction_Count', 'Avg_Transaction']
        )
        fig_top.update_layout(height=500)
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Performance metrics comparison
        top_10 = summary_df.head(10)
        
        # Normalize values for comparison
        metrics_df = pd.DataFrame({
            'Assessor': top_10['AssessorReal'],
            'Revenue_Score': (top_10['Comissão'] / top_10['Comissão'].max() * 100).round(1),
            'Profit_Score': (top_10['Profit_Margin'] / top_10['Profit_Margin'].max() * 100).round(1),
            'Transaction_Score': (top_10['Transaction_Count'] / top_10['Transaction_Count'].max() * 100).round(1)
        })
        
        fig_radar_comp = go.Figure()
        
        for i, assessor in enumerate(metrics_df['Assessor'][:5]):  # Show top 5
            assessor_data = metrics_df[metrics_df['Assessor'] == assessor].iloc[0]
            
            fig_radar_comp.add_trace(go.Scatterpolar(
                r=[assessor_data['Revenue_Score'], assessor_data['Profit_Score'], assessor_data['Transaction_Score']],
                theta=['Revenue', 'Profit Margin', 'Transactions'],
                fill='toself',
                name=assessor[:20] + '...' if len(assessor) > 20 else assessor,
                opacity=0.6
            ))
        
        fig_radar_comp.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="🎯 Top 5 Performance Comparison (Normalized Scores)",
            height=500
        )
        st.plotly_chart(fig_radar_comp, use_container_width=True)
    
    # Detailed Table Section
    st.markdown('<div class="section-header">📋 Detailed Performance Table</div>', unsafe_allow_html=True)
    
    # Table display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_all = st.checkbox("📊 Show All Assessors", value=False)
        display_rows = len(summary_df) if show_all else min(50, len(summary_df))
    
    with col2:
        sort_option = st.selectbox(
            "🔄 Sort by",
            ["Revenue (Desc)", "Revenue (Asc)", "Profit Margin (Desc)", "Profit Margin (Asc)", "Transactions (Desc)"]
        )
    
    with col3:
        export_format = st.selectbox("📤 Export Format", ["CSV", "Excel"])
    
    # Apply sorting
    if "Revenue (Desc)" in sort_option:
        display_df = summary_df.sort_values("Comissão", ascending=False)
    elif "Revenue (Asc)" in sort_option:
        display_df = summary_df.sort_values("Comissão", ascending=True)
    elif "Profit Margin (Desc)" in sort_option:
        display_df = summary_df.sort_values("Profit_Margin", ascending=False)
    elif "Profit Margin (Asc)" in sort_option:
        display_df = summary_df.sort_values("Profit_Margin", ascending=True)
    else:  # Transactions
        display_df = summary_df.sort_values("Transaction_Count", ascending=False)
    
    # Format display dataframe
    display_table = display_df.head(display_rows).copy()
    
    # Format financial columns
    for col in financial_cols:
        if col in display_table.columns:
            display_table[f"{col}_Formatted"] = display_table[col].apply(format_currency)
    
    display_table['Avg_Transaction_Formatted'] = display_table['Avg_Transaction'].apply(format_currency)
    display_table['Profit_Margin_Formatted'] = display_table['Profit_Margin'].apply(lambda x: format_percentage(x))
    
    # Select columns for display
    display_cols = ['AssessorReal', 'Transaction_Count']
    for col in financial_cols:
        if f"{col}_Formatted" in display_table.columns:
            display_cols.append(f"{col}_Formatted")
    display_cols.extend(['Avg_Transaction_Formatted', 'Profit_Margin_Formatted'])
    
    # Rename columns for better presentation
    column_mapping = {
        'AssessorReal': '👤 Assessor',
        'Transaction_Count': '📊 Transactions',
        'Avg_Transaction_Formatted': '💰 Avg Transaction',
        'Profit_Margin_Formatted': '📈 Profit Margin'
    }
    
    for col in financial_cols:
        if f"{col}_Formatted" in display_table.columns:
            column_mapping[f"{col}_Formatted"] = f"💵 {col}"
    
    final_display = display_table[display_cols].rename(columns=column_mapping)
    
    # Display table with styling
    st.dataframe(
        final_display,
        use_container_width=True,
        height=min(600, len(final_display) * 35 + 38)
    )
    
    # Export Section
    st.markdown('<div class="section-header">📥 Export & Download Options</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export filtered results
        if export_format == "CSV":
            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            filename = f"Macro_Analysis_{'_'.join(map(str, selected_chaves[:3]))}.csv"
            st.download_button(
                "📊 Download Complete Analysis (CSV)",
                csv_data,
                filename,
                "text/csv",
                help="Download all filtered data"
            )
        else:
            from io import BytesIO
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Assessor_Summary', index=False)
                top_performers.to_excel(writer, sheet_name='Top_Performers', index=False)
                
                # Add metadata sheet
                metadata = pd.DataFrame({
                    'Parameter': ['Analysis Date', 'Selected Periods', 'Total Assessors', 'Min Revenue Filter', 'Min Transactions Filter'],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        ', '.join(selected_chaves[:5]) + ('...' if len(selected_chaves) > 5 else ''),
                        len(summary_df),
                        f"R$ {min_revenue:,.2f}",
                        min_transactions
                    ]
                })
                metadata.to_excel(writer, sheet_name='Analysis_Metadata', index=False)
            
            filename = f"Macro_Analysis_{'_'.join(map(str, selected_chaves[:3]))}.xlsx"
            st.download_button(
                "📊 Download Complete Analysis (Excel)",
                buffer.getvalue(),
                filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download comprehensive Excel report with multiple sheets"
            )
    
    with col2:
        # Export top performers only
        top_20_performers = summary_df.head(20)
        csv_top = top_20_performers.to_csv(index=False).encode('utf-8')
        st.download_button(
            "🏆 Download Top 20 Performers (CSV)",
            csv_top,
            f"Top_20_Performers_{'_'.join(map(str, selected_chaves[:3]))}.csv",
            "text/csv",
            help="Download top 20 performers data"
        )
    
    with col3:
        # Export performance summary
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Assessors',
                'Total Revenue',
                'Average Revenue per Assessor', 
                'Total Company Profit',
                'Average Profit Margin',
                'Total Transactions',
                'Average Transactions per Assessor'
            ],
            'Value': [
                len(summary_df),
                f"R$ {summary_df['Comissão'].sum():,.2f}",
                f"R$ {summary_df['Comissão'].mean():,.2f}",
                f"R$ {summary_df['Lucro_Empresa'].sum():,.2f}",
                f"{summary_df['Profit_Margin'].mean():.2f}%",
                summary_df['Transaction_Count'].sum(),
                f"{summary_df['Transaction_Count'].mean():.1f}"
            ]
        })
        
        csv_summary = summary_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📈 Download Summary Statistics (CSV)",
            csv_summary,
            f"Summary_Stats_{'_'.join(map(str, selected_chaves[:3]))}.csv",
            "text/csv",
            help="Download key performance statistics"
        )
    
    # Additional insights
    if len(summary_df) > 0:
        st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Performance Insights")
            
            # Calculate insights
            high_performers = summary_df[summary_df['Profit_Margin'] > summary_df['Profit_Margin'].quantile(0.75)]
            low_performers = summary_df[summary_df['Profit_Margin'] < summary_df['Profit_Margin'].quantile(0.25)]
            
            insights = [
                f"🎯 **Top 25% of assessors** generate **{high_performers['Comissão'].sum() / summary_df['Comissão'].sum() * 100:.1f}%** of total revenue",
                f"📈 **Average profit margin** across all assessors is **{summary_df['Profit_Margin'].mean():.1f}%**",
                f"⚡ **Most active assessor** completed **{summary_df['Transaction_Count'].max():,}** transactions",
                f"💰 **Revenue concentration**: Top 10% handle **{summary_df.head(max(1, len(summary_df)//10))['Comissão'].sum() / summary_df['Comissão'].sum() * 100:.1f}%** of revenue"
            ]
            
            for insight in insights:
                st.markdown(insight)
        
        with col2:
            st.markdown("#### 🎯 Recommendations")
            
            recommendations = []
            
            if len(low_performers) > 0:
                recommendations.append(f"🔄 **Focus on {len(low_performers)} underperforming assessors** with profit margins below {summary_df['Profit_Margin'].quantile(0.25):.1f}%")
            
            if summary_df['Profit_Margin'].std() > 10:
                recommendations.append("📊 **High profit margin variance detected** - consider standardizing processes")
            
            top_performer_avg = summary_df.head(10)['Avg_Transaction'].mean()
            overall_avg = summary_df['Avg_Transaction'].mean()
            if top_performer_avg > overall_avg * 1.5:
                recommendations.append(f"💡 **Top performers** average **{format_currency(top_performer_avg)}** per transaction vs overall **{format_currency(overall_avg)}**")
            
            recommendations.append("🎯 **Consider** implementing best practices from top performers across the team")
            
            for recommendation in recommendations:
                st.markdown(recommendation)

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Individual Assessor Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("⚠️ Please upload the Excel file first using the Upload page.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Enhanced Filters
    with st.container():
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Individual Analysis Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sorted time periods
            chave_list = sort_chave_periods(df["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "🕐 Select Time Periods",
                chave_list,
                default=chave_list[-6:] if len(chave_list) >= 6 else chave_list,
                help="Choose time periods for individual analysis"
            )
        
        with col2:
            # Assessor selection with search
            assessor_list = sorted(df["AssessorReal"].dropna().unique())
            selected_assessor = st.selectbox(
                "👤 Select Assessor", 
                assessor_list,
                help="Choose an assessor for detailed analysis"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_chaves:
        st.warning("⚠️ Please select at least one time period to continue.")
        st.stop()
    
    if selected_chaves and selected_assessor:
        df_filtered = df[
            (df["AssessorReal"] == selected_assessor) &
            (df["Chave"].isin(selected_chaves))
        ]
        
        if df_filtered.empty:
            st.warning(f"⚠️ No data found for **{selected_assessor}** in the selected time periods.")
            st.stop()
        
        # Individual KPIs
        st.markdown(f'<div class="section-header">📊 Relatório Gerencial | TAURUS : {selected_assessor} | NOVEMBRO </div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)

        # Calculate metrics
        total_revenue = df_filtered["Receita Bruta"].sum() if "Receita Bruta" in df_filtered.columns else 0
        total_commission = df_filtered["Comissão"].sum()
        total_transactions = len(df_filtered)
        total_pix = df_filtered["Pix_Assessor"].sum()
        total_profit = df_filtered["Lucro_Empresa"].sum()
        
        with col1:
            create_metric_card("Total Revenue", total_revenue)
        with col2:
            create_metric_card("Commission", total_commission)
        with col3:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col4:
            create_metric_card("Total Pix Assessor", total_pix)
        """with col5:
            create_metric_card("Generated Profit", total_profit)"""
        
        # Performance trends
        st.markdown('<div class="section-header">📈 Performance Trends</div>', unsafe_allow_html=True)
        
        monthly_performance = df_filtered.groupby('Chave').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'Pix_Assessor': 'sum'
        }).reset_index()
        monthly_performance['Transaction_Count'] = df_filtered.groupby('Chave').size().values
        
        # Sort by date
        monthly_performance['Chave_Date'] = monthly_performance['Chave'].apply(parse_chave_to_date)
        monthly_performance = monthly_performance.sort_values('Chave_Date')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend
            fig_trend = px.line(
                monthly_performance,
                x='Chave',
                y='Comissão',
                title=f'📈 {selected_assessor} - Revenue Trend',
                markers=True
            )
            fig_trend.update_layout(xaxis_title="Period", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Transaction count trend
            fig_transactions = px.bar(
                monthly_performance,
                x='Chave',
                y='Transaction_Count',
                title=f'📊 {selected_assessor} - Transaction Volume',
                color='Transaction_Count',
                color_continuous_scale='Blues'
            )
            fig_transactions.update_layout(xaxis_title="Period", yaxis_title="Number of Transactions")
            st.plotly_chart(fig_transactions, use_container_width=True)
        
        # Category analysis
        st.markdown('<div class="section-header">🎯 Category Performance</div>', unsafe_allow_html=True)
        
        financial_cols = ["Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        if "Receita Bruta" in df_filtered.columns:
            financial_cols.insert(0, "Receita Bruta")
        
        category_summary = df_filtered.groupby("Categoria")[financial_cols].sum().reset_index()
        category_summary['Transaction_Count'] = df_filtered.groupby("Categoria").size().values
        category_summary['Avg_Transaction'] = category_summary['Comissão'] / category_summary['Transaction_Count']
        category_summary['Profit_Margin'] = (category_summary['Lucro_Empresa'] / category_summary['Comissão']) * 100
        
        # Sort by revenue
        category_summary = category_summary.sort_values('Comissão', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Category Performance Summary")
            
            # Format display
            display_category = category_summary.copy()
            for col in financial_cols:
                if col in display_category.columns:
                    display_category[f"{col}_Formatted"] = display_category[col].apply(format_currency)
            
            display_category['Avg_Transaction_Formatted'] = display_category['Avg_Transaction'].apply(format_currency)
            display_category['Profit_Margin_Formatted'] = display_category['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
            
            # Select display columns
            display_cols = ['Categoria', 'Transaction_Count']
            for col in financial_cols:
                if f"{col}_Formatted" in display_category.columns:
                    display_cols.append(f"{col}_Formatted")
            display_cols.extend(['Avg_Transaction_Formatted', 'Profit_Margin_Formatted'])
            
            st.dataframe(display_category[display_cols], use_container_width=True, height=400)
        
        with col2:
            # Category treemap
            fig_category = px.treemap(
                category_summary,
                path=['Categoria'],
                values='Comissão',
                title=f'🎯 {selected_assessor} - Revenue by Category',
                color='Profit_Margin',
                color_continuous_scale='RdYlGn',
                hover_data=['Transaction_Count', 'Avg_Transaction']
            )
            fig_category.update_traces(textinfo="label+percent root")
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Detailed transaction analysis
        st.markdown('<div class="section-header">🔍 Transaction Details</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Recent transactions
            st.markdown("#### 📅 Recent Transactions")
            recent_transactions = df_filtered.nlargest(10, 'Comissão')[
                ['Chave', 'Categoria', 'Comissão', 'Lucro_Empresa']
            ].copy()
            
            recent_transactions['Comissão'] = recent_transactions['Comissão'].apply(format_currency)
            recent_transactions['Lucro_Empresa'] = recent_transactions['Lucro_Empresa'].apply(format_currency)
            
            st.dataframe(recent_transactions, use_container_width=True)
        
        with col2:
            # Performance metrics
            st.markdown("#### 📊 Key Metrics")
            
            avg_transaction = df_filtered['Comissão'].mean()
            profit_margin = (total_profit / total_commission * 100) if total_commission > 0 else 0
            
            # Compare with overall performance
            overall_avg = df['Comissão'].mean()
            overall_margin = (df['Lucro_Empresa'].sum() / df['Comissão'].sum() * 100) if df['Comissão'].sum() > 0 else 0
            
            metrics_comparison = pd.DataFrame({
                'Metric': [
                    'Average Transaction Value',
                    'Profit Margin (%)',
                    'Total Revenue',
                    'Total Transactions',
                    'Revenue per Transaction'
                ],
                f'{selected_assessor}': [
                    format_currency(avg_transaction),
                    f"{profit_margin:.1f}%",
                    format_currency(total_commission),
                    f"{total_transactions:,}",
                    format_currency(avg_transaction)
                ],
                'Company Average': [
                    format_currency(overall_avg),
                    f"{overall_margin:.1f}%",
                    format_currency(df['Comissão'].mean()),
                    f"{len(df) // df['AssessorReal'].nunique():.0f}",
                    format_currency(overall_avg)
                ]
            })
            
            st.dataframe(metrics_comparison, use_container_width=True)
        
        # Export section
        st.markdown('<div class="section-header">📥 Export Individual Report</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Category summary export
            csv_summary = category_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Category Summary (CSV)",
                csv_summary,
                f"{selected_assessor}_Category_Summary.csv",
                "text/csv",
                help="Download category performance summary"
            )
        
        with col2:
            # Complete Excel report
            from io import BytesIO
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                
                # --- MODIFICATION START ---
                # Define the exact columns requested by the user
                report_columns_requested = [
                    "Data Receita", "Conta", "Cliente", "Produto", "Ativo", 
                    "Receita Bruta", "Receita Líquida", "Comissão", 
                    "Tributo_Retido", "Repasse", "Pix_Assessor"
                ]
                
                # Filter this list to include only columns that actually exist in the dataframe
                report_cols_available = [col for col in report_columns_requested if col in df_filtered.columns]
                
                # Create the 'Transactions' sheet with only the available requested columns
                if report_cols_available:
                    df_filtered[report_cols_available].to_excel(writer, sheet_name='Transactions', index=False)
                else:
                    # If no columns match, create a sheet with a message
                    pd.DataFrame({"Message": ["None of the requested columns were found in the data."]}).to_excel(writer, sheet_name='Transactions', index=False)
                # --- MODIFICATION END ---

                # Add the other sheets as before
                category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
                monthly_performance.to_excel(writer, sheet_name='Monthly_Performance', index=False)
                
                # Summary sheet
                summary_data = pd.DataFrame({
                    'Metric': [
                        'Assessor Name',
                        'Analysis Period',
                        'Total Revenue',
                        'Total Commission',
                        'Total Transactions',
                        'Average Transaction',
                        'Profit Margin',
                        'Total Profit Generated'
                    ],
                    'Value': [
                        selected_assessor,
                        f"{min(selected_chaves)} to {max(selected_chaves)}",
                        format_currency(total_revenue),
                        format_currency(total_commission),
                        total_transactions,
                        format_currency(avg_transaction),
                        f"{profit_margin:.1f}%",
                        format_currency(total_profit)
                    ]
                })
                summary_data.to_excel(writer, sheet_name='Executive_Summary', index=False)
            
            st.download_button(
                "📋 Download Complete Report (Excel)",
                buffer.getvalue(),
                f"{selected_assessor}_Complete_Report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download comprehensive Excel report with multiple sheets"
            )
        
        with col3:
            # Performance summary
            performance_summary = pd.DataFrame({
                'KPI': [
                    'Total Revenue',
                    'Total Commission', 
                    'Total Transactions',
                    'Average Transaction',
                    'Profit Margin',
                    'Total Profit'
                ],
                'Value': [
                    format_currency(total_revenue),
                    format_currency(total_commission),
                    total_transactions,
                    format_currency(avg_transaction),
                    f"{profit_margin:.1f}%",
                    format_currency(total_profit)
                ]
            })
            
            csv_perf = performance_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🎯 Download KPI Summary (CSV)",
                csv_perf,
                f"{selected_assessor}_KPI_Summary.csv",
                "text/csv",
                help="Download key performance indicators"
            )

# --- PERFORMANCE ANALYTICS ---
elif page == "📈 Performance Analytics":
    st.markdown('<h1 class="main-header">📈 Advanced Performance Analytics</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("⚠️ Please upload the Excel file first using the Upload page.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Analytics options
    analysis_type = st.selectbox(
        "📊 Select Analysis Type",
        ["Trend Analysis", "Comparative Analysis", "Seasonal Analysis", "Growth Analysis"],
        help="Choose the type of advanced analysis to perform"
    )
    
    if analysis_type == "Trend Analysis":
        st.markdown('<div class="section-header">📈 Revenue and Profit Trends</div>', unsafe_allow_html=True)
        
        # Time series analysis
        monthly_data = df.groupby('Chave').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'Pix_Assessor': 'sum',
            'AssessorReal': 'nunique'
        }).reset_index()
        
        monthly_data['Chave_Date'] = monthly_data['Chave'].apply(parse_chave_to_date)
        monthly_data = monthly_data.sort_values('Chave_Date')
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Profit Trend', 'Pix Assessor', 'Active Assessors'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Chave'], y=monthly_data['Comissão'], 
                      mode='lines+markers', name='Revenue', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # Profit trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Chave'], y=monthly_data['Lucro_Empresa'], 
                      mode='lines+markers', name='Profit', line=dict(color='green')),
            row=1, col=2
        )
        
        # Pix trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Chave'], y=monthly_data['Pix_Assessor'], 
                      mode='lines+markers', name='Pix', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Active assessors
        fig.add_trace(
            go.Scatter(x=monthly_data['Chave'], y=monthly_data['AssessorReal'], 
                      mode='lines+markers', name='Assessors', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="📊 Comprehensive Trend Analysis", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth rate analysis
        st.markdown("#### 📊 Growth Rate Analysis")
        
        monthly_data['Revenue_Growth'] = monthly_data['Comissão'].pct_change() * 100
        monthly_data['Profit_Growth'] = monthly_data['Lucro_Empresa'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_growth = px.bar(
                monthly_data.dropna(),
                x='Chave',
                y='Revenue_Growth',
                title='📈 Month-over-Month Revenue Growth (%)',
                color='Revenue_Growth',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            # Display growth statistics
            st.markdown("##### 📊 Growth Statistics")
            avg_growth = monthly_data['Revenue_Growth'].mean()
            max_growth = monthly_data['Revenue_Growth'].max()
            min_growth = monthly_data['Revenue_Growth'].min()
            
            growth_stats = pd.DataFrame({
                'Metric': [
                    'Average Monthly Growth',
                    'Highest Monthly Growth',
                    'Lowest Monthly Growth',
                    'Growth Volatility (Std Dev)'
                ],
                'Value': [
                    f"{avg_growth:.1f}%" if not pd.isna(avg_growth) else "N/A",
                    f"{max_growth:.1f}%" if not pd.isna(max_growth) else "N/A",
                    f"{min_growth:.1f}%" if not pd.isna(min_growth) else "N/A",
                    f"{monthly_data['Revenue_Growth'].std():.1f}%" if not pd.isna(monthly_data['Revenue_Growth'].std()) else "N/A"
                ]
            })
            
            st.dataframe(growth_stats, use_container_width=True)
    
    elif analysis_type == "Comparative Analysis":
        st.markdown('<div class="section-header">🔍 Assessor Comparison</div>', unsafe_allow_html=True)
        
        # Top assessors selector
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Select top N assessors", 3, 20, 10)
        with col2:
            comparison_metric = st.selectbox("Compare by", ["Revenue", "Profit", "Profit Margin", "Transactions"])
        
        # Get top assessors
        metric_mapping = {
            "Revenue": "Comissão",
            "Profit": "Lucro_Empresa", 
            "Transactions": "Transaction_Count"
        }
        
        if comparison_metric == "Profit Margin":
            # Calculate profit margin for each assessor
            assessor_summary = df.groupby('AssessorReal').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum'
            })
            assessor_summary['Profit_Margin'] = (assessor_summary['Lucro_Empresa'] / assessor_summary['Comissão']) * 100
            top_assessors = assessor_summary.nlargest(top_n, 'Profit_Margin').index
        else:
            if comparison_metric == "Transactions":
                top_assessors = df.groupby('AssessorReal').size().nlargest(top_n).index
            else:
                top_assessors = df.groupby('AssessorReal')[metric_mapping[comparison_metric]].sum().nlargest(top_n).index
        
        df_top = df[df['AssessorReal'].isin(top_assessors)]
        
        # Comparison metrics
        comparison_data = df_top.groupby('AssessorReal').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'Pix_Assessor': 'sum',
            'Chave': 'count'
        }).rename(columns={'Chave': 'Transaction_Count'})
        
        comparison_data['Avg_Transaction'] = comparison_data['Comissão'] / comparison_data['Transaction_Count']
        comparison_data['Profit_Margin'] = (comparison_data['Lucro_Empresa'] / comparison_data['Comissão']) * 100
        
        # Radar chart for top 5
        st.markdown(f"#### 🎯 Top {min(5, len(top_assessors))} Assessors Performance Radar")
        
        fig_radar = go.Figure()
        
        for i, assessor in enumerate(list(top_assessors)[:5]):
            assessor_data = comparison_data.loc[assessor]
            
            # Normalize values for radar chart (0-100 scale)
            metrics = ['Comissão', 'Lucro_Empresa', 'Pix_Assessor', 'Avg_Transaction', 'Profit_Margin']
            normalized_values = []
            
            for metric in metrics:
                if metric == 'Profit_Margin':
                    # For profit margin, use actual percentage but cap at 100
                    normalized_values.append(min(100, max(0, assessor_data[metric])))
                else:
                    # For other metrics, normalize to 0-100 scale
                    max_val = comparison_data[metric].max()
                    normalized_values.append((assessor_data[metric] / max_val) * 100)
            
            assessor_display_name = assessor[:15] + '...' if len(assessor) > 15 else assessor
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=['Revenue', 'Profit', 'Pix', 'Avg Transaction', 'Profit Margin (%)'],
                fill='toself',
                name=assessor_display_name,
                opacity=0.6
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"Performance Comparison - Top {min(5, len(top_assessors))} Assessors"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("#### 📊 Detailed Comparison")
        comparison_display = comparison_data.copy()
        
        # Format for display
        comparison_display['Comissão'] = comparison_display['Comissão'].apply(format_currency)
        comparison_display['Lucro_Empresa'] = comparison_display['Lucro_Empresa'].apply(format_currency)
        comparison_display['Pix_Assessor'] = comparison_display['Pix_Assessor'].apply(format_currency)
        comparison_display['Avg_Transaction'] = comparison_display['Avg_Transaction'].apply(format_currency)
        comparison_display['Profit_Margin'] = comparison_display['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(comparison_display, use_container_width=True)
    
    elif analysis_type == "Seasonal Analysis":
        st.markdown('<div class="section-header">🌿 Seasonal Performance Analysis</div>', unsafe_allow_html=True)
        
        # Add month extraction for seasonal analysis
        df_seasonal = df.copy()
        df_seasonal['Month'] = df_seasonal['Chave_Date'].dt.month
        df_seasonal['Month_Name'] = df_seasonal['Chave_Date'].dt.strftime('%B')
        
        # Monthly aggregation
        monthly_performance = df_seasonal.groupby(['Month', 'Month_Name']).agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'AssessorReal': 'nunique'
        }).reset_index().sort_values('Month')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly revenue pattern
            fig_seasonal = px.bar(
                monthly_performance,
                x='Month_Name',
                y='Comissão',
                title='📊 Revenue by Month (Seasonal Pattern)',
                color='Comissão',
                color_continuous_scale='Blues'
            )
            fig_seasonal.update_layout(xaxis_title="Month", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            # Monthly profit pattern
            fig_profit_seasonal = px.line(
                monthly_performance,
                x='Month_Name',
                y='Lucro_Empresa',
                title='📈 Profit by Month (Seasonal Trend)',
                markers=True,
                line_shape='spline'
            )
            fig_profit_seasonal.update_layout(xaxis_title="Month", yaxis_title="Profit (R$)")
            st.plotly_chart(fig_profit_seasonal, use_container_width=True)
        
        # Seasonal insights
        st.markdown("#### 💡 Seasonal Insights")
        
        best_month = monthly_performance.loc[monthly_performance['Comissão'].idxmax(), 'Month_Name']
        worst_month = monthly_performance.loc[monthly_performance['Comissão'].idxmin(), 'Month_Name']
        seasonal_variance = monthly_performance['Comissão'].std() / monthly_performance['Comissão'].mean() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏆 Best Performing Month", 
                best_month, 
                f"{format_currency(monthly_performance['Comissão'].max())}"
            )
        
        with col2:
            st.metric(
                "📉 Lowest Performing Month", 
                worst_month, 
                f"{format_currency(monthly_performance['Comissão'].min())}"
            )
        
        with col3:
            st.metric(
                "📊 Seasonal Variance", 
                f"{seasonal_variance:.1f}%",
                help="Coefficient of variation - higher values indicate more seasonality"
            )
    
    elif analysis_type == "Growth Analysis":
        st.markdown('<div class="section-header">📊 Growth Analysis</div>', unsafe_allow_html=True)
        
        # Time period selection for growth analysis
        chave_list = sort_chave_periods(df["Chave"].dropna().unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_period = st.selectbox(
                "📅 Select Baseline Period",
                chave_list,
                help="Choose the baseline period for growth comparison"
            )
        
        with col2:
            comparison_periods = st.multiselect(
                "📅 Select Comparison Periods",
                [p for p in chave_list if p != baseline_period],
                default=[p for p in chave_list if p != baseline_period][-3:] if len(chave_list) > 1 else [],
                help="Choose periods to compare against the baseline"
            )
        
        if baseline_period and comparison_periods:
            # Calculate growth metrics
            baseline_data = df[df['Chave'] == baseline_period].groupby('AssessorReal').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum'
            }).add_suffix('_baseline')
            
            growth_results = []
            
            for period in comparison_periods:
                period_data = df[df['Chave'] == period].groupby('AssessorReal').agg({
                    'Comissão': 'sum',
                    'Lucro_Empresa': 'sum'
                }).add_suffix(f'_{period}')
                
                # Merge with baseline
                growth_df = baseline_data.merge(period_data, left_index=True, right_index=True, how='outer').fillna(0)
                
                # Calculate growth rates
                growth_df[f'Revenue_Growth_{period}'] = calculate_growth_rate(
                    growth_df[f'Comissão_{period}'], 
                    growth_df['Comissão_baseline']
                )
                
                growth_df[f'Profit_Growth_{period}'] = calculate_growth_rate(
                    growth_df[f'Lucro_Empresa_{period}'], 
                    growth_df['Lucro_Empresa_baseline']
                )
                
                growth_results.append({
                    'Period': period,
                    'Avg_Revenue_Growth': growth_df[f'Revenue_Growth_{period}'].mean(),
                    'Avg_Profit_Growth': growth_df[f'Profit_Growth_{period}'].mean(),
                    'Assessors_with_Growth': (growth_df[f'Revenue_Growth_{period}'] > 0).sum(),
                    'Total_Assessors': len(growth_df)
                })
            
            growth_summary = pd.DataFrame(growth_results)
            
            # Growth visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig_growth_trend = px.bar(
                    growth_summary,
                    x='Period',
                    y='Avg_Revenue_Growth',
                    title=f'📈 Average Revenue Growth vs {baseline_period}',
                    color='Avg_Revenue_Growth',
                    color_continuous_scale='RdYlGn'
                )
                fig_growth_trend.update_layout(yaxis_title="Average Growth (%)")
                st.plotly_chart(fig_growth_trend, use_container_width=True)
            
            with col2:
                fig_assessor_growth = px.bar(
                    growth_summary,
                    x='Period',
                    y='Assessors_with_Growth',
                    title=f'👥 Assessors with Positive Growth vs {baseline_period}',
                    color='Assessors_with_Growth',
                    color_continuous_scale='Blues'
                )
                fig_assessor_growth.update_layout(yaxis_title="Number of Assessors")
                st.plotly_chart(fig_assessor_growth, use_container_width=True)
            
            # Growth summary table
            st.markdown("#### 📊 Growth Summary")
            
            display_growth = growth_summary.copy()
            display_growth['Avg_Revenue_Growth'] = display_growth['Avg_Revenue_Growth'].apply(lambda x: f"{x:.1f}%")
            display_growth['Avg_Profit_Growth'] = display_growth['Avg_Profit_Growth'].apply(lambda x: f"{x:.1f}%")
            display_growth['Growth_Rate'] = (display_growth['Assessors_with_Growth'] / display_growth['Total_Assessors'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(
                display_growth[['Period', 'Avg_Revenue_Growth', 'Avg_Profit_Growth', 'Assessors_with_Growth', 'Total_Assessors', 'Growth_Rate']].rename(columns={
                    'Period': '📅 Period',
                    'Avg_Revenue_Growth': '📈 Avg Revenue Growth',
                    'Avg_Profit_Growth': '💰 Avg Profit Growth',
                    'Assessors_with_Growth': '📊 Assessors Growing',
                    'Total_Assessors': '👥 Total Assessors',
                    'Growth_Rate': '📊 Success Rate'
                }),
                use_container_width=True
            )
            
            # Top and bottom performers in growth
            if len(comparison_periods) > 0:
                latest_period = comparison_periods[-1]
                latest_growth_df = baseline_data.merge(
                    df[df['Chave'] == latest_period].groupby('AssessorReal').agg({
                        'Comissão': 'sum',
                        'Lucro_Empresa': 'sum'
                    }).add_suffix(f'_{latest_period}'),
                    left_index=True, right_index=True, how='outer'
                ).fillna(0)
                
                latest_growth_df['Revenue_Growth'] = latest_growth_df.apply(
                    lambda row: calculate_growth_rate(row[f'Comissão_{latest_period}'], row['Comissão_baseline']), axis=1
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### 🚀 Top Growth Performers ({latest_period} vs {baseline_period})")
                    top_growth = latest_growth_df.nlargest(10, 'Revenue_Growth')[['Revenue_Growth']].round(1)
                    top_growth['Revenue_Growth'] = top_growth['Revenue_Growth'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(top_growth.rename(columns={'Revenue_Growth': '📈 Growth %'}))
                
                with col2:
                    st.markdown(f"#### 📉 Bottom Growth Performers ({latest_period} vs {baseline_period})")
                    bottom_growth = latest_growth_df.nsmallest(10, 'Revenue_Growth')[['Revenue_Growth']].round(1)
                    bottom_growth['Revenue_Growth'] = bottom_growth['Revenue_Growth'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(bottom_growth.rename(columns={'Revenue_Growth': '📉 Growth %'}))
        
        else:
            st.info("Please select a baseline period and at least one comparison period to perform growth analysis.")
    
    # Export analytics results
    st.markdown('<div class="section-header">📥 Export Analytics Results</div>', unsafe_allow_html=True)
    
    if st.button("📊 Generate Analytics Report"):
        with st.spinner("Generating comprehensive analytics report..."):
            from io import BytesIO
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Overall summary
                overall_summary = df.groupby('AssessorReal').agg({
                    'Comissão': ['sum', 'mean', 'count'],
                    'Lucro_Empresa': 'sum',
                    'Pix_Assessor': 'sum'
                }).round(2)
                overall_summary.columns = [f"{col[0]}_{col[1]}" for col in overall_summary.columns]
                overall_summary['Profit_Margin'] = (overall_summary['Lucro_Empresa_sum'] / overall_summary['Comissão_sum'] * 100).round(2)
                overall_summary.to_excel(writer, sheet_name='Overall_Summary')
                
                # Monthly trends
                if analysis_type == "Trend Analysis":
                    monthly_data.to_excel(writer, sheet_name='Monthly_Trends', index=False)
                
                # Seasonal analysis
                if 'monthly_performance' in locals():
                    monthly_performance.to_excel(writer, sheet_name='Seasonal_Analysis', index=False)
                
                # Growth analysis
                if 'growth_summary' in locals():
                    growth_summary.to_excel(writer, sheet_name='Growth_Analysis', index=False)
                
                # Metadata
                metadata = pd.DataFrame({
                    'Parameter': [
                        'Report Generated',
                        'Analysis Type',
                        'Total Assessors',
                        'Total Periods',
                        'Total Revenue',
                        'Total Profit'
                    ],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        analysis_type,
                        df['AssessorReal'].nunique(),
                        df['Chave'].nunique(),
                        format_currency(df['Comissão'].sum()),
                        format_currency(df['Lucro_Empresa'].sum())
                    ]
                })
                metadata.to_excel(writer, sheet_name='Report_Metadata', index=False)
        
        st.download_button(
            "📋 Download Analytics Report (Excel)",
            buffer.getvalue(),
            f"Analytics_Report_{analysis_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download comprehensive analytics report with all analysis results"
        )
        
        st.success("✅ Analytics report generated successfully!")

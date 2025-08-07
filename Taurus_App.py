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
                    if 'Data Receita' in df_taurus.columns:
                        date_range = f"{df_taurus['Data Receita'].min()} to {df_taurus['Data Receita'].max()}"
                        st.info(f"📅 **Date Range:** {date_range}")
                    
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

    # Time Period Filter (same as Macro View)
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

        total_revenue = df_filtered["Comissão"].sum()
        total_pix = df_filtered["Pix_Assessor"].sum()
        total_profit = df_filtered["Lucro_Empresa"].sum()
        avg_transaction = df_filtered["Comissão"].mean()
        active_assessors = df_filtered["AssessorReal"].nunique()

        with col1:
            st.metric("Total Revenue", format_currency(total_revenue))
        with col2:
            st.metric("Total Pix Assessor", format_currency(total_pix))
        with col3:
            st.metric("Company Profit", format_currency(total_profit))
        with col4:
            st.metric("Avg Transaction", format_currency(avg_transaction))
        with col5:
            st.metric("Active Assessors", active_assessors)

        # Charts Row 1
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
            fig_revenue.update_layout(xaxis_title="Period", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig_revenue, use_container_width=True)

        with col2:
            # Top Assessors
            top_assessors = df_filtered.groupby('AssessorReal')['Comissão'].sum().nlargest(10)
            fig_assessors = px.bar(
                x=top_assessors.values,
                y=top_assessors.index,
                orientation='h',
                title='🏆 Top 10 Assessors by Revenue'
            )
            fig_assessors.update_layout(xaxis_title="Revenue (R$)", yaxis_title="Assessor")
            st.plotly_chart(fig_assessors, use_container_width=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            # Category Distribution
            category_dist = df_filtered.groupby('Categoria')['Comissão'].sum()
            fig_pie = px.pie(
                values=category_dist.values,
                names=category_dist.index,
                title='🎯 Revenue Distribution by Category'
            )
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
            st.plotly_chart(fig_margin, use_container_width=True)
    else:
        st.info("Please select at least one time period to display the dashboard.")

# --- MACRO VIEW PAGE ---
elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Macro View - Assessor Performance</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
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
        min_revenue = st.number_input("💰 Minimum Revenue Filter", min_value=0.0, value=0.0, step=1000.0)
    
    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]
        
        # Summary calculations
        financial_cols = ["Receita Bruta" , "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        summary_df = df_filtered.groupby("AssessorReal")[financial_cols].sum().reset_index()
        
        # Add calculated metrics
        summary_df['Transaction_Count'] = df_filtered.groupby("AssessorReal").size().values
        summary_df['Avg_Transaction'] = summary_df['Comissão'] / summary_df['Transaction_Count']
        summary_df['Profit_Margin'] = (summary_df['Lucro_Empresa'] / summary_df['Comissão']) * 100
        
        # Filter by minimum revenue
        summary_df = summary_df[summary_df['Comissão'] >= min_revenue]
        summary_df = summary_df.sort_values("Comissão", ascending=False)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessors", len(summary_df))
        with col2:
            st.metric("Total Revenue", format_currency(summary_df['Comissão'].sum()))
        with col3:
            st.metric("Avg Revenue/Assessor", format_currency(summary_df['Comissão'].mean()))
        
        # Enhanced table with formatting
        st.markdown("### 📋 Assessor Performance Summary")
        
        # Format the display dataframe
        display_df = summary_df.copy()
        for col in financial_cols:
            display_df[col] = display_df[col].apply(lambda x: f"R$ {x:,.2f}")
        display_df['Avg_Transaction'] = display_df['Avg_Transaction'].apply(lambda x: f"R$ {x:,.2f}")
        display_df['Profit_Margin'] = display_df['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers
            top_10 = summary_df.head(10)
            fig_top = px.bar(
                top_10,
                x='Comissão',
                y='AssessorReal',
                orientation='h',
                title='🏆 Top 10 Performers',
                color='Profit_Margin',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Scatter plot: Revenue vs Profit Margin
            fig_scatter = px.scatter(
                summary_df,
                x='Comissão',
                y='Profit_Margin',
                size='Transaction_Count',
                hover_data=['AssessorReal'],
                title='💰 Revenue vs Profit Margin',
                labels={'Comissão': 'Revenue (R$)', 'Profit_Margin': 'Profit Margin (%)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Complete Summary (CSV)",
                csv_data,
                f"Macro_Summary_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )
        
        with col2:
            # Top performers only
            top_performers = summary_df.head(20)
            csv_top = top_performers.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🏆 Download Top 20 Performers (CSV)",
                csv_top,
                f"Top_Performers_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Individual Assessor Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
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
            
            col1, col2, col3, col4, col5 = st.columns(5)

            total_revenue = df_filtered["Receita Bruta"].sum()
            PagoTaurus = df_filtered["Comissão"].sum()
            total_transactions = len(df_filtered)
            total_pix = df_filtered["Pix_Assessor"].sum()
            total_profit = df_filtered["Lucro_Empresa"].sum()
            
            with col1:
                st.metric("Total Revenue", format_currency(total_revenue))
            with col2:
                st.metric("Comissão", format_currency(PagoTaurus))
            with col3:
                st.metric("Total Transactions", total_transactions)
            with col4:
                st.metric("Total Pix Assessor", format_currency(total_pix))
            with col5:
                st.metric("Generated Profit", format_currency(total_profit))
            
            # Performance over time
            monthly_performance = df_filtered.groupby('Chave').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum',
                'Chave': 'count'
            }).rename(columns={'Chave': 'Transaction_Count'})
            monthly_performance = monthly_performance.reset_index() 
           
            # Category breakdown
            financial_cols = ["Receita Bruta" , "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
            category_summary = df_filtered.groupby("Categoria")[financial_cols].sum().reset_index()
            
            # Add transaction count per category
            category_summary['Transaction_Count'] = df_filtered.groupby("Categoria").size().values
            
            st.markdown("### 📋 Performance by Category")
            st.dataframe(category_summary.round(2), use_container_width=True)
            
            # Category visualization
            fig_category = px.treemap(
                category_summary,
                path=['Categoria'],
                values='Comissão',
                title=f'🎯 {selected_assessor} - Revenue by Category',
                color='Lucro_Empresa',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Download section
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Category summary
                csv_summary = category_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Download Category Summary",
                    csv_summary,
                    f"{selected_assessor}_Category_Summary.csv",
                    "text/csv"
                )
            
            with col2:
                # Detailed transactions
                detailed_cols = [
                    "Data Receita", "Conta", "Cliente", "AssessorReal" , "Categoria", "Produto",
                    "Comissão", "Receita Assessor", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa" , "Chave"
                ]
                available_cols = [col for col in detailed_cols if col in df_filtered.columns]
                
                if available_cols:
                    from io import BytesIO
                    buffer = BytesIO()
                    
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_filtered[available_cols].to_excel(writer, sheet_name='Taurus', index=False)
                        category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
                        monthly_performance.to_excel(writer, sheet_name='Monthly_Performance', index=False)
                    
                    st.download_button(
                        "📋 Download Complete Report (Excel)",
                        buffer.getvalue(),
                        f"{selected_assessor}_Complete_Report.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                # Performance summary
                performance_summary = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Total Transactions', 'Avg Transaction', 'Total Profit'],
                    'Value': [total_revenue, total_transactions, Avg_Transaction, total_profit]
                })
                csv_perf = performance_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🎯 Download Performance Summary",
                    csv_perf,
                    f"{selected_assessor}_Performance_Summary.csv",
                    "text/csv"
                )

# --- PERFORMANCE ANALYTICS ---
elif page == "📈 Performance Analytics":
    st.markdown('<h1 class="main-header">📈 Advanced Performance Analytics</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Analytics options
    analysis_type = st.selectbox(
        "📊 Select Analysis Type",
        ["Trend Analysis", "Comparative Analysis", "Seasonal Analysis", "Growth Analysis"]
    )
    
    if analysis_type == "Trend Analysis":
        st.markdown("### 📈 Revenue and Profit Trends")
        
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
                      mode='lines+markers', name='Revenue'),
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
        
        fig.update_layout(height=600, title_text="📊 Comprehensive Trend Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Comparative Analysis":
        st.markdown("### 🔍 Assessor Comparison")
        
        # Top assessors selector
        top_n = st.slider("Select top N assessors", 3, 20, 10)
        
        top_assessors = df.groupby('AssessorReal')['Comissão'].sum().nlargest(top_n).index
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
        
        # Radar chart
        fig_radar = go.Figure()
        
        for assessor in top_assessors[:5]:  # Show top 5 in radar
            assessor_data = comparison_data.loc[assessor]
            
            # Normalize values for radar chart
            metrics = ['Comissão', 'Lucro_Empresa', 'Pix_Assessor', 'Avg_Transaction', 'Profit_Margin']
            normalized_values = []
            
            for metric in metrics:
                max_val = comparison_data[metric].max()
                normalized_values.append((assessor_data[metric] / max_val) * 100)
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=metrics,
                fill='toself',
                name=assessor
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="🎯 Top 5 Assessors Performance Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    

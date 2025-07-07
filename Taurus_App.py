import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(
    page_title="Taurus Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Enhanced Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #007bff;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #721c24;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
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
    "📈 Performance Analytics",
    "🎯 Goal Tracking",
    "💰 Profit Center",
    "📋 Reports",
    "🔮 Predictive Analytics",
    "🏆 Leaderboard",
    "📊 Real-time Monitoring"
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
    if pd.isna(value):
        return "R$ 0.00"
    return f"R$ {value:,.2f}"

def calculate_growth_rate(current, previous):
    """Calculate growth rate between two values"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def create_gauge_chart(value, title, max_val=None, target=None):
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
                'value': target if target else max_val*0.9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def calculate_percentile_rank(df, column, value):
    """Calculate percentile rank of a value in a column"""
    return (df[column] < value).sum() / len(df) * 100

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def create_advanced_metrics(df):
    """Create advanced performance metrics"""
    metrics = {}
    
    # Efficiency metrics
    metrics['Revenue_per_Transaction'] = df['Comissão'].sum() / len(df)
    metrics['Profit_Margin'] = (df['Lucro_Empresa'].sum() / df['Comissão'].sum()) * 100
    metrics['Assessor_Productivity'] = df['Comissão'].sum() / df['AssessorReal'].nunique()
    
    # Consistency metrics
    monthly_revenue = df.groupby('Chave')['Comissão'].sum()
    metrics['Revenue_Volatility'] = monthly_revenue.std() / monthly_revenue.mean() * 100
    
    # Growth metrics
    if len(monthly_revenue) > 1:
        metrics['MoM_Growth'] = calculate_growth_rate(monthly_revenue.iloc[-1], monthly_revenue.iloc[-2])
    else:
        metrics['MoM_Growth'] = 0
    
    return metrics

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
        st.info("📋 **Required Columns:**\n- Chave\n- AssessorReal\n- Categoria\n- Comissão\n- Tributo_Retido\n- Pix_Assessor\n- Lucro_Empresa\n- Data Receita (optional)")
    
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
                # Enhanced data preprocessing
                df_taurus['Chave_Date'] = df_taurus['Chave'].apply(parse_chave_to_date)
                df_taurus['Month_Year'] = df_taurus['Chave_Date'].dt.strftime('%Y-%m')
                
                # Parse Data Receita if available
                if 'Data Receita' in df_taurus.columns:
                    df_taurus['Data Receita'] = pd.to_datetime(df_taurus['Data Receita'], errors='coerce')
                    df_taurus['Day_of_Week'] = df_taurus['Data Receita'].dt.day_name()
                    df_taurus['Week_of_Year'] = df_taurus['Data Receita'].dt.isocalendar().week
                    df_taurus['Quarter'] = df_taurus['Data Receita'].dt.quarter
                
                # Calculate additional metrics
                df_taurus['Profit_Margin'] = (df_taurus['Lucro_Empresa'] / df_taurus['Comissão']) * 100
                df_taurus['Net_Assessor_Payment'] = df_taurus['Pix_Assessor'] - df_taurus['Tributo_Retido']
                
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
                
                # Advanced insights
                st.markdown("### 🔍 Advanced Data Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Data quality assessment
                    missing_data = df_taurus.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning("⚠️ Data Quality Issues:")
                        for col, missing_count in missing_data[missing_data > 0].items():
                            st.write(f"• {col}: {missing_count} missing values ({missing_count/len(df_taurus)*100:.1f}%)")
                    else:
                        st.success("✅ No missing data detected")
                    
                    # Outlier detection
                    outliers = detect_outliers(df_taurus, 'Comissão')
                    if len(outliers) > 0:
                        st.warning(f"⚠️ {len(outliers)} potential outliers detected in Revenue")
                    
                with col2:
                    # Date range and patterns
                    if 'Data Receita' in df_taurus.columns:
                        date_range = f"{df_taurus['Data Receita'].min().strftime('%Y-%m-%d')} to {df_taurus['Data Receita'].max().strftime('%Y-%m-%d')}"
                        st.info(f"📅 **Date Range:** {date_range}")
                        
                        # Most active day
                        most_active_day = df_taurus['Day_of_Week'].value_counts().index[0]
                        st.info(f"📈 **Most Active Day:** {most_active_day}")
                    
                    # Performance metrics
                    avg_profit_margin = df_taurus['Profit_Margin'].mean()
                    st.info(f"💰 **Avg Profit Margin:** {avg_profit_margin:.1f}%")
                    
                    # Top performer
                    top_performer = df_taurus.groupby('AssessorReal')['Comissão'].sum().idxmax()
                    st.info(f"🏆 **Top Performer:** {top_performer}")
                
                # Sample data with better formatting
                st.markdown("### 👀 Sample Data Preview")
                display_cols = ['Chave', 'AssessorReal', 'Categoria', 'Comissão', 'Pix_Assessor', 'Lucro_Empresa', 'Profit_Margin']
                available_cols = [col for col in display_cols if col in df_taurus.columns]
                sample_data = df_taurus[available_cols].head(10)
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
    
    # Time period filter
    st.sidebar.markdown("### 🕐 Time Period")
    chave_list = sorted(df["Chave"].dropna().unique())
    selected_chaves = st.sidebar.multiselect(
        "Select periods",
        chave_list,
        default=chave_list[-6:] if len(chave_list) >= 6 else chave_list  # Last 6 months
    )
    
    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]
        
        # Advanced metrics calculation
        advanced_metrics = create_advanced_metrics(df_filtered)
        
        # KPI Cards with enhanced styling
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
        
        # Second row of KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Profit Margin", f"{advanced_metrics['Profit_Margin']:.1f}%")
        with col2:
            st.metric("Revenue Volatility", f"{advanced_metrics['Revenue_Volatility']:.1f}%")
        with col3:
            st.metric("MoM Growth", f"{advanced_metrics['MoM_Growth']:.1f}%")
        with col4:
            st.metric("Avg Revenue/Assessor", format_currency(advanced_metrics['Assessor_Productivity']))
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Revenue Evolution with trend line
            monthly_revenue = df_filtered.groupby('Chave')['Comissão'].sum().reset_index()
            monthly_revenue['Chave_Date'] = monthly_revenue['Chave'].apply(parse_chave_to_date)
            monthly_revenue = monthly_revenue.sort_values('Chave_Date')
            
            fig_revenue = px.line(
                monthly_revenue, 
                x='Chave', 
                y='Comissão',
                title='📈 Revenue Evolution with Trend',
                markers=True
            )
            
            # Add trend line
            z = np.polyfit(range(len(monthly_revenue)), monthly_revenue['Comissão'], 1)
            p = np.poly1d(z)
            fig_revenue.add_trace(go.Scatter(
                x=monthly_revenue['Chave'],
                y=p(range(len(monthly_revenue))),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
            
            fig_revenue.update_layout(xaxis_title="Period", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Top Assessors with performance indicators
            top_assessors = df_filtered.groupby('AssessorReal').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum'
            }).nlargest(10, 'Comissão')
            
            top_assessors['Profit_Margin'] = (top_assessors['Lucro_Empresa'] / top_assessors['Comissão']) * 100
            
            fig_assessors = px.bar(
                x=top_assessors['Comissão'],
                y=top_assessors.index,
                orientation='h',
                title='🏆 Top 10 Assessors by Revenue',
                color=top_assessors['Profit_Margin'],
                color_continuous_scale='RdYlGn'
            )
            fig_assessors.update_layout(xaxis_title="Revenue (R$)", yaxis_title="Assessor")
            st.plotly_chart(fig_assessors, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Category Distribution
            category_dist = df_filtered.groupby('Categoria').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum'
            })
            category_dist['Profit_Margin'] = (category_dist['Lucro_Empresa'] / category_dist['Comissão']) * 100
            
            fig_pie = px.pie(
                values=category_dist['Comissão'],
                names=category_dist.index,
                title='🎯 Revenue Distribution by Category',
                hover_data=['Profit_Margin']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Profit Margin Analysis with target line
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
            
            # Add target line
            target_margin = 25  # Example target
            fig_margin.add_hline(y=target_margin, line_dash="dash", line_color="red", 
                                annotation_text="Target: 25%")
            
            st.plotly_chart(fig_margin, use_container_width=True)
        
        # Insights section
        st.markdown("### 💡 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            if advanced_metrics['MoM_Growth'] > 0:
                st.markdown('<div class="insight-box">📈 <strong>Positive Growth:</strong> Revenue is growing at {:.1f}% month-over-month</div>'.format(advanced_metrics['MoM_Growth']), unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">📉 <strong>Attention:</strong> Revenue declined by {:.1f}% last month</div>'.format(abs(advanced_metrics['MoM_Growth'])), unsafe_allow_html=True)
        
        with col2:
            if advanced_metrics['Profit_Margin'] > 20:
                st.markdown('<div class="insight-box">💰 <strong>Healthy Margins:</strong> Profit margin of {:.1f}% is above industry average</div>'.format(advanced_metrics['Profit_Margin']), unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ <strong>Margin Alert:</strong> Profit margin of {:.1f}% needs attention</div>'.format(advanced_metrics['Profit_Margin']), unsafe_allow_html=True)

# --- PERFORMANCE ANALYTICS (COMPLETED) ---
elif page == "📈 Performance Analytics":
    st.markdown('<h1 class="main-header">📈 Advanced Performance Analytics</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Analytics options
    analysis_type = st.selectbox(
        "📊 Select Analysis Type",
        ["Trend Analysis", "Comparative Analysis", "Seasonal Analysis", "Growth Analysis", "Correlation Analysis"]
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
        
        # Trend insights
        st.markdown("### 📊 Trend Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_growth = calculate_growth_rate(monthly_data['Comissão'].iloc[-1], monthly_data['Comissão'].iloc[0])
            st.metric("Overall Revenue Growth", f"{revenue_growth:.1f}%")
            
            profit_growth = calculate_growth_rate(monthly_data['Lucro_Empresa'].iloc[-1], monthly_data['Lucro_Empresa'].iloc[0])
            st.metric("Overall Profit Growth", f"{profit_growth:.1f}%")
        
        with col2:
            # Volatility analysis
            revenue_volatility = monthly_data['Comissão'].std() / monthly_data['Comissão'].mean() * 100
            st.metric("Revenue Volatility", f"{revenue_volatility:.1f}%")
            
            assessor_growth = calculate_growth_rate(monthly_data['AssessorReal'].iloc[-1], monthly_data['AssessorReal'].iloc[0])
            st.metric("Assessor Base Growth", f"{assessor_growth:.1f}%")
    
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
        
        # Performance matrix
        st.markdown("### 📊 Performance Matrix")
        
        # Create performance matrix
        fig_matrix = px.scatter(
            comparison_data,
            x='Comissão',
            y='Profit_Margin',
            size='Transaction_Count',
            hover_data=['AssessorReal', 'Avg_Transaction'],
            title='💼 Revenue vs Profit Margin Matrix',
            labels={'Comissão': 'Total Revenue (R$)', 'Profit_Margin': 'Profit Margin (%)'}
        )
        
        # Add quadrant lines
        median_revenue = comparison_data['Comissão'].median()
        median_margin = comparison_data['Profit_Margin'].median()
        
        fig_matrix.add_vline(x=median_revenue, line_dash="dash", line_color="gray", annotation_text="Median Revenue")
        fig_matrix.add_hline(y=median_margin, line_dash="dash", line_color="gray", annotation_text="Median Margin")
        
        st.plotly_chart(fig_matrix, use_container_width=True)
    
   

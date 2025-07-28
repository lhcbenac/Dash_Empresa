import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import io
import math

# Set page configuration
st.set_page_config(
    page_title="Trading Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for professional styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metric cards enhancement */
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border: 1px solid #e0e6ed;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .stMetric > label {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric > div {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    
    /* Section headers */
    .section-header {
        background: #f8fafc;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 2rem 0 1rem 0;
    }
    
    .section-header h2 {
        margin: 0;
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Statistics cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .stat-card h3 {
        color: #1e293b;
        font-size: 1.125rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0.25rem 0;
        color: #374151;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f1f5f9;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Progress indicators */
    .progress-bar {
        background: #e2e8f0;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the backtesting data"""
    try:
        # Load the CSV file with more explicit parameters
        df_raw = pd.read_csv('Backtesting2_history.csv', 
                           encoding='utf-8',
                           low_memory=False)
        
        # Show date range in raw data
        if 'Loop_Date' in df_raw.columns:
            try:
                raw_dates = pd.to_datetime(df_raw['Loop_Date'], errors='coerce')
                valid_dates = raw_dates.dropna()
                if len(valid_dates) > 0:
                    years = valid_dates.dt.year.value_counts().sort_index()
                    st.success(f"📊 Loaded {len(df_raw):,} total records spanning {len(years)} years ({min(years.index)}-{max(years.index)})")
            except Exception as date_error:
                st.warning(f"⚠️ Date parsing issue: {date_error}")
        
        # Check if Operou_Dia column exists
        if 'Operou_Dia' not in df_raw.columns:
            st.error("❌ Column 'Operou_Dia' not found in the CSV file.")
            st.write("Available columns:", df_raw.columns.tolist())
            return None
        
        # Filter only rows where Operou_Dia = True
        mask = (
            (df_raw['Operou_Dia'] == True) | 
            (df_raw['Operou_Dia'] == 'True') | 
            (df_raw['Operou_Dia'] == 1) |
            (df_raw['Operou_Dia'] == '1') |
            (df_raw['Operou_Dia'].astype(str).str.upper() == 'TRUE')
        )
        
        df = df_raw[mask].copy()
        
        if len(df) == 0:
            st.error("❌ No trading records found (Operou_Dia = True)")
            return None
        
        # Process dates
        try:
            df['Loop_Date'] = pd.to_datetime(df['Loop_Date'], errors='coerce')
            invalid_dates = df['Loop_Date'].isna().sum()
            if invalid_dates > 0:
                st.warning(f"⚠️ Removed {invalid_dates} records with invalid dates")
                df = df.dropna(subset=['Loop_Date'])
            
            # Normalize dates (remove time component)
            df['Loop_Date'] = df['Loop_Date'].dt.date
            df['Loop_Date'] = pd.to_datetime(df['Loop_Date'])
            
        except Exception as date_error:
            st.error(f"❌ Date processing error: {date_error}")
            return None
        
        # Sort by date (oldest to newest) - CRITICAL for proper cumulative calculations
        df = df.sort_values('Loop_Date', ascending=True).reset_index(drop=True)
        
        # Success message
        if len(df) > 0:
            st.success(f"✅ Successfully processed {len(df):,} trading records from {df['Loop_Date'].min().date()} to {df['Loop_Date'].max().date()}")
        
        return df
        
    except FileNotFoundError:
        st.error("❌ File 'Backtesting2_history.csv' not found. Please upload the file to the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def calculate_lot_size(gatilho_dia_value, initial_balance=50000):
    """Calculate lot size based on gatilho_dia value"""
    if pd.isna(gatilho_dia_value) or gatilho_dia_value is None or gatilho_dia_value <= 0:
        return 100
    
    try:
        gatilho_dia_value = float(gatilho_dia_value)
        lot_size = initial_balance / gatilho_dia_value
        lot_size = max(100, math.floor(lot_size / 100) * 100)
        return int(lot_size)
    except (ValueError, TypeError, ZeroDivisionError):
        return 100

def calculate_drawdown(cumulative_pnl):
    """Calculate drawdown statistics"""
    peak = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - peak
    max_drawdown = drawdown.min()
    max_drawdown_idx = drawdown.idxmin()
    peak_value = peak.iloc[max_drawdown_idx]
    
    if peak_value != 0:
        max_drawdown_pct = (max_drawdown / peak_value) * 100
    else:
        max_drawdown_pct = 0
    
    return drawdown, max_drawdown, max_drawdown_pct, peak

def create_performance_chart(df_filtered):
    """Create enhanced performance chart with area fill"""
    fig = go.Figure()
    
    # Add cumulative balance area
    fig.add_trace(go.Scatter(
        x=df_filtered['Loop_Date'],
        y=df_filtered['Cumulative_Balance'],
        mode='lines',
        name='Portfolio Balance',
        line=dict(color='#3b82f6', width=3),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.2)',
        hovertemplate='<b>%{x}</b><br>Balance: $%{y:,.2f}<br><extra></extra>'
    ))
    
    # Add initial balance reference line
    fig.add_hline(
        y=50000, 
        line_dash="dash", 
        line_color="#6b7280",
        line_width=2,
        annotation_text="Initial Capital ($50,000)",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title={
            'text': "📈 Portfolio Performance Over Time",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter', 'color': '#1f2937'}
        },
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        showlegend=False,
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12, color="#374151"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f3f4f6',
            showline=True,
            linewidth=1,
            linecolor='#e5e7eb'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f3f4f6',
            showline=True,
            linewidth=1,
            linecolor='#e5e7eb',
            tickformat='$,.0f'
        )
    )
    
    return fig

def create_drawdown_chart(df_filtered, drawdown_series):
    """Create enhanced drawdown chart"""
    fig = go.Figure()
    
    # Add drawdown area
    fig.add_trace(go.Scatter(
        x=df_filtered['Loop_Date'],
        y=drawdown_series,
        fill='tonexty',
        mode='lines',
        name='Drawdown',
        line=dict(color='#ef4444', width=2),
        fillcolor='rgba(239, 68, 68, 0.3)',
        hovertemplate='<b>%{x}</b><br>Drawdown: $%{y:,.2f}<br><extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#374151", line_width=1)
    
    fig.update_layout(
        title={
            'text': "📉 Portfolio Drawdown Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter', 'color': '#1f2937'}
        },
        xaxis_title="Date",
        yaxis_title="Drawdown ($)",
        hovermode='x unified',
        showlegend=False,
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12, color="#374151"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f3f4f6',
            showline=True,
            linewidth=1,
            linecolor='#e5e7eb'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f3f4f6',
            showline=True,
            linewidth=1,
            linecolor='#e5e7eb',
            tickformat='$,.0f'
        )
    )
    
    return fig

def export_to_excel(df):
    """Export dataframe to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    return output.getvalue()

def create_metric_card(label, value, delta=None, help_text=None):
    """Create a professional metric display"""
    delta_html = f"<div style='color: {'#10b981' if delta and float(delta.replace('$', '').replace(',', '')) > 0 else '#ef4444'}; font-size: 0.875rem; font-weight: 500; margin-top: 0.25rem;'>{delta}</div>" if delta else ""
    
    return f"""
    <div class="stat-card">
        <div style="color: #6b7280; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            {label}
        </div>
        <div style="color: #1f2937; font-size: 1.875rem; font-weight: 700; line-height: 1;">
            {value}
        </div>
        {delta_html}
    </div>
    """

def main():
    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Trading Performance Dashboard</h1>
        <p>Advanced Analytics & Portfolio Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control panel
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading trading data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("⚠️ No data available. Please check your data file.")
        return
    
    # Advanced debug information
    with st.expander("🔍 Data Quality Report", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Data Overview**")
            st.write(f"• Total Records: {len(df):,}")
            st.write(f"• Date Range: {(df['Loop_Date'].max() - df['Loop_Date'].min()).days} days")
            st.write(f"• Assets Traded: {df['Ativo'].nunique()}")
        
        with col2:
            st.markdown("**📅 Temporal Distribution**")
            years_dist = df['Loop_Date'].dt.year.value_counts().sort_index()
            for year, count in years_dist.items():
                st.write(f"• {year}: {count:,} trades")
        
        with col3:
            st.markdown("**🔧 Data Quality**")
            st.write(f"• Missing PNL: {df['PNL'].isna().sum()}")
            st.write(f"• Missing Gatilho_Dia: {df['Gatilho_Dia'].isna().sum()}")
            st.write(f"• Missing Ativo: {df['Ativo'].isna().sum()}")
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Date filter
        st.markdown("#### 📅 Time Period")
        available_months = df['Loop_Date'].dt.to_period('M').unique()
        available_months_str = [str(month) for month in sorted(available_months)]
        
        selected_months = st.multiselect(
            "Select Months",
            options=available_months_str,
            default=available_months_str,
            help="Filter by specific time periods"
        )
        
        # Asset filter
        st.markdown("#### 🏢 Assets")
        available_ativos = sorted(df['Ativo'].dropna().unique())
        selected_ativos = st.multiselect(
            "Select Assets",
            options=available_ativos,
            default=available_ativos,
            help="Filter by specific trading assets"
        )
        
        # Filter summary
        st.markdown("---")
        st.markdown("#### 📋 Filter Summary")
        st.info(f"**Periods:** {len(selected_months)}/{len(available_months_str)}\n\n**Assets:** {len(selected_ativos)}/{len(available_ativos)}")
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_months:
        mask_months = df_filtered['Loop_Date'].dt.to_period('M').astype(str).isin(selected_months)
        df_filtered = df_filtered[mask_months]
    
    if selected_ativos:
        mask_ativos = df_filtered['Ativo'].isin(selected_ativos)
        df_filtered = df_filtered[mask_ativos]
    
    if df_filtered.empty:
        st.warning("⚠️ No data matches your current filters. Please adjust your selection.")
        return
    
    # Calculate metrics
    try:
        df_filtered['Gatilho_Dia'] = pd.to_numeric(df_filtered['Gatilho_Dia'], errors='coerce')
        df_filtered['PNL'] = pd.to_numeric(df_filtered['PNL'], errors='coerce')
        
        df_filtered['Lot_Size'] = df_filtered['Gatilho_Dia'].apply(calculate_lot_size)
        df_filtered['Daily_PNL'] = df_filtered['Lot_Size'] * df_filtered['PNL'].fillna(0)
        df_filtered['Cumulative_PNL'] = df_filtered['Daily_PNL'].cumsum()
        df_filtered['Cumulative_Balance'] = 50000 + df_filtered['Cumulative_PNL']
        
    except Exception as e:
        st.error(f"❌ Calculation error: {str(e)}")
        return
    
    # Calculate additional metrics
    drawdown_series, max_drawdown, max_drawdown_pct, peak_series = calculate_drawdown(df_filtered['Cumulative_Balance'])
    
    # Key Performance Indicators
    st.markdown('<div class="section-header"><h2>📊 Portfolio Overview</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_trades = len(df_filtered)
    final_balance = df_filtered['Cumulative_Balance'].iloc[-1]
    performance_pct = ((final_balance - 50000) / 50000) * 100
    
    with col1:
        st.metric(
            label="📈 Total Trades",
            value=f"{total_trades:,}",
            help="Total number of executed trades"
        )
    
    with col2:
        st.metric(
            label="💰 Current Balance",
            value=f"${final_balance:,.2f}",
            delta=f"${final_balance - 50000:,.2f}",
            help="Current portfolio value"
        )
    
    with col3:
        st.metric(
            label="📊 Total Return",
            value=f"{performance_pct:+.2f}%",
            delta=f"{performance_pct:.2f}%",
            help="Overall portfolio performance"
        )
    
    with col4:
        st.metric(
            label="📉 Max Drawdown",
            value=f"${max_drawdown:,.2f}",
            delta=f"{max_drawdown_pct:.2f}%",
            delta_color="inverse",
            help="Maximum peak-to-trough decline"
        )
    
    # Performance visualization
    st.markdown('<div class="section-header"><h2>📈 Performance Analysis</h2></div>', unsafe_allow_html=True)
    
    # Performance chart
    perf_chart = create_performance_chart(df_filtered)
    st.plotly_chart(perf_chart, use_container_width=True)
    
    # Drawdown chart
    drawdown_chart = create_drawdown_chart(df_filtered, drawdown_series)
    st.plotly_chart(drawdown_chart, use_container_width=True)
    
    # Detailed statistics
    st.markdown('<div class="section-header"><h2>📋 Detailed Analytics</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stat-card"><h3>💹 Trade Statistics</h3>', unsafe_allow_html=True)
        winning_trades = len(df_filtered[df_filtered['PNL'] > 0])
        losing_trades = len(df_filtered[df_filtered['PNL'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = df_filtered[df_filtered['PNL'] > 0]['Daily_PNL'].mean() if winning_trades > 0 else 0
        avg_loss = df_filtered[df_filtered['PNL'] < 0]['Daily_PNL'].mean() if losing_trades > 0 else 0
        
        st.markdown(f'<div class="stat-value"><strong>Winning Trades:</strong> {winning_trades:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Losing Trades:</strong> {losing_trades:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Win Rate:</strong> {win_rate:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Avg Win:</strong> ${avg_win:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Avg Loss:</strong> ${avg_loss:,.2f}</div>', unsafe_allow_html=True)
        
        if avg_loss != 0:
            profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades)
            st.markdown(f'<div class="stat-value"><strong>Profit Factor:</strong> {profit_factor:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card"><h3>📊 Risk Metrics</h3>', unsafe_allow_html=True)
        
        returns = df_filtered['Daily_PNL'] / 50000
        avg_return = returns.mean()
        std_return = returns.std()
        
        if std_return != 0:
            sharpe_ratio = avg_return / std_return * np.sqrt(252)
            st.markdown(f'<div class="stat-value"><strong>Sharpe Ratio:</strong> {sharpe_ratio:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="stat-value"><strong>Max Drawdown:</strong> ${max_drawdown:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Drawdown %:</strong> {max_drawdown_pct:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Total Return:</strong> ${df_filtered["Cumulative_PNL"].iloc[-1]:,.2f}</div>', unsafe_allow_html=True)
        
        best_trade = df_filtered['Daily_PNL'].max()
        worst_trade = df_filtered['Daily_PNL'].min()
        st.markdown(f'<div class="stat-value"><strong>Best Trade:</strong> ${best_trade:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value"><strong>Worst Trade:</strong> ${worst_trade:,.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card"><h3>🔄 Buy vs Sell Analysis</h3>', unsafe_allow_html=True)
        
        if 'Rastreador' in df_filtered.columns:
            df_filtered['Trade_Direction'] = df_filtered['Rastreador'].apply(
                lambda x: 'Buy' if 'Buy' in str(x) else ('Sell' if 'Sell' in str(x) else 'Unknown')
            )
            
            # Buy analysis
            buy_trades = df_filtered[df_filtered['Trade_Direction'] == 'Buy']
            if len(buy_trades) > 0:
                buy_winning = len(buy_trades[buy_trades['PNL'] > 0])
                buy_win_rate = (buy_winning / len(buy_trades)) * 100
                buy_avg_win = buy_trades[buy_trades['PNL'] > 0]['Daily_PNL'].mean() if buy_winning > 0 else 0
                buy_avg_loss = buy_trades[buy_trades['PNL'] < 0]['Daily_PNL'].mean() if len(buy_trades[buy_trades['PNL'] < 0]) > 0 else 0
                buy_profit_factor = abs(buy_avg_win * buy_winning) / abs(buy_avg_loss * len(buy_trades[buy_trades['PNL'] < 0])) if len(buy_trades[buy_trades['PNL'] < 0]) > 0 else float('inf')
                
                st.markdown('<div style="background: #ecfdf5; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0;">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: 600; color: #065f46; margin-bottom: 0.5rem;">📈 BUY Trades</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.875rem;">Trades: {len(buy_trades):,} | Win Rate: {buy_win_rate:.1f}% | PF: {buy_profit_factor:.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Sell analysis
            sell_trades = df_filtered[df_filtered['Trade_Direction'] == 'Sell']
            if len(sell_trades) > 0:
                sell_winning = len(sell_trades[sell_trades['PNL'] > 0])
                sell_win_rate = (sell_winning / len(sell_trades)) * 100
                sell_avg_win = sell_trades[sell_trades['PNL'] > 0]['Daily_PNL'].mean() if sell_winning > 0 else 0
                sell_avg_loss = sell_trades[sell_trades['PNL'] < 0]['Daily_PNL'].mean() if len(sell_trades[sell_trades['PNL'] < 0]) > 0 else 0
                sell_profit_factor = abs(sell_avg_win * sell_winning) / abs(sell_avg_loss * len(sell_trades[sell_trades['PNL'] < 0])) if len(sell_trades[sell_trades['PNL'] < 0]) > 0 else float('inf')
                
                st.markdown('<div style="background: #fef2f2; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0;">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: 600; color: #991b1b; margin-bottom: 0.5rem;">📉 SELL Trades</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.875rem;">Trades: {len(sell_trades):,} | Win Rate: {sell_win_rate:.1f}% | PF: {sell_profit_factor:.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: #6b7280; font-style: italic;">Rastreador column not available</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy Performance Analysis
    st.markdown('<div class="section-header"><h2>🎯 Strategy Performance Analysis</h2></div>', unsafe_allow_html=True)
    
    if 'Rastreador' in df_filtered.columns:
        def extract_strategy(rastreador_text):
            text = str(rastreador_text)
            if 'Corpo' in text:
                return 'Corpo'
            elif 'Pavio' in text:
                return 'Pavio'
            elif 'GAP' in text:
                return 'GAP'
            elif 'Movimento' in text:
                return 'Movimento'
            else:
                return 'Unknown'
        
        df_filtered['Strategy'] = df_filtered['Rastreador'].apply(extract_strategy)
        
        # Calculate strategy performance
        strategy_stats = []
        strategies = ['Corpo', 'Pavio', 'GAP', 'Movimento']
        
        for strategy in strategies:
            strategy_trades = df_filtered[df_filtered['Strategy'] == strategy]
            
            if len(strategy_trades) > 0:
                total_trades = len(strategy_trades)
                winning_trades = len(strategy_trades[strategy_trades['PNL'] > 0])
                losing_trades = len(strategy_trades[strategy_trades['PNL'] < 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                total_pnl = strategy_trades['Daily_PNL'].sum()
                avg_win = strategy_trades[strategy_trades['PNL'] > 0]['Daily_PNL'].mean() if winning_trades > 0 else 0
                avg_loss = strategy_trades[strategy_trades['PNL'] < 0]['Daily_PNL'].mean() if losing_trades > 0 else 0
                profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades) if losing_trades > 0 else float('inf')
                
                strategy_stats.append({
                    'Strategy': strategy,
                    'Total_Trades': total_trades,
                    'Win_Rate': win_rate,
                    'Total_PNL': total_pnl,
                    'Profit_Factor': profit_factor,
                    'Avg_Win': avg_win,
                    'Avg_Loss': avg_loss
                })
        
        if strategy_stats:
            strategy_df = pd.DataFrame(strategy_stats)
            strategy_df = strategy_df.sort_values('Total_PNL', ascending=False)
            
            # Strategy performance table
            st.dataframe(
                strategy_df.style.format({
                    'Win_Rate': '{:.1f}%',
                    'Total_PNL': '${:,.2f}',
                    'Profit_Factor': '{:.2f}',
                    'Avg_Win': '${:,.2f}',
                    'Avg_Loss': '${:,.2f}'
                }).background_gradient(subset=['Total_PNL'], cmap='RdYlGn'),
                use_container_width=True,
                height=250
            )
            
            # Strategy summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_strategy = strategy_df.iloc[0]
                st.metric(
                    label="🏆 Best Strategy",
                    value=best_strategy['Strategy'],
                    delta=f"${best_strategy['Total_PNL']:,.2f}",
                    help="Strategy with highest total profit"
                )
            
            with col2:
                highest_win_rate = strategy_df.loc[strategy_df['Win_Rate'].idxmax()]
                st.metric(
                    label="🎯 Highest Win Rate",
                    value=f"{highest_win_rate['Win_Rate']:.1f}%",
                    delta=highest_win_rate['Strategy'],
                    help="Strategy with best success rate"
                )
            
            with col3:
                best_profit_factor = strategy_df.loc[strategy_df['Profit_Factor'].idxmax()]
                st.metric(
                    label="⚖️ Best Risk/Reward",
                    value=f"{best_profit_factor['Profit_Factor']:.2f}",
                    delta=best_profit_factor['Strategy'],
                    help="Strategy with best profit factor"
                )
            
            # Export strategy data
            col1, col2 = st.columns([3, 1])
            with col2:
                strategy_excel_data = export_to_excel(strategy_df)
                st.download_button(
                    label="📥 Export Strategy Data",
                    data=strategy_excel_data,
                    file_name=f"strategy_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Asset Performance Analysis
    st.markdown('<div class="section-header"><h2>📊 Asset Performance Analysis</h2></div>', unsafe_allow_html=True)
    
    if len(df_filtered) > 0:
        # Calculate comprehensive asset performance
        ativo_stats = df_filtered.groupby('Ativo').agg({
            'Daily_PNL': ['sum', 'count', 'mean', 'std'],
            'PNL': [
                lambda x: (x > 0).sum(),
                lambda x: (x < 0).sum(),
                lambda x: (x > 0).sum() / len(x) * 100,
                'max',
                'min'
            ]
        }).round(2)
        
        # Flatten column names
        ativo_stats.columns = [
            'Total_PNL', 'Total_Trades', 'Avg_PNL', 'PNL_Std',
            'Winning_Trades', 'Losing_Trades', 'Win_Rate', 'Best_Trade', 'Worst_Trade'
        ]
        
        # Calculate profit factor
        ativo_stats['Profit_Factor'] = np.where(
            ativo_stats['Losing_Trades'] > 0,
            (ativo_stats['Winning_Trades'] * ativo_stats['Avg_PNL']) / abs(ativo_stats['Losing_Trades'] * ativo_stats['Avg_PNL']),
            np.inf
        )
        
        ativo_stats = ativo_stats.sort_values('Total_PNL', ascending=False)
        
        # Asset performance table
        st.dataframe(
            ativo_stats.style.format({
                'Total_PNL': '${:,.2f}',
                'Avg_PNL': '${:,.2f}',
                'PNL_Std': '${:,.2f}',
                'Win_Rate': '{:.1f}%',
                'Best_Trade': '${:,.2f}',
                'Worst_Trade': '${:,.2f}',
                'Profit_Factor': '{:.2f}'
            }).background_gradient(subset=['Total_PNL'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Export asset data
        col1, col2 = st.columns([3, 1])
        with col2:
            asset_excel_data = export_to_excel(ativo_stats.reset_index())
            st.download_button(
                label="📥 Export Asset Data",
                data=asset_excel_data,
                file_name=f"asset_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Trade List
    st.markdown('<div class="section-header"><h2>📝 Detailed Trade Log</h2></div>', unsafe_allow_html=True)
    
    # Column selector
    display_columns = st.multiselect(
        "📋 Select columns to display:",
        options=df_filtered.columns.tolist(),
        default=['Loop_Date', 'Ativo', 'Gatilho_Dia', 'Lot_Size', 'PNL', 'Daily_PNL', 'Cumulative_Balance'],
        help="Choose which data columns to show in the trade log"
    )
    
    if display_columns:
        # Format display dataframe
        display_df = df_filtered[display_columns].copy()
        
        # Format numeric columns
        numeric_columns = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['Cumulative_Balance', 'Daily_PNL', 'Gatilho_Dia', 'PNL']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            elif col == 'Lot_Size':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
        
        # Display trade log
        st.dataframe(
            display_df,
            use_container_width=True,
            height=450
        )
        
        # Export trade log
        col1, col2 = st.columns([3, 1])
        with col2:
            excel_data = export_to_excel(df_filtered)
            st.download_button(
                label="📥 Export Trade Log",
                data=excel_data,
                file_name=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 10px; margin-top: 2rem;'>
            <h4 style='color: #64748b; margin: 0;'>📊 Trading Performance Dashboard</h4>
            <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
                Advanced analytics platform for portfolio performance analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

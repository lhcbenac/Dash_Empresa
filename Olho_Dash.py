import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import openpyxl
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .positive-pnl {
        color: #28a745;
        font-weight: bold;
    }
    .negative-pnl {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def load_data_from_file():
    """Load data from the Backtesting_history.xlsx file in the repository"""
    try:
        # Load the Excel file directly from the repository
        workbook = openpyxl.load_workbook('Backtesting_history.xlsx')
        
        # Get Trades sheet
        trades_sheet = workbook["Trades"]
        trades_data = []
        
        # Skip header and process each row
        for row in trades_sheet.iter_rows(min_row=2, values_only=True):
            if row[0] is not None:  # Skip empty rows
                csv_string = str(row[0])
                values = csv_string.split(',')
                
                if len(values) >= 15:  # Ensure we have enough columns
                    trades_data.append({
                        'date': values[0],
                        'asset': values[1],
                        'strategy': values[2],
                        'operation': values[3],
                        'direction': values[4],
                        'trigger_price': float(values[5]) if values[5] else 0,
                        'exit_price': float(values[6]) if values[6] else 0,
                        'position_size': int(float(values[7])) if values[7] else 0,
                        'pnl_percent': float(values[8]) if values[8] else 0,
                        'pnl': float(values[9]) if values[9] else 0,
                        'tracker': values[10],
                        'high': float(values[11]) if values[11] else 0,
                        'low': float(values[12]) if values[12] else 0,
                        'open': float(values[13]) if values[13] else 0,
                        'close': float(values[14]) if values[14] else 0
                    })
        
        df = pd.DataFrame(trades_data)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Backtesting_history.xlsx file not found in the repository!")
        st.info("Please make sure the file 'Backtesting_history.xlsx' is uploaded to your GitHub repository.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def calculate_drawdown(df, portfolio_size=50000):
    """Calculate drawdown based on cumulative PnL with 50k portfolio base"""
    df_sorted = df.sort_values('date').copy()
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
    
    # Calculate portfolio value (starting at 50k + cumulative PnL)
    df_sorted['portfolio_value'] = portfolio_size + df_sorted['cumulative_pnl']
    
    # Calculate running maximum (peak) of portfolio value
    df_sorted['peak_portfolio'] = df_sorted['portfolio_value'].expanding().max()
    
    # Calculate drawdown as the difference from peak
    df_sorted['drawdown'] = df_sorted['portfolio_value'] - df_sorted['peak_portfolio']
    
    # Calculate drawdown percentage based on peak portfolio value
    df_sorted['drawdown_percent'] = (df_sorted['drawdown'] / df_sorted['peak_portfolio']) * 100
    
    # Handle division by zero
    df_sorted['drawdown_percent'] = df_sorted['drawdown_percent'].fillna(0)
    
    return df_sorted

def create_evolution_chart(df):
    """Create PnL evolution chart"""
    df_chart = df.sort_values('date').copy()
    df_chart['cumulative_pnl'] = df_chart['pnl'].cumsum()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative PnL Evolution', 'Drawdown'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Cumulative PnL line
    fig.add_trace(
        go.Scatter(
            x=df_chart['date'],
            y=df_chart['cumulative_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Calculate and add drawdown
    df_dd = calculate_drawdown(df_chart)
    fig.add_trace(
        go.Scatter(
            x=df_dd['date'],
            y=df_dd['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=2),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Trading Performance Analysis"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="PnL (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (R$)", row=2, col=1)
    
    return fig

def create_monthly_performance_chart(df):
    """Create monthly performance bar chart"""
    monthly_stats = df.groupby('month').agg({
        'pnl': 'sum',
        'date': 'count'
    }).rename(columns={'date': 'operations'})
    
    monthly_stats.index = monthly_stats.index.astype(str)
    
    fig = go.Figure()
    
    colors = ['#28a745' if pnl > 0 else '#dc3545' for pnl in monthly_stats['pnl']]
    
    fig.add_trace(go.Bar(
        x=monthly_stats.index,
        y=monthly_stats['pnl'],
        marker_color=colors,
        name='Monthly PnL',
        text=[f'R${pnl:,.0f}<br>{ops} ops' for pnl, ops in 
              zip(monthly_stats['pnl'], monthly_stats['operations'])],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Monthly Performance',
        xaxis_title='Month',
        yaxis_title='PnL (R$)',
        height=400
    )
    
    return fig

# Main dashboard
def main():
    st.title("📈 Trading Strategy Dashboard")
    st.markdown("---")
    
    # Load data from the repository file
    with st.spinner("Loading trading data..."):
        df = load_data_from_file()
    
    if df is not None and not df.empty:
        # Success message
        st.success(f"✅ Data loaded successfully! Found {len(df)} trading operations.")
        
        # Filters section in main area
        st.header("🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Month filter
            available_months = sorted(df['month'].astype(str).unique())
            selected_months = st.multiselect(
                "Select Month(s)",
                options=available_months,
                default=available_months,
                help="Filter data by specific months"
            )
        
        with col2:
            # Strategy filter
            available_strategies = sorted(df['strategy'].unique())
            selected_strategies = st.multiselect(
                "Select Strategy(ies)",
                options=available_strategies,
                default=available_strategies,
                help="Filter data by trading strategies"
            )
        
        with col3:
            # Asset filter
            available_assets = sorted(df['asset'].unique())
            selected_assets = st.multiselect(
                "Select Asset(s)",
                options=available_assets,
                default=available_assets,
                help="Filter data by specific assets"
            )
        
        # Apply filters
        filtered_df = df[
            (df['month'].astype(str).isin(selected_months)) &
            (df['strategy'].isin(selected_strategies)) &
            (df['asset'].isin(selected_assets))
        ].copy()
        
        st.markdown("---")
        
        if not filtered_df.empty:
            # Key Metrics Row
            st.header("📊 Key Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_operations = len(filtered_df)
            total_pnl = filtered_df['pnl'].sum()
            avg_pnl_per_trade = filtered_df['pnl'].mean()
            win_rate = (filtered_df['pnl'] > 0).mean() * 100
            
            # Calculate profit factor
            winning_trades = filtered_df[filtered_df['pnl'] > 0]['pnl'].sum()
            losing_trades = abs(filtered_df[filtered_df['pnl'] < 0]['pnl'].sum())
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')
            
            # Calculate portfolio percentage return
            portfolio_return_pct = (total_pnl / 50000) * 100
            
            with col1:
                st.metric(
                    label="Total Operations",
                    value=f"{total_operations:,}",
                    help="Total number of trading operations"
                )
            
            with col2:
                st.metric(
                    label="Total PnL",
                    value=f"R${total_pnl:,.2f}",
                    delta=f"{portfolio_return_pct:+.2f}%" if total_pnl != 0 else "0%",
                    help="Total profit and loss with portfolio percentage"
                )
            
            with col3:
                st.metric(
                    label="Average PnL/Trade",
                    value=f"R${avg_pnl_per_trade:.2f}",
                    help="Average profit per trade"
                )
            
            with col4:
                st.metric(
                    label="Win Rate",
                    value=f"{win_rate:.1f}%",
                    help="Percentage of profitable trades"
                )
            
            with col5:
                st.metric(
                    label="Profit Factor",
                    value=f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                    help="Ratio of gross profits to gross losses"
                )
            
            st.markdown("---")
            
            # Charts Row
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 PnL Evolution & Drawdown")
                evolution_fig = create_evolution_chart(filtered_df)
                st.plotly_chart(evolution_fig, use_container_width=True)
            
            with col2:
                st.subheader("📅 Monthly Performance")
                monthly_fig = create_monthly_performance_chart(filtered_df)
                st.plotly_chart(monthly_fig, use_container_width=True)
            
            # Additional Analytics
            st.subheader("📋 Detailed Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Operations by Day**")
                daily_ops = filtered_df.groupby(filtered_df['date'].dt.date).size()
                avg_ops_per_day = daily_ops.mean()
                st.metric("Average Operations/Day", f"{avg_ops_per_day:.1f}")
                
                # Show daily distribution
                daily_dist = daily_ops.value_counts().sort_index()
                st.bar_chart(daily_dist)
            
            with col2:
                st.write("**Drawdown Analysis**")
                dd_df = calculate_drawdown(filtered_df, portfolio_size=50000)
                max_drawdown = dd_df['drawdown'].min()
                max_drawdown_pct = dd_df['drawdown_percent'].min()
                max_dd_date = dd_df.loc[dd_df['drawdown'].idxmin(), 'date'].strftime('%Y-%m-%d')
                
                st.metric("Max Drawdown", f"R${max_drawdown:.2f}")
                st.metric("Max Drawdown %", f"{max_drawdown_pct:.2f}%")
                st.write(f"**Worst Date:** {max_dd_date}")
                st.caption("*Based on R$50k portfolio")
            
            with col3:
                st.write("**Strategy Performance**")
                strategy_performance = filtered_df.groupby('strategy').agg({
                    'pnl': ['sum', 'count', 'mean']
                }).round(2)
                strategy_performance.columns = ['Total PnL', 'Operations', 'Avg PnL']
                st.dataframe(strategy_performance, use_container_width=True)
            
            # Detailed Data Table
            with st.expander("📝 View Detailed Trading Data"):
                display_df = filtered_df.copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    display_df[['date', 'asset', 'strategy', 'operation', 'direction', 
                              'trigger_price', 'exit_price', 'position_size', 'pnl_percent', 'pnl']],
                    use_container_width=True,
                    height=400
                )
            
            # Export functionality
            st.subheader("💾 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Filtered Data as CSV",
                    data=csv_data,
                    file_name=f"trading_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Summary report
                dd_df = calculate_drawdown(filtered_df, portfolio_size=50000)
                max_drawdown = dd_df['drawdown'].min()
                max_drawdown_pct = dd_df['drawdown_percent'].min()
                
                summary_data = {
                    'Metric': ['Total Operations', 'Total PnL', 'Portfolio Return %', 'Win Rate', 'Profit Factor',
                             'Average PnL/Trade', 'Max Drawdown', 'Max Drawdown %', 'Avg Operations/Day'],
                    'Value': [total_operations, f"R${total_pnl:.2f}", f"{portfolio_return_pct:.2f}%", 
                            f"{win_rate:.1f}%", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                            f"R${avg_pnl_per_trade:.2f}", f"R${max_drawdown:.2f}", 
                            f"{max_drawdown_pct:.2f}%", f"{avg_ops_per_day:.1f}"]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Summary Report",
                    data=summary_csv,
                    file_name=f"trading_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.warning("No data matches the selected filters. Please adjust your selections.")
    
    else:
        # Show error message when file cannot be loaded
        st.header("📊 Trading Strategy Dashboard")
        st.error("❌ Unable to load the trading data file.")
        st.info("""
        **To fix this issue:**
        1. Make sure `Backtesting_history.xlsx` is uploaded to your GitHub repository
        2. Ensure the file contains a 'Trades' sheet with the proper data format
        3. Redeploy your Streamlit app after uploading the file
        """)
        
        with st.expander("📋 Expected Data Format"):
            st.write("""
            Your Excel file should contain a 'Trades' sheet with the following columns:
            - date, asset, strategy, operation, direction
            - trigger_price, exit_price, position_size
            - pnl_percent, pnl, tracker
            - high, low, open, close
            """)

if __name__ == "__main__":
    main()

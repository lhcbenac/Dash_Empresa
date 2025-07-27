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
    page_title="Backtesting Dashboard",
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
    .stMetric > label {
        font-size: 14px !important;
        font-weight: bold !important;
    }
    .stMetric > div {
        font-size: 24px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the backtesting data"""
    try:
        # Load the CSV file with more explicit parameters
        df_raw = pd.read_csv('Backtesting2_history.csv', 
                           encoding='utf-8',  # Try different encodings if needed
                           low_memory=False)  # Don't infer dtypes chunk by chunk
        
        # Debug: Show total rows before filtering
        st.info(f"📊 Total rows in CSV file: {len(df_raw)}")
        
        # Show date range in raw data
        if 'Loop_Date' in df_raw.columns:
            try:
                raw_dates = pd.to_datetime(df_raw['Loop_Date'], errors='coerce')
                valid_dates = raw_dates.dropna()
                if len(valid_dates) > 0:
                    st.info(f"📅 Raw date range: {valid_dates.min()} to {valid_dates.max()}")
                    
                    # Show year distribution
                    years = valid_dates.dt.year.value_counts().sort_index()
                    st.info(f"📅 Years in data: {dict(years)}")
                else:
                    st.warning("⚠️ No valid dates found in Loop_Date column")
            except Exception as date_error:
                st.warning(f"⚠️ Error parsing dates: {date_error}")
        
        # Check if Operou_Dia column exists and its values
        if 'Operou_Dia' not in df_raw.columns:
            st.error("❌ Column 'Operou_Dia' not found in the CSV file.")
            st.write("Available columns:", df_raw.columns.tolist())
            return None
        
        # Show Operou_Dia value distribution
        operou_dia_counts = df_raw['Operou_Dia'].value_counts()
        st.info(f"📈 Operou_Dia distribution: {dict(operou_dia_counts)}")
        
        # Filter only rows where Operou_Dia = True
        # Handle different possible representations of True
        mask = (
            (df_raw['Operou_Dia'] == True) | 
            (df_raw['Operou_Dia'] == 'True') | 
            (df_raw['Operou_Dia'] == 1) |
            (df_raw['Operou_Dia'] == '1') |
            (df_raw['Operou_Dia'].astype(str).str.upper() == 'TRUE')
        )
        
        df = df_raw[mask].copy()
        
        st.info(f"📊 Rows after filtering Operou_Dia = True: {len(df)}")
        
        if len(df) == 0:
            st.warning("⚠️ No rows found where Operou_Dia = True. Check your data.")
            st.write("Sample Operou_Dia values:", df_raw['Operou_Dia'].head(20).tolist())
            return None
        
        # Convert Loop_Date to datetime and normalize (remove time component)
        try:
            df['Loop_Date'] = pd.to_datetime(df['Loop_Date'], errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = df['Loop_Date'].isna().sum()
            if invalid_dates > 0:
                st.warning(f"⚠️ Found {invalid_dates} rows with invalid dates - removing them")
                df = df.dropna(subset=['Loop_Date'])
            
            # Normalize dates (remove time component)
            df['Loop_Date'] = df['Loop_Date'].dt.date
            df['Loop_Date'] = pd.to_datetime(df['Loop_Date'])  # Convert back to datetime for consistency
            
        except Exception as date_error:
            st.error(f"❌ Error processing dates: {date_error}")
            return None
        
        # Sort by date
        df = df.sort_values('Loop_Date').reset_index(drop=True)
        
        # Final check on date range
        if len(df) > 0:
            st.success(f"✅ Successfully loaded {len(df)} records from {df['Loop_Date'].min().date()} to {df['Loop_Date'].max().date()}")
        
        return df
        
    except FileNotFoundError:
        st.error("❌ File 'Backtesting2_history.csv' not found. Please upload the file to the same directory.")
        return None
    except UnicodeDecodeError:
        st.error("❌ Error reading CSV file. Try saving it with UTF-8 encoding.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        return None

def calculate_lot_size(gatilho_dia_value, initial_balance=50000):
    """Calculate lot size based on gatilho_dia value"""
    # Handle NaN, None, zero, or negative values
    if pd.isna(gatilho_dia_value) or gatilho_dia_value is None or gatilho_dia_value <= 0:
        return 100
    
    try:
        # Convert to float to ensure numeric operation
        gatilho_dia_value = float(gatilho_dia_value)
        lot_size = initial_balance / gatilho_dia_value
        
        # Round down to nearest 100, minimum 100
        lot_size = max(100, math.floor(lot_size / 100) * 100)
        return int(lot_size)
    except (ValueError, TypeError, ZeroDivisionError):
        # Return default value if any error occurs
        return 100

def calculate_drawdown(cumulative_pnl):
    """Calculate drawdown statistics"""
    # Calculate running maximum (peak)
    peak = cumulative_pnl.cummax()
    
    # Calculate drawdown
    drawdown = cumulative_pnl - peak
    
    # Find maximum drawdown
    max_drawdown = drawdown.min()
    max_drawdown_idx = drawdown.idxmin()
    
    # Find the peak before max drawdown
    peak_value = peak.iloc[max_drawdown_idx]
    
    # Calculate drawdown percentage
    if peak_value != 0:
        max_drawdown_pct = (max_drawdown / peak_value) * 100
    else:
        max_drawdown_pct = 0
    
    return drawdown, max_drawdown, max_drawdown_pct, peak

def create_performance_chart(df_filtered):
    """Create performance chart with cumulative PNL"""
    fig = go.Figure()
    
    # Add cumulative PNL line
    fig.add_trace(go.Scatter(
        x=df_filtered['Loop_Date'],
        y=df_filtered['Cumulative_Balance'],
        mode='lines',
        name='Cumulative Balance',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Balance: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add initial balance line
    fig.add_hline(
        y=50000, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Initial Balance ($50,000)"
    )
    
    fig.update_layout(
        title="📈 Cumulative Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_drawdown_chart(df_filtered, drawdown_series):
    """Create drawdown chart"""
    fig = go.Figure()
    
    # Add drawdown area
    fig.add_trace(go.Scatter(
        x=df_filtered['Loop_Date'],
        y=drawdown_series,
        fill='tonexty',
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)',
        hovertemplate='Date: %{x}<br>Drawdown: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title="📉 Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown ($)",
        hovermode='x unified',
        showlegend=True,
        height=300
    )
    
    return fig

def export_to_excel(df):
    """Export dataframe to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Trade_List', index=False)
    
    return output.getvalue()

def main():
    st.title("📈 Backtesting Dashboard")
    st.markdown("---")
    
    # Add a button to clear cache
    if st.button("🔄 Reload Data (Clear Cache)"):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    df = load_data()
    
    if df is None or df.empty:
        st.warning("⚠️ No data available. Please check your data file.")
        return
    
    # Debug information (can be removed later)
    with st.expander("🔍 Data Debug Info (Click to expand)"):
        try:
            df_raw_debug = pd.read_csv('Backtesting2_history.csv', low_memory=False)
            st.write(f"**Total rows in file:** {len(df_raw_debug)}")
            
            # Check years in raw data
            if 'Loop_Date' in df_raw_debug.columns:
                raw_dates = pd.to_datetime(df_raw_debug['Loop_Date'], errors='coerce')
                if raw_dates.notna().any():
                    years_raw = raw_dates.dt.year.value_counts().sort_index()
                    st.write(f"**Years in raw CSV:** {dict(years_raw)}")
                    
                    # Check Operou_Dia by year
                    df_raw_debug['Year'] = raw_dates.dt.year
                    operou_by_year = df_raw_debug.groupby('Year')['Operou_Dia'].value_counts()
                    st.write(f"**Operou_Dia by year:**")
                    for (year, operou), count in operou_by_year.items():
                        st.write(f"- {year} - {operou}: {count}")
        except:
            st.write("Could not read raw file for debugging")
        
        st.write(f"**Rows with Operou_Dia = True:** {len(df)}")
        st.write(f"**Date range:** {df['Loop_Date'].min()} to {df['Loop_Date'].max()}")
        
        # Show all available months in the filtered data
        all_months = df['Loop_Date'].dt.to_period('M').value_counts().sort_index()
        st.write(f"**Available months with trade counts:**")
        for month, count in all_months.items():
            st.write(f"- {month}: {count} trades")
        
        # Show years in filtered data
        years_filtered = df['Loop_Date'].dt.year.value_counts().sort_index()
        st.write(f"**Years in filtered data:** {dict(years_filtered)}")
        
        # Check for any issues with Operou_Dia column
        if 'Operou_Dia' in df.columns:
            operou_dia_values = df['Operou_Dia'].value_counts()
            st.write(f"**Operou_Dia values in filtered data:**")
            for value, count in operou_dia_values.items():
                st.write(f"- {value}: {count}")
        
        st.write(f"**Gatilho_Dia column info:**")
        st.write(f"- Data type: {df['Gatilho_Dia'].dtype}")
        st.write(f"- Non-null values: {df['Gatilho_Dia'].notna().sum()}")
        st.write(f"- Unique values: {df['Gatilho_Dia'].nunique()}")
        st.write(f"- Sample values: {df['Gatilho_Dia'].head().tolist()}")
        
        # Try to convert to numeric and check for issues
        gatilho_numeric = pd.to_numeric(df['Gatilho_Dia'], errors='coerce')
        st.write(f"- Values that can't be converted to numbers: {gatilho_numeric.isna().sum()}")
        if gatilho_numeric.notna().any():
            st.write(f"- Zero values (numeric): {(gatilho_numeric == 0).sum()}")
            st.write(f"- Negative values (numeric): {(gatilho_numeric < 0).sum()}")
        
        st.write(f"**PNL column info:**")
        st.write(f"- Data type: {df['PNL'].dtype}")
        st.write(f"- Non-null values: {df['PNL'].notna().sum()}")
        
        st.write(f"**Sample data:**")
        st.dataframe(df[['Loop_Date', 'Ativo', 'Gatilho_Dia', 'PNL', 'Operou_Dia']].head(10))
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Month filter
    available_months = df['Loop_Date'].dt.to_period('M').unique()
    available_months_str = [str(month) for month in sorted(available_months)]
    
    selected_months = st.sidebar.multiselect(
        "Select Months",
        options=available_months_str,
        default=available_months_str,
        help="Filter data by specific months (YYYY-MM format)"
    )
    
    # Ativo filter
    available_ativos = sorted(df['Ativo'].dropna().unique())
    selected_ativos = st.sidebar.multiselect(
        "Select Assets (Ativo)",
        options=available_ativos,
        default=available_ativos,
        help="Filter data by specific assets"
    )
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_months:
        mask_months = df_filtered['Loop_Date'].dt.to_period('M').astype(str).isin(selected_months)
        df_filtered = df_filtered[mask_months]
    
    if selected_ativos:
        mask_ativos = df_filtered['Ativo'].isin(selected_ativos)
        df_filtered = df_filtered[mask_ativos]
    
    if df_filtered.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Calculate lot sizes and PNL
    try:
        # Clean the Gatilho_Dia and PNL columns
        df_filtered['Gatilho_Dia'] = pd.to_numeric(df_filtered['Gatilho_Dia'], errors='coerce')
        df_filtered['PNL'] = pd.to_numeric(df_filtered['PNL'], errors='coerce')
        
        # Calculate lot sizes based on 50k / Gatilho_Dia
        df_filtered['Lot_Size'] = df_filtered['Gatilho_Dia'].apply(calculate_lot_size)
        
        # Calculate actual PNL: PNL per unit × Lot Size
        df_filtered['Daily_PNL'] = df_filtered['Lot_Size'] * df_filtered['PNL'].fillna(0)
        df_filtered['Cumulative_PNL'] = df_filtered['Daily_PNL'].cumsum()
        df_filtered['Cumulative_Balance'] = 50000 + df_filtered['Cumulative_PNL']
        
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        st.info("Please check your data for invalid values in Gatilho_Dia or PNL columns.")
        return
    
    # Calculate drawdown
    drawdown_series, max_drawdown, max_drawdown_pct, peak_series = calculate_drawdown(df_filtered['Cumulative_Balance'])
    
    # Key Metrics Row
    st.header("📊 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(df_filtered)
        st.metric(
            label="🔢 Total Trades",
            value=f"{total_trades:,}",
            help="Total number of trades executed"
        )
    
    with col2:
        final_balance = df_filtered['Cumulative_Balance'].iloc[-1]
        st.metric(
            label="💰 Final Balance",
            value=f"${final_balance:,.2f}",
            delta=f"${final_balance - 50000:,.2f}",
            help="Final balance after all trades"
        )
    
    with col3:
        performance_pct = ((final_balance - 50000) / 50000) * 100
        st.metric(
            label="📈 Performance %",
            value=f"{performance_pct:.2f}%",
            delta=f"{performance_pct:.2f}%",
            help="Overall performance percentage"
        )
    
    with col4:
        st.metric(
            label="📉 Max Drawdown",
            value=f"${max_drawdown:,.2f}",
            delta=f"{max_drawdown_pct:.2f}%",
            delta_color="inverse",
            help="Maximum peak-to-trough decline"
        )
    
    st.markdown("---")
    
    # Performance Charts
    st.header("📈 Performance Analysis")
    
    # Performance chart
    perf_chart = create_performance_chart(df_filtered)
    st.plotly_chart(perf_chart, use_container_width=True)
    
    # Drawdown chart
    drawdown_chart = create_drawdown_chart(df_filtered, drawdown_series)
    st.plotly_chart(drawdown_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Additional Statistics
    st.header("📋 Detailed Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💹 Trade Statistics")
        winning_trades = len(df_filtered[df_filtered['PNL'] > 0])
        losing_trades = len(df_filtered[df_filtered['PNL'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = df_filtered[df_filtered['PNL'] > 0]['Daily_PNL'].mean() if winning_trades > 0 else 0
        avg_loss = df_filtered[df_filtered['PNL'] < 0]['Daily_PNL'].mean() if losing_trades > 0 else 0
        
        st.write(f"**Winning Trades:** {winning_trades}")
        st.write(f"**Losing Trades:** {losing_trades}")
        st.write(f"**Win Rate:** {win_rate:.2f}%")
        st.write(f"**Average Win:** ${avg_win:,.2f}")
        st.write(f"**Average Loss:** ${avg_loss:,.2f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades)
            st.write(f"**Profit Factor:** {profit_factor:.2f}")
    
    with col2:
        st.subheader("📊 Risk Metrics")
        
        # Sharpe-like ratio (simplified)
        returns = df_filtered['Daily_PNL'] / 50000
        avg_return = returns.mean()
        std_return = returns.std()
        
        if std_return != 0:
            sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Assuming daily data
            st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
        
        st.write(f"**Maximum Drawdown:** ${max_drawdown:,.2f}")
        st.write(f"**Maximum Drawdown %:** {max_drawdown_pct:.2f}%")
        st.write(f"**Total Return:** ${df_filtered['Cumulative_PNL'].iloc[-1]:,.2f}")
        
        # Best and worst trades
        best_trade = df_filtered['Daily_PNL'].max()
        worst_trade = df_filtered['Daily_PNL'].min()
        st.write(f"**Best Trade:** ${best_trade:,.2f}")
        st.write(f"**Worst Trade:** ${worst_trade:,.2f}")
    
    with col3:
        st.subheader("🏆 Asset Performance Rankings")
        
        # Calculate performance by Ativo
        if len(df_filtered) > 0:
            ativo_performance = df_filtered.groupby('Ativo').agg({
                'Daily_PNL': ['sum', 'count', 'mean'],
                'PNL': lambda x: (x > 0).sum() / len(x) * 100  # Win rate
            }).round(2)
            
            # Flatten column names
            ativo_performance.columns = ['Total_PNL', 'Trade_Count', 'Avg_PNL', 'Win_Rate']
            ativo_performance = ativo_performance.sort_values('Total_PNL', ascending=False)
            
            # Top 5 Best Performers
            st.write("**🥇 Top 5 Best Assets:**")
            top_5 = ativo_performance.head(5)
            for i, (ativo, data) in enumerate(top_5.iterrows(), 1):
                st.write(f"{i}. **{ativo}**: ${data['Total_PNL']:,.2f} ({data['Trade_Count']:.0f} trades)")
            
            st.write("---")
            
            # Top 5 Worst Performers
            st.write("**📉 Top 5 Worst Assets:**")
            bottom_5 = ativo_performance.tail(5)
            for i, (ativo, data) in enumerate(bottom_5.iterrows(), 1):
                st.write(f"{i}. **{ativo}**: ${data['Total_PNL']:,.2f} ({data['Trade_Count']:.0f} trades)")
        else:
            st.write("No data available for asset rankings")
    
    st.markdown("---")
    
    # Asset Performance Table
    st.header("📊 Asset Performance Analysis")
    
    if len(df_filtered) > 0:
        # Calculate comprehensive performance metrics by Ativo
        ativo_stats = df_filtered.groupby('Ativo').agg({
            'Daily_PNL': ['sum', 'count', 'mean', 'std'],
            'PNL': [
                lambda x: (x > 0).sum(),  # Winning trades
                lambda x: (x < 0).sum(),  # Losing trades
                lambda x: (x > 0).sum() / len(x) * 100,  # Win rate
                'max',  # Best trade
                'min'   # Worst trade
            ]
        }).round(2)
        
        # Flatten column names
        ativo_stats.columns = [
            'Total_PNL', 'Total_Trades', 'Avg_PNL', 'PNL_Std',
            'Winning_Trades', 'Losing_Trades', 'Win_Rate', 'Best_Trade', 'Worst_Trade'
        ]
        
        # Calculate additional metrics
        ativo_stats['Profit_Factor'] = np.where(
            ativo_stats['Losing_Trades'] > 0,
            (ativo_stats['Winning_Trades'] * ativo_stats['Avg_PNL']) / abs(ativo_stats['Losing_Trades'] * ativo_stats['Avg_PNL']),
            np.inf
        )
        
        # Sort by total PNL
        ativo_stats = ativo_stats.sort_values('Total_PNL', ascending=False)
        
        # Display the table
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
        
        # Add download button for asset performance
        asset_excel_data = export_to_excel(ativo_stats.reset_index())
        st.download_button(
            label="📥 Export Asset Performance to Excel",
            data=asset_excel_data,
            file_name=f"asset_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.markdown("---")
    
    # Trade List
    st.header("📝 Trade List")
    
    # Display columns selector
    display_columns = st.multiselect(
        "Select columns to display:",
        options=df_filtered.columns.tolist(),
        default=['Loop_Date', 'Ativo', 'Gatilho_Dia', 'Lot_Size', 'PNL', 'Daily_PNL', 'Cumulative_Balance'],
        help="Choose which columns to show in the trade list"
    )
    
    if display_columns:
        # Format the dataframe for display
        display_df = df_filtered[display_columns].copy()
        
        # Format numeric columns
        numeric_columns = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['Cumulative_Balance', 'Daily_PNL', 'Gatilho_Dia', 'PNL']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            elif col == 'Lot_Size':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Export button
        excel_data = export_to_excel(df_filtered)
        st.download_button(
            label="📥 Export Trade List to Excel",
            data=excel_data,
            file_name=f"backtesting_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>📈 Backtesting Dashboard | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

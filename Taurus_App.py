import streamlit as st
import pandas as pd
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

# --- UPLOAD PAGE ---
if page == "Upload":
    st.title("📤 Upload Taurus Excel File")
    uploaded_file = st.file_uploader("Upload the Excel file with 'Taurus' sheet", type=["xlsx"])
    
    if uploaded_file:
        try:
            # Read specifically the 'Taurus' sheet
            df_taurus = pd.read_excel(uploaded_file, sheet_name="Taurus", engine="openpyxl")
            
            # Check if required columns exist
            required_cols = {"Chave", "AssessorReal", "Categoria", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"}
            
            if not required_cols.issubset(df_taurus.columns):
                missing_cols = required_cols - set(df_taurus.columns)
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: Chave, AssessorReal, Categoria, Comissão, Tributo_Retido, Pix_Assessor, Lucro_Empresa")
            else:
                # Store data in session state
                st.session_state["df_taurus"] = df_taurus
                st.success("✅ Taurus data successfully loaded!")
                
                # Show basic info
                st.markdown("### 📊 Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df_taurus))
                with col2:
                    st.metric("Unique Assessors", df_taurus["AssessorReal"].nunique())
                with col3:
                    st.metric("Unique Chaves", df_taurus["Chave"].nunique())
                
                # Show sample data
                st.markdown("### 👀 Sample Data")
                st.dataframe(df_taurus.head(), use_container_width=True )
                
        except ValueError as e:
            if "Worksheet named 'Taurus' not found" in str(e):
                st.error("❌ Sheet named 'Taurus' not found in the uploaded file.")
                st.info("Please make sure your Excel file contains a sheet named 'Taurus'.")
            else:
                st.error(f"❌ Error reading file: {e}")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- MACRO VIEW PAGE ---
elif page == "Macro View":
    st.title("📊 Macro View - Summary by Assessor")
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    df_taurus = st.session_state["df_taurus"]
    
    # Chave filter
    st.markdown("### 🔍 Filter Options")
    chave_list = sorted(df_taurus["Chave"].dropna().unique())
    selected_chaves = st.multiselect(
        "Select Chave periods",
        chave_list,
        default=chave_list  # Default to all selected
    )
    
    if selected_chaves:
        df_filtered = df_taurus[df_taurus["Chave"].isin(selected_chaves)]
        
        st.markdown(f"### Summary for Chave(s): `{', '.join(map(str, selected_chaves))}`")
        
        # Create pivot table with AssessorReal as rows and sum of financial columns
        financial_cols = ["Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        
        # Group by AssessorReal and sum the financial columns
        summary_df = (
            df_filtered.groupby("AssessorReal")[financial_cols]
            .sum()
            .reset_index()
        )
        
        # Sort by Lucro_Empresa (descending - greater to lower)
        summary_df = summary_df.sort_values("Lucro_Empresa", ascending=False)
        
        # Add totals row
        totals_row = pd.DataFrame({
            "AssessorReal": ["TOTAL"],
            "Comissão": [summary_df["Comissão"].sum()],
            "Tributo_Retido": [summary_df["Tributo_Retido"].sum()],
            "Pix_Assessor": [summary_df["Pix_Assessor"].sum()],
            "Lucro_Empresa": [summary_df["Lucro_Empresa"].sum()]
        })
        
        summary_with_totals = pd.concat([summary_df, totals_row], ignore_index=True)
        
        # Display the table
        st.dataframe(summary_with_totals.round(2), use_container_width=True ,height=460)
        
        # Export CSV
        csv = summary_with_totals.round(2).to_csv(index=False).encode("utf-8")
        filename = f"Macro_Summary_{'_'.join(map(str, selected_chaves))}.csv"
        st.download_button("📥 Download Summary CSV", csv, filename, "text/csv")
        
    else:
        st.warning("Please select at least one Chave.")

# --- ASSESSOR VIEW PAGE ---
elif page == "Assessor View":
    st.title("👤 Assessor View - Breakdown by Category")
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    df_taurus = st.session_state["df_taurus"]
    
    # Chave filter
    st.markdown("### 🔍 Filter by Chave")
    chave_list = sorted(df_taurus["Chave"].dropna().unique())
    selected_chaves = st.multiselect(
        "Select Chave period(s)",
        chave_list,
        default=chave_list
    )
    
    # Assessor selection
    assessor_list = sorted(df_taurus["AssessorReal"].dropna().unique())
    selected_assessor = st.selectbox("Select AssessorReal", assessor_list)
    
    # Filter data by selected assessor and Chave
    df_filtered = df_taurus[
        (df_taurus["AssessorReal"] == selected_assessor) &
        (df_taurus["Chave"].isin(selected_chaves))
    ]
    
    if df_filtered.empty:
        st.warning("No data for the selected AssessorReal & Chave combination.")
    else:
        st.markdown(
            f"### Summary for `{selected_assessor}` "
            f"(Chave: {', '.join(map(str, selected_chaves))})"
        )
        
        # Financial columns to sum
        financial_cols = ["Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        
        # Group by Categoria and sum financial columns
        category_summary = (
            df_filtered
            .groupby("Categoria")[financial_cols]
            .sum()
            .reset_index()
        )
        
        # Add totals row
        totals_row = pd.DataFrame({
            "Categoria": ["TOTAL"],
            "Comissão": [category_summary["Comissão"].sum()],
            "Tributo_Retido": [category_summary["Tributo_Retido"].sum()],
            "Pix_Assessor": [category_summary["Pix_Assessor"].sum()],
            "Lucro_Empresa": [category_summary["Lucro_Empresa"].sum()]
        })
        
        category_with_totals = pd.concat([category_summary, totals_row], ignore_index=True)
        
        # Display the table
        st.dataframe(category_with_totals.round(2), use_container_width=True )
        
        # --- DOWNLOAD SECTION ---
        st.markdown("### 📥 Download Options")
        
        # Create two columns for download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Export category summary CSV
            csv_summary = category_with_totals.round(2).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Download Category Summary CSV",
                csv_summary,
                f"{selected_assessor}_Category_Summary_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )
        
        with col2:
            # Export detailed transactions Excel
            # Select only the required columns for the detailed export
            detailed_cols = [
                "Data Receita", "Conta", "Cliente", "Comissão", 
                "Receita Assessor", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"
            ]
            
            # Check which columns exist in the dataframe
            available_cols = [col for col in detailed_cols if col in df_filtered.columns]
            
            if available_cols:
                # Create detailed export with available columns
                detailed_export = df_filtered[available_cols].copy()
                
                # Round numeric columns to 2 decimal places
                numeric_cols = detailed_export.select_dtypes(include=['float64', 'int64']).columns
                detailed_export[numeric_cols] = detailed_export[numeric_cols].round(2)
                
                # Convert to Excel format
                from io import BytesIO
                buffer = BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    detailed_export.to_excel(writer, sheet_name='Payment_Details', index=False)
                
                excel_data = buffer.getvalue()
                
                st.download_button(
                    "📋 Download Detailed Payment Info (Excel)",
                    excel_data,
                    f"{selected_assessor}_Payment_Details_{'_'.join(map(str, selected_chaves))}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Required columns for detailed export not found in the data.")
        
        # Show preview of detailed data
        if 'detailed_export' in locals():
            st.markdown("### 👀 Preview of Detailed Payment Information")
            st.dataframe(detailed_export.head(10), use_container_width=True)
            st.info(f"📊 Total transactions for {selected_assessor}: {len(detailed_export)}")

# --- PROFIT PAGE ---
elif page == "Profit":
    st.title("💰 Profit Summary - Lucro_Empresa by Chave")
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    df_taurus = st.session_state["df_taurus"]
    
    # Chave filter
    st.markdown("### 🔍 Filter Options")
    chave_list = sorted(df_taurus["Chave"].dropna().unique())
    selected_chaves = st.multiselect(
        "Select Chave periods to include (leave empty for all)",
        chave_list,
        default=chave_list  # Default to all selected
    )
    
    # Filter data based on selection
    if selected_chaves:
        df_filtered = df_taurus[df_taurus["Chave"].isin(selected_chaves)]
    else:
        df_filtered = df_taurus
    
    # Group by Chave and sum Lucro_Empresa
    profit_summary = (
        df_filtered.groupby("Chave")["Lucro_Empresa"]
        .sum()
        .reset_index()
        .sort_values("Chave")
    )
    
    # Calculate total sum
    total_sum = profit_summary["Lucro_Empresa"].sum()
    
    st.markdown("### 📈 Lucro_Empresa by Chave")
    
    # Display total sum
    st.metric(
        label="💰 Total Lucro_Empresa",
        value=f"{total_sum:,.2f}",
        help="Sum of all Lucro_Empresa values in the chart below"
    )
    
    # Display bar chart
    st.bar_chart(profit_summary.set_index("Chave"))
    
    # Show summary table
    st.markdown("### 📊 Summary Table")
    st.dataframe(profit_summary.round(2), use_container_width=True )
    
    # Download button
    csv = profit_summary.to_csv(index=False).encode("utf-8")
    filename = f"Profit_by_Chave_{'_'.join(map(str, selected_chaves)) if selected_chaves else 'All'}.csv"
    st.download_button("📥 Download Profit CSV", csv, filename, "text/csv")
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
    
    elif analysis_type == "Seasonal Analysis":
        st.markdown("### 🌊 Seasonal Performance Analysis")
        
        # Extract month from Chave
        df['Month'] = df['Chave'].str[:2].astype(int)
        df['Month_Name'] = df['Month'].apply(lambda x: calendar.month_name[x])
        
        # Monthly aggregation
        seasonal_data = df.groupby('Month_Name').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'AssessorReal': 'nunique'
        }).reset_index()
        
        # Reorder by month
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        seasonal_data['Month_Name'] = pd.Categorical(seasonal_data['Month_Name'], categories=month_order, ordered=True)
        seasonal_data = seasonal_data.sort_values('Month_Name')
        
        # Seasonal charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_seasonal = px.bar(
                seasonal_data,
                x='Month_Name',
                y='Comissão',
                title='📊 Revenue by Month',
                color='Comissão',
                color_continuous_scale='viridis'
            )
            fig_seasonal.update_xaxes(tickangle=45)
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            fig_profit = px.line(
                seasonal_data,
                x='Month_Name',
                y='Lucro_Empresa',
                title='📈 Profit by Month',
                markers=True
            )
            fig_profit.update_xaxes(tickangle=45)
            st.plotly_chart(fig_profit, use_container_width=True)
    
    elif analysis_type == "Growth Analysis":
        st.markdown("### 📊 Growth Rate Analysis")
        
        # Period-over-period growth
        monthly_data = df.groupby('Chave').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'AssessorReal': 'nunique'
        }).reset_index()
        
        monthly_data['Chave_Date'] = monthly_data['Chave'].apply(parse_chave_to_date)
        monthly_data = monthly_data.sort_values('Chave_Date')
        
        # Calculate growth rates
        monthly_data['Revenue_Growth'] = monthly_data['Comissão'].pct_change() * 100
        monthly_data['Profit_Growth'] = monthly_data['Lucro_Empresa'].pct_change() * 100
        monthly_data['Assessor_Growth'] = monthly_data['AssessorReal'].pct_change() * 100
        
        # Growth visualization
        fig_growth = go.Figure()
        
        fig_growth.add_trace(go.Scatter(
            x=monthly_data['Chave'],
            y=monthly_data['Revenue_Growth'],
            mode='lines+markers',
            name='Revenue Growth %',
            line=dict(color='blue')
        ))
        
        fig_growth.add_trace(go.Scatter(
            x=monthly_data['Chave'],
            y=monthly_data['Profit_Growth'],
            mode='lines+markers',
            name='Profit Growth %',
            line=dict(color='green')
        ))
        
        fig_growth.add_trace(go.Scatter(
            x=monthly_data['Chave'],
            y=monthly_data['Assessor_Growth'],
            mode='lines+markers',
            name='Assessor Growth %',
            line=dict(color='orange')
        ))
        
        fig_growth.update_layout(
            title='📈 Month-over-Month Growth Rates',
            xaxis_title='Period',
            yaxis_title='Growth Rate (%)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Growth statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_revenue_growth = monthly_data['Revenue_Growth'].mean()
            st.metric("Avg Revenue Growth", f"{avg_revenue_growth:.1f}%")
        
        with col2:
            avg_profit_growth = monthly_data['Profit_Growth'].mean()
            st.metric("Avg Profit Growth", f"{avg_profit_growth:.1f}%")
        
        with col3:
            avg_assessor_growth = monthly_data['Assessor_Growth'].mean()
            st.metric("Avg Assessor Growth", f"{avg_assessor_growth:.1f}%")

# --- GOAL TRACKING ---
elif page == "🎯 Goal Tracking":
    st.markdown('<h1 class="main-header">🎯 Goal Tracking & Targets</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Goal setting
    st.sidebar.markdown("### 🎯 Set Goals")
    monthly_revenue_goal = st.sidebar.number_input("Monthly Revenue Goal (R$)", value=1000000.0, step=50000.0)
    monthly_profit_goal = st.sidebar.number_input("Monthly Profit Goal (R$)", value=200000.0, step=10000.0)
    assessor_count_goal = st.sidebar.number_input("Active Assessors Goal", value=50, step=5)
    
    # Current period selection
    current_period = st.sidebar.selectbox(
        "Select Current Period",
        sorted(df["Chave"].unique(), reverse=True)
    )
    
    # Calculate current metrics
    current_data = df[df["Chave"] == current_period]
    
    if not current_data.empty:
        current_revenue = current_data["Comissão"].sum()
        current_profit = current_data["Lucro_Empresa"].sum()
        current_assessors = current_data["AssessorReal"].nunique()
        
        # Goal achievement percentages
        revenue_achievement = (current_revenue / monthly_revenue_goal) * 100
        profit_achievement = (current_profit / monthly_profit_goal) * 100
        assessor_achievement = (current_assessors / assessor_count_goal) * 100
        
        # Goal tracking dashboard
        st.markdown("### 📊 Goal Achievement Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_revenue_gauge = create_gauge_chart(
                revenue_achievement,
                "Revenue Achievement %",
                max_val=150
            )
            st.plotly_chart(fig_revenue_gauge, use_container_width=True)
            st.metric(
                "Revenue vs Goal",
                format_currency(current_revenue),
                f"{revenue_achievement:.1f}% of goal"
            )
        
        with col2:
            fig_profit_gauge = create_gauge_chart(
                profit_achievement,
                "Profit Achievement %",
                max_val=150
            )
            st.plotly_chart(fig_profit_gauge, use_container_width=True)
            st.metric(
                "Profit vs Goal",
                format_currency(current_profit),
                f"{profit_achievement:.1f}% of goal"
            )
        
        with col3:
            fig_assessor_gauge = create_gauge_chart(
                assessor_achievement,
                "Assessor Count Achievement %",
                max_val=150
            )
            st.plotly_chart(fig_assessor_gauge, use_container_width=True)
            st.metric(
                "Assessors vs Goal",
                current_assessors,
                f"{assessor_achievement:.1f}% of goal"
            )
        
        # Historical goal tracking
        st.markdown("### 📈 Historical Goal Performance")
        
        # Calculate historical achievements
        historical_data = df.groupby('Chave').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'AssessorReal': 'nunique'
        }).reset_index()
        
        historical_data['Revenue_Achievement'] = (historical_data['Comissão'] / monthly_revenue_goal) * 100
        historical_data['Profit_Achievement'] = (historical_data['Lucro_Empresa'] / monthly_profit_goal) * 100
        historical_data['Assessor_Achievement'] = (historical_data['AssessorReal'] / assessor_count_goal) * 100
        
        # Sort by period
        historical_data['Chave_Date'] = historical_data['Chave'].apply(parse_chave_to_date)
        historical_data = historical_data.sort_values('Chave_Date')
        
        # Goal achievement chart
        fig_goals = go.Figure()
        
        fig_goals.add_trace(go.Scatter(
            x=historical_data['Chave'],
            y=historical_data['Revenue_Achievement'],
            mode='lines+markers',
            name='Revenue Achievement %',
            line=dict(color='blue')
        ))
        
        fig_goals.add_trace(go.Scatter(
            x=historical_data['Chave'],
            y=historical_data['Profit_Achievement'],
            mode='lines+markers',
            name='Profit Achievement %',
            line=dict(color='green')
        ))
        
        fig_goals.add_trace(go.Scatter(
            x=historical_data['Chave'],
            y=historical_data['Assessor_Achievement'],
            mode='lines+markers',
            name='Assessor Achievement %',
            line=dict(color='orange')
        ))
        
        # Add goal line
        fig_goals.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Goal (100%)")
        
        fig_goals.update_layout(
            title='🎯 Goal Achievement Over Time',
            xaxis_title='Period',
            yaxis_title='Achievement (%)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_goals, use_container_width=True)

# --- PROFIT CENTER ---
elif page == "💰 Profit Center":
    st.markdown('<h1 class="main-header">💰 Profit Center Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Profit analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Profit Overview", "🎯 Profit by Category", "👤 Assessor Profitability"])
    
    with tab1:
        st.markdown("### 💰 Overall Profit Analysis")
        
        # Key profit metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = df["Comissão"].sum()
        total_profit = df["Lucro_Empresa"].sum()
        total_pix = df["Pix_Assessor"].sum()
        profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
        
        with col1:
            st.metric("Total Revenue", format_currency(total_revenue))
        with col2:
            st.metric("Total Profit", format_currency(total_profit))
        with col3:
            st.metric("Total Pix Assessor", format_currency(total_pix))
        with col4:
            st.metric("Profit Margin", f"{profit_margin:.2f}%")
        
        # Profit evolution
        monthly_profit = df.groupby('Chave').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'Pix_Assessor': 'sum'
        }).reset_index()
        
        monthly_profit['Profit_Margin'] = (monthly_profit['Lucro_Empresa'] / monthly_profit['Comissão']) * 100
        monthly_profit['Chave_Date'] = monthly_profit['Chave'].apply(parse_chave_to_date)
        monthly_profit = monthly_profit.sort_values('Chave_Date')
        
        # Profit waterfall chart
        fig_waterfall = go.Figure()
        
        fig_waterfall.add_trace(go.Bar(
            x=monthly_profit['Chave'],
            y=monthly_profit['Comissão'],
            name='Revenue',
            marker_color='lightblue'
        ))
        
        fig_waterfall.add_trace(go.Bar(
            x=monthly_profit['Chave'],
            y=monthly_profit['Lucro_Empresa'],
            name='Profit',
            marker_color='green'
        ))
        
        fig_waterfall.add_trace(go.Bar(
            x=monthly_profit['Chave'],
            y=monthly_profit['Pix_Assessor'],
            name='Pix Assessor',
            marker_color='orange'
        ))
        
        fig_waterfall.update_layout(
            title='💰 Revenue vs Profit vs Pix by Period',
            xaxis_title='Period',
            yaxis_title='Amount (R$)',
            barmode='group'
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Profit margin trend
        fig_margin = px.line(
            monthly_profit,
            x='Chave',
            y='Profit_Margin',
            title='📈 Profit Margin Trend (%)',
            markers=True,
            line_shape='spline'
        )
        fig_margin.update_yaxes(title_text="Profit Margin (%)")
        st.plotly_chart(fig_margin, use_container_width=True)
    
    with tab2:
        st.markdown("### 🎯 Profit by Category")
        
        # Category profit analysis
        category_profit = df.groupby('Categoria').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'Pix_Assessor': 'sum',
            'Chave': 'count'
        }).rename(columns={'Chave': 'Transaction_Count'})
        
        category_profit['Profit_Margin'] = (category_profit['Lucro_Empresa'] / category_profit['Comissão']) * 100
        category_profit['Avg_Profit_per_Transaction'] = category_profit['Lucro_Empresa'] / category_profit['Transaction_Count']
        
        # Sort by profit
        category_profit = category_profit.sort_values('Lucro_Empresa', ascending=False)
        
        # Category profitability chart
        fig_category = px.bar(
            category_profit.reset_index(),
            x='Categoria',
            y='Lucro_Empresa',
            color='Profit_Margin',
            title='💰 Profit by Category',
            color_continuous_scale='RdYlGn'
        )
        fig_category.update_xaxes(tickangle=45)
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Category details table
        st.markdown("### 📋 Category Profitability Details")
        
        # Format for display
        display_category = category_profit.copy()
        display_category['Comissão'] = display_category['Comissão'].apply(format_currency)
        display_category['Lucro_Empresa'] = display_category['Lucro_Empresa'].apply(format_currency)
        display_category['Pix_Assessor'] = display_category['Pix_Assessor'].apply(format_currency)
        display_category['Avg_Profit_per_Transaction'] = display_category['Avg_Profit_per_Transaction'].apply(format_currency)
        display_category['Profit_Margin'] = display_category['Profit_Margin'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_category, use_container_width=True)
    
    with tab3:
        st.markdown("### 👤 Assessor Profitability")
        
        # Assessor profit analysis
        assessor_profit = df.groupby('AssessorReal').agg({
            'Comissão': 'sum',
            'Lucro_Empresa': 'sum',
            'Pix_Assessor': 'sum',
            'Chave': 'count'
        }).rename(columns={'Chave': 'Transaction_Count'})
        
        assessor_profit['Profit_Margin'] = (assessor_profit['Lucro_Empresa'] / assessor_profit['Comissão']) * 100
        assessor_profit['Profit_per_Transaction'] = assessor_profit['Lucro_Empresa'] / assessor_profit['Transaction_Count']
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            min_transactions = st.number_input("Min Transactions", min_value=1, value=10)
        with col2:
            top_n_assessors = st.number_input("Show Top N Assessors", min_value=5, max_value=50, value=20)
        
        # Filter and sort
        filtered_assessors = assessor_profit[assessor_profit['Transaction_Count'] >= min_transactions]
        filtered_assessors = filtered_assessors.sort_values('Lucro_Empresa', ascending=False).head(top_n_assessors)
        
        # Profitability scatter plot
        fig_scatter = px.scatter(
            filtered_assessors.reset_index(),
            x='Comissão',
            y='Lucro_Empresa',
            size='Transaction_Count',
            color='Profit_Margin',
            hover_data=['AssessorReal'],
            title='💰 Assessor Profitability Analysis',
            labels={'Comissão': 'Revenue (R$)', 'Lucro_Empresa': 'Profit (R$)'},
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top profitable assessors
        st.markdown("### 🏆 Most Profitable Assessors")
        
        # Format for display
        display_assessors = filtered_assessors.copy()
        display_assessors['Comissão'] = display_assessors['Comissão'].apply(format_currency)
        display_assessors['Lucro_Empresa'] = display_assessors['Lucro_Empresa'].apply(format_currency)
        display_assessors['Pix_Assessor'] = display_assessors['Pix_Assessor'].apply(format_currency)
        display_assessors['Profit_per_Transaction'] = display_assessors['Profit_per_Transaction'].apply(format_currency)
        display_assessors['Profit_Margin'] = display_assessors['Profit_Margin'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_assessors, use_container_width=True)

# --- REPORTS ---
elif page == "📋 Reports":
    st.markdown('<h1 class="main-header">📋 Reports & Export</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file first.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Report generation options
    st.markdown("### 📊 Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Assessor Performance", "Category Analysis", "Profit Report", "Custom Report"]
        )
    
    with col2:
        # Date range for report
        chave_list = sorted(df["Chave"].unique())
        selected_period = st.multiselect(
            "Select Reporting Period",
            chave_list,
            default=chave_list
        )
    
    if selected_period:
        df_report = df[df["Chave"].isin(selected_period)]
        
        if report_type == "Executive Summary":
            st.markdown("### 📊 Executive Summary Report")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Revenue", format_currency(df_report["Comissão"].sum()))
            with col2:
                st.metric("Total Profit", format_currency(df_report["Lucro_Empresa"].sum()))
            with col3:
                st.metric("Active Assessors", df_report["AssessorReal"].nunique())
            with col4:
                st.metric("Total Transactions", len(df_report))
            
            # Summary tables
            summary_data = {
                'Monthly Performance': df_report.groupby('Chave').agg({
                    'Comissão': 'sum',
                    'Lucro_Empresa': 'sum',
                    'AssessorReal': 'nunique'
                }).round(2),
                'Top 10 Assessors': df_report.groupby('AssessorReal')['Comissão'].sum().nlargest(10),
                'Category Performance': df_report.groupby('Categoria').agg({
                    'Comissão': 'sum',
                    'Lucro_Empresa': 'sum'
                }).round(2)
            }
            
            # Create Excel report
            from io import BytesIO
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                for sheet_name, data in summary_data.items():
                    data.to_excel(writer, sheet_name=sheet_name)
            
            st.download_button(
                "📥 Download Executive Summary Report",
                buffer.getvalue(),
                f"Executive_Summary_{'-'.join(selected_period)}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        elif report_type == "Custom Report":
            st.markdown("### 🛠️ Custom Report Builder")
            
            # Custom report options
            col1, col2 = st.columns(2)
            
            with col1:
                groupby_column = st.selectbox(
                    "Group By",
                    ["AssessorReal", "Categoria", "Chave"]
                )
            
            with col2:
                aggregation_columns = st.multiselect(
                    "Metrics to Include",
                    ["Comissão", "Lucro_Empresa", "Pix_Assessor", "Tributo_Retido"],
                    default=["Comissão", "Lucro_Empresa"]
                )
            
            if aggregation_columns:
                custom_report = df_report.groupby(groupby_column)[aggregation_columns].agg(['sum', 'mean', 'count'])
                custom_report.columns = [f"{col}_{agg}" for col, agg in custom_report.columns]
                
                st.dataframe(custom_report, use_container_width=True)
                
                # Export custom report
                csv_data = custom_report.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Download Custom Report",
                    csv_data,
                    f"Custom_Report_{groupby_column}.csv",
                    "text/csv"
                )
    
    # Automated report scheduling (placeholder)
    st.markdown("### 🔄 Automated Reports")
    st.info("💡 **Future Feature**: Set up automated report generation and email delivery schedules.")
    
    # Quick stats
    st.markdown("### 📈 Quick Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", len(df))
    with col2:
        st.metric("Time Periods", df["Chave"].nunique())
    with col3:
        st.metric("Categories", df["Categoria"].nunique())

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Taurus Analytics Dashboard**")
    st.sidebar.markdown("Version 2.0")
    st.sidebar.markdown("Built with Streamlit")

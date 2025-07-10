import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- CONFIG ---
st.set_page_config(
    page_title="Pix Assessor Dashboard", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 📂 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["📤 Upload", "📊 Macro View", "👤 Assessor View", "💰 Profit Analysis"])

# --- SESSION STORAGE ---
if "df_all" not in st.session_state:
    st.session_state["df_all"] = None
if "skipped_sheets" not in st.session_state:
    st.session_state["skipped_sheets"] = []
if "uploaded_file_data" not in st.session_state:
    st.session_state["uploaded_file_data"] = None

# --- HELPER FUNCTIONS ---
def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

def format_currency(value):
    """Format currency values"""
    return f"R$ {value:,.2f}" if pd.notna(value) else "R$ 0.00"

def create_performance_chart(df, x_col, y_col, title, chart_type="bar"):
    """Create performance charts"""
    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=title,
                     color=y_col, color_continuous_scale="viridis")
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
    elif chart_type == "pie":
        fig = px.pie(df, values=y_col, names=x_col, title=title)
    
    fig.update_layout(
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# --- UPLOAD PAGE ---
if page == "📤 Upload":
    st.markdown('<div class="main-header"><h1>📤 Upload Excel File</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload the 'Utor_Detalhado.xlsx' file", 
            type=["xlsx"],
            help="Please upload your Excel file containing assessor data"
        )
    
    if uploaded_file:
        st.session_state["uploaded_file_data"] = uploaded_file
        
        with st.spinner("Processing your file..."):
            try:
                xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                all_sheets = xls.sheet_names
                required_cols = {"Chave", "AssessorReal", "Pix_Assessor"}
                data = []
                skipped_sheets = []
                
                progress_bar = st.progress(0)
                for i, sheet in enumerate(all_sheets):
                    try:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        if not required_cols.issubset(df.columns):
                            skipped_sheets.append(sheet)
                            continue
                        
                        # Include all available columns
                        available_cols = ["Chave", "AssessorReal", "Pix_Assessor"]
                        optional_cols = ["Cliente", "Comissão", "Imposto", "Valor Liquido", "Lucro_Empresa", "Chave_Interna"]
                        
                        for col in optional_cols:
                            if col in df.columns:
                                available_cols.append(col)
                        
                        df = df[available_cols]
                        df["Distribuidor"] = sheet
                        data.append(df)
                    except Exception as e:
                        skipped_sheets.append(f"{sheet} (Error: {str(e)})")
                    
                    progress_bar.progress((i + 1) / len(all_sheets))
                
                if not data:
                    st.error("❌ No valid sheets found. Please check columns.")
                else:
                    df_all = pd.concat(data, ignore_index=True)
                    st.session_state["df_all"] = df_all
                    st.session_state["skipped_sheets"] = skipped_sheets
                    
                    # Success message with file info
                    st.success("✅ Data successfully loaded!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Rows", len(df_all))
                    with col2:
                        st.metric("📋 Sheets Processed", len(data))
                    with col3:
                        st.metric("👥 Unique Assessors", df_all["AssessorReal"].nunique())
                    
                    # Show loaded columns info
                    st.info(f"📋 Loaded columns: {', '.join(df_all.columns)}")
                    
                    if skipped_sheets:
                        with st.expander("⚠️ Skipped Sheets", expanded=False):
                            for s in skipped_sheets:
                                st.write(f"- {s}")
                    
                    # Preview data
                    with st.expander("👁️ Preview Data", expanded=False):
                        st.dataframe(df_all.head(10), use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

# --- MACRO VIEW PAGE ---
elif page == "📊 Macro View":
    st.markdown('<div class="main-header"><h1>📊 Macro Summary View</h1></div>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file in the Upload section.")
        st.stop()
    
    df_all = st.session_state["df_all"]
    
    # Filters section
    st.markdown("### 🔍 Filter Options")
    col1, col2 = st.columns(2)
    
    with col1:
        chave_list = sorted(df_all["Chave"].dropna().unique())
        selected_chaves = st.multiselect(
            "Select Chave periods", 
            chave_list, 
            default=chave_list[:1] if chave_list else []
        )
    
    with col2:
        assessor_list = sorted(df_all["AssessorReal"].dropna().unique())
        selected_assessors = st.multiselect(
            "Select Assessors (leave empty for all)", 
            assessor_list,
            default=[]
        )
    
    if selected_chaves:
        # Filter data
        df_filtered = df_all[df_all["Chave"].isin(selected_chaves)]
        if selected_assessors:
            df_filtered = df_filtered[df_filtered["AssessorReal"].isin(selected_assessors)]
        
        # Key Performance Indicators
        st.markdown("### 📈 Key Performance Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            if "Comissão" in df_filtered.columns:
                total_commission = df_filtered["Comissão"].sum()
                st.metric("💰 Total Commission", format_currency(total_commission))
            else:
                st.metric("💰 Total Commission", "N/A")
        
        with kpi_col2:
            if "Imposto" in df_filtered.columns:
                total_tax = df_filtered["Imposto"].sum()
                st.metric("🏛️ Total Tax", format_currency(total_tax))
            else:
                st.metric("🏛️ Total Tax", "N/A")
        
        with kpi_col3:
            total_pix = df_filtered["Pix_Assessor"].sum()
            st.metric("💳 Total Pix Assessor", format_currency(total_pix))
        
        with kpi_col4:
            if "Lucro_Empresa" in df_filtered.columns:
                total_profit = df_filtered["Lucro_Empresa"].sum()
                st.metric("📊 Company Profit", format_currency(total_profit))
            else:
                st.metric("📊 Company Profit", "N/A")
        
        # Charts Section
        st.markdown("### 📊 Performance Analytics")
        
        # Create charts based on available data
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Pix Assessor by Distribuidor
            pix_by_dist = df_filtered.groupby("Distribuidor")["Pix_Assessor"].sum().reset_index()
            pix_by_dist = pix_by_dist.sort_values("Pix_Assessor", ascending=False)
            
            fig_pix = create_performance_chart(pix_by_dist, "Distribuidor", "Pix_Assessor", 
                                             "Pix Assessor by Distribuidor", "bar")
            st.plotly_chart(fig_pix, use_container_width=True)
        
        with chart_col2:
            # Top Assessors Performance
            top_assessors = df_filtered.groupby("AssessorReal")["Pix_Assessor"].sum().reset_index()
            top_assessors = top_assessors.sort_values("Pix_Assessor", ascending=False).head(10)
            
            fig_assessors = create_performance_chart(top_assessors, "AssessorReal", "Pix_Assessor", 
                                                   "Top 10 Assessors by Pix", "bar")
            fig_assessors.update_xaxes(tickangle=45)
            st.plotly_chart(fig_assessors, use_container_width=True)
        
        # Additional charts if commission data is available
        if "Comissão" in df_filtered.columns:
            chart_col3, chart_col4 = st.columns(2)
            
            with chart_col3:
                # Commission vs Profit Analysis
                comm_profit = df_filtered.groupby("Distribuidor").agg({
                    "Comissão": "sum",
                    "Lucro_Empresa": "sum" if "Lucro_Empresa" in df_filtered.columns else lambda x: 0
                }).reset_index()
                
                fig_comm = go.Figure()
                fig_comm.add_trace(go.Bar(name='Commission', x=comm_profit["Distribuidor"], y=comm_profit["Comissão"]))
                if "Lucro_Empresa" in df_filtered.columns:
                    fig_comm.add_trace(go.Bar(name='Profit', x=comm_profit["Distribuidor"], y=comm_profit["Lucro_Empresa"]))
                
                fig_comm.update_layout(title="Commission vs Profit by Distribuidor", barmode='group')
                st.plotly_chart(fig_comm, use_container_width=True)
            
            with chart_col4:
                # Profit Margin Analysis
                if "Lucro_Empresa" in df_filtered.columns:
                    margin_data = df_filtered.groupby("Distribuidor").agg({
                        "Comissão": "sum",
                        "Lucro_Empresa": "sum"
                    }).reset_index()
                    margin_data["Profit_Margin"] = (margin_data["Lucro_Empresa"] / margin_data["Comissão"] * 100).round(2)
                    
                    fig_margin = px.bar(margin_data, x="Distribuidor", y="Profit_Margin", 
                                      title="Profit Margin % by Distribuidor")
                    st.plotly_chart(fig_margin, use_container_width=True)
        
        # Summary Table
        st.markdown("### 📋 Summary Table")
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
        
        st.dataframe(pivot_df.round(2), use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
            filename = f"Pix_Summary_{'_'.join(map(str, selected_chaves))}.csv"
            st.download_button("📥 Download Summary CSV", csv, filename, "text/csv")
        
        with col2:
            csv_all = df_filtered.to_csv(index=False).encode("utf-8")
            filename_raw = f"FullData_{'_'.join(map(str, selected_chaves))}.csv"
            st.download_button("📦 Download Full Data", csv_all, filename_raw, "text/csv")
    
    else:
        st.warning("Please select at least one Chave period.")

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<div class="main-header"><h1>👤 Assessor Breakdown View</h1></div>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file in the Upload section.")
        st.stop()
    
    df_all = st.session_state["df_all"]
    
    # Filters section
    st.markdown("### 🔍 Filter Options")
    col1, col2 = st.columns(2)
    
    with col1:
        assessor_list = sorted(df_all["AssessorReal"].dropna().unique())
        selected_assessor = st.selectbox("Select AssessorReal", assessor_list)
    
    with col2:
        chave_list = sorted(df_all["Chave"].dropna().unique())
        selected_months = st.multiselect(
            "Select Months (Chave periods)", 
            chave_list,
            default=chave_list
        )
    
    # Filter data
    df_filtered = df_all[df_all["AssessorReal"] == selected_assessor]
    if selected_months:
        df_filtered = df_filtered[df_filtered["Chave"].isin(selected_months)]
    
    if df_filtered.empty:
        st.warning("No data for selected AssessorReal and/or months.")
    else:
        # Key Highlights Section
        st.markdown("### 🎯 Key Highlights")
        
        highlight_col1, highlight_col2, highlight_col3, highlight_col4 = st.columns(4)
        
        with highlight_col1:
            if "Comissão" in df_filtered.columns:
                total_commission = df_filtered["Comissão"].sum()
                st.metric("💰 Commission", format_currency(total_commission))
            else:
                st.metric("💰 Commission", "N/A")
        
        with highlight_col2:
            if "Imposto" in df_filtered.columns:
                total_tax = df_filtered["Imposto"].sum()
                st.metric("🏛️ Tax", format_currency(total_tax))
            else:
                st.metric("🏛️ Tax", "N/A")
        
        with highlight_col3:
            total_pix = df_filtered["Pix_Assessor"].sum()
            st.metric("💳 Pix Assessor", format_currency(total_pix))
        
        with highlight_col4:
            if "Lucro_Empresa" in df_filtered.columns:
                total_profit = df_filtered["Lucro_Empresa"].sum()
                st.metric("📊 Profit", format_currency(total_profit))
            else:
                st.metric("📊 Profit", "N/A")
        
        # Performance Charts
        st.markdown("### 📊 Performance Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Performance by month
            if len(selected_months) > 1:
                monthly_perf = df_filtered.groupby("Chave")["Pix_Assessor"].sum().reset_index()
                monthly_perf = monthly_perf.sort_values("Chave")
                
                fig_monthly = create_performance_chart(monthly_perf, "Chave", "Pix_Assessor", 
                                                     "Monthly Performance", "line")
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with chart_col2:
            # Performance by Distribuidor
            dist_perf = df_filtered.groupby("Distribuidor")["Pix_Assessor"].sum().reset_index()
            dist_perf = dist_perf.sort_values("Pix_Assessor", ascending=False)
            
            fig_dist = create_performance_chart(dist_perf, "Distribuidor", "Pix_Assessor", 
                                              "Performance by Distribuidor", "bar")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Main pivot table
        st.markdown("### 📋 Detailed Breakdown")
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
        
        # Distribuidor totals
        st.markdown("### 📊 Total by Distribuidor")
        sheet_totals = (
            df_filtered.groupby("Distribuidor")["Pix_Assessor"]
            .sum()
            .reset_index()
            .sort_values("Pix_Assessor", ascending=False)
        )
        sheet_totals["Pix_Assessor"] = sheet_totals["Pix_Assessor"].round(2)
        
        # Add grand total
        grand_total = sheet_totals["Pix_Assessor"].sum()
        total_row = pd.DataFrame({"Distribuidor": ["GRAND TOTAL"], "Pix_Assessor": [grand_total]})
        sheet_totals_with_total = pd.concat([sheet_totals, total_row], ignore_index=True)
        
        st.dataframe(sheet_totals_with_total, use_container_width=True, hide_index=True)
        
        # Export section
        st.markdown("### 📥 Export Options")
        
        # Client information
        if "Cliente" in df_filtered.columns:
            st.success("✅ 'Cliente' column found and will be included in exports!")
            export_info = f"Export will include: {', '.join(df_filtered.columns)}"
        else:
            st.warning("⚠️ 'Cliente' column not found in the data.")
            export_info = f"Export will include: {', '.join(df_filtered.columns)}"
        
        st.info(export_info)
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Summary CSV", csv, f"{selected_assessor}_Summary.csv", "text/csv")
        
        with col2:
            csv_totals = sheet_totals_with_total.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Totals CSV", csv_totals, f"{selected_assessor}_Totals.csv", "text/csv")
        
        with col3:
            csv_all = df_filtered.to_csv(index=False).encode("utf-8")
            filename_raw = f"FullData_{selected_assessor}.csv"
            st.download_button("📦 Download Full Data", csv_all, filename_raw, "text/csv")

# --- PROFIT PAGE ---
elif page == "💰 Profit Analysis":
    st.markdown('<div class="main-header"><h1>💰 Profit Analysis</h1></div>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None or st.session_state["uploaded_file_data"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    uploaded_file = st.session_state["uploaded_file_data"]
    
    with st.spinner("Analyzing profit data..."):
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            all_sheets = xls.sheet_names
            lucro_data = []
            
            for sheet in all_sheets:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    if {"Chave", "Lucro_Empresa"}.issubset(df.columns):
                        temp = df[["Chave", "Lucro_Empresa"]].copy()
                        temp["Distribuidor"] = sheet
                        lucro_data.append(temp)
                except:
                    continue
            
            if not lucro_data:
                st.error("❌ No sheets contained both 'Chave' and 'Lucro_Empresa' columns.")
            else:
                df_lucro = pd.concat(lucro_data, ignore_index=True)
                
                # Filter options
                st.markdown("### 🔍 Filter Options")
                chave_list = sorted(df_lucro["Chave"].dropna().unique())
                selected_chaves = st.multiselect(
                    "Select Chave periods (leave empty for all)",
                    chave_list,
                    default=chave_list
                )
                
                df_lucro_filtered = df_lucro[df_lucro["Chave"].isin(selected_chaves)] if selected_chaves else df_lucro
                
                # Key metrics
                st.markdown("### 📈 Profit Overview")
                
                total_profit = df_lucro_filtered["Lucro_Empresa"].sum()
                avg_profit = df_lucro_filtered["Lucro_Empresa"].mean()
                max_profit = df_lucro_filtered["Lucro_Empresa"].max()
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("💰 Total Profit", format_currency(total_profit))
                with metric_col2:
                    st.metric("📊 Average Profit", format_currency(avg_profit))
                with metric_col3:
                    st.metric("🏆 Maximum Profit", format_currency(max_profit))
                
                # Charts
                st.markdown("### 📊 Profit Analytics")
                
                # Profit by Chave
                lucro_summary = (
                    df_lucro_filtered.groupby("Chave")["Lucro_Empresa"]
                    .sum()
                    .reset_index()
                    .sort_values("Chave")
                )
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    fig_line = create_performance_chart(lucro_summary, "Chave", "Lucro_Empresa", 
                                                      "Profit Trend by Chave", "line")
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with chart_col2:
                    fig_bar = create_performance_chart(lucro_summary, "Chave", "Lucro_Empresa", 
                                                     "Profit by Chave", "bar")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Profit by Distribuidor
                dist_profit = (
                    df_lucro_filtered.groupby("Distribuidor")["Lucro_Empresa"]
                    .sum()
                    .reset_index()
                    .sort_values("Lucro_Empresa", ascending=False)
                )
                
                st.markdown("### 📊 Profit by Distribuidor")
                fig_dist_profit = create_performance_chart(dist_profit, "Distribuidor", "Lucro_Empresa", 
                                                         "Profit by Distribuidor", "bar")
                st.plotly_chart(fig_dist_profit, use_container_width=True)
                
                # Summary table
                st.markdown("### 📋 Summary Table")
                st.dataframe(lucro_summary.round(2), use_container_width=True)
                
                # Export
                st.markdown("### 📥 Export Options")
                csv = lucro_summary.to_csv(index=False).encode("utf-8")
                filename = f"Profit_Analysis_{'_'.join(map(str, selected_chaves)) if selected_chaves else 'All'}.csv"
                st.download_button("📥 Download Profit Analysis", csv, filename, "text/csv")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- SIDEBAR INFO ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dashboard Info")
st.sidebar.markdown("**Version:** 2.0")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- 📈 Advanced Analytics")
st.sidebar.markdown("- 🎯 Key Highlights")
st.sidebar.markdown("- 📊 Interactive Charts")
st.sidebar.markdown("- 💰 Profit Analysis")
st.sidebar.markdown("- 📥 Enhanced Exports")

if st.session_state["df_all"] is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Current Data")
    st.sidebar.markdown(f"**Rows:** {len(st.session_state['df_all'])}")
    st.sidebar.markdown(f"**Assessors:** {st.session_state['df_all']['AssessorReal'].nunique()}")
    st.sidebar.markdown(f"**Periods:** {st.session_state['df_all']['Chave'].nunique()}")

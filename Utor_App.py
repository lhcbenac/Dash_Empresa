import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import traceback
import logging
from io import BytesIO
from typing import Optional, Tuple, List

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .error-box {
        background-color: #ffecec;
        border: 1px solid #ff6b6b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- ERROR HANDLER CLASS ---
class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

def safe_operation(func, *args, default=None, error_message="An error occurred", **kwargs):
    """
    Safely execute a function with error handling
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ {error_message}: {str(e)}")
        return default

def validate_dataframe(df: pd.DataFrame, required_cols: set, sheet_name: str = "Unknown") -> Tuple[bool, str]:
    """
    Validate DataFrame structure and content
    """
    try:
        if df is None or df.empty:
            return False, f"Sheet '{sheet_name}' is empty"
        
        # Check columns (handling potential whitespace issues handled in main loop)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return False, f"Sheet '{sheet_name}' missing columns: {', '.join(missing_cols)}"
        
        return True, ""
    except Exception as e:
        logger.error(f"Validation error for sheet '{sheet_name}': {str(e)}")
        return False, f"Error validating sheet '{sheet_name}': {str(e)}"

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
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

# --- HELPER FUNCTIONS ---
def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(label=title, value=value, delta=delta, delta_color=delta_color)
    except Exception as e:
        logger.error(f"Error creating metric card: {str(e)}")
        st.error(f"❌ Error displaying metric: {str(e)}")

def format_currency(value):
    """Format currency values with error handling"""
    try:
        if pd.isna(value) or value is None:
            return "R$ 0.00"
        return f"R$ {float(value):,.2f}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting currency value '{value}': {str(e)}")
        return "R$ 0.00"

def create_performance_chart(df, x_col, y_col, title, chart_type="bar"):
    """Create performance charts with error handling"""
    try:
        if df is None or df.empty:
            st.warning(f"No data available for chart: {title}")
            return None
        
        # Validate columns exist
        if x_col not in df.columns or y_col not in df.columns:
            st.error(f"Missing columns for chart: {title}")
            return None
        
        # Convert y_col to numeric
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title,
                         color=y_col, color_continuous_scale="viridis")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
        elif chart_type == "pie":
            fig = px.pie(df, values=y_col, names=x_col, title=title)
        else:
            st.error(f"Unknown chart type: {chart_type}")
            return None
        
        fig.update_layout(
            title_font_size=16,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating performance chart '{title}': {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ Error creating chart '{title}': {str(e)}")
        return None

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
                # Validate file
                if uploaded_file.size == 0:
                    st.error("❌ The uploaded file is empty. Please upload a valid Excel file.")
                    st.stop()
                
                logger.info(f"Loading Excel file: {uploaded_file.name}")
                xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                all_sheets = xls.sheet_names
                
                if not all_sheets:
                    st.error("❌ The Excel file contains no sheets.")
                    st.stop()
                
                logger.info(f"Found {len(all_sheets)} sheets: {all_sheets}")
                
                required_cols = {"Chave", "AssessorReal", "Pix_Assessor"}
                data = []
                skipped_sheets = []
                
                progress_bar = st.progress(0)
                for i, sheet in enumerate(all_sheets):
                    try:
                        logger.info(f"Processing sheet: {sheet}")
                        df = pd.read_excel(xls, sheet_name=sheet)
                        
                        # --- FIX 1: Clean column names to remove spaces ---
                        df.columns = df.columns.astype(str).str.strip()
                        
                        # --- FIX 2: Rename 'Valor Liquido' immediately ---
                        if "Valor Liquido" in df.columns:
                            df = df.rename(columns={"Valor Liquido": "VALOR_LIQUIDO_IR"})
                        
                        # Validate sheet
                        is_valid, error_msg = validate_dataframe(df, required_cols, sheet)
                        if not is_valid:
                            skipped_sheets.append(f"{sheet} - {error_msg}")
                            logger.warning(f"Skipped sheet '{sheet}': {error_msg}")
                            progress_bar.progress((i + 1) / len(all_sheets))
                            continue
                        
                        # --- FIX 3: Enforce Standard Structure (The "Cookie Cutter") ---
                        # UPDATED: Added "Categoria" and "Tipo Receita" to this list
                        target_columns = [
                            "Chave", 
                            "AssessorReal", 
                            "Pix_Assessor",
                            "Cliente", 
                            "Conta",          
                            "Ativo",
                            "Categoria",      # <--- ADDED
                            "Tipo Receita",   # <--- ADDED
                            "VALOR_LIQUIDO_IR",
                            "Comissão", 
                            "Imposto", 
                            "Lucro_Empresa", 
                            "Chave_Interna", 
                            "Data Receita"
                        ]
                        
                        for col in target_columns:
                            if col not in df.columns:
                                df[col] = None # Fill missing columns with None/NaN
                        
                        # Select exactly the target columns
                        df = df[target_columns].copy()

                        # Convert numeric columns
                        numeric_cols = ["Pix_Assessor", "Comissão", "Imposto", "VALOR_LIQUIDO_IR", "Lucro_Empresa"]
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df["Distribuidor"] = sheet
                        data.append(df)
                        logger.info(f"Successfully processed sheet '{sheet}' with {len(df)} rows")
                        
                    except Exception as e:
                        error_detail = f"{sheet} (Error: {str(e)})"
                        skipped_sheets.append(error_detail)
                        logger.error(f"Error processing sheet '{sheet}': {str(e)}\n{traceback.format_exc()}")
                    
                    progress_bar.progress((i + 1) / len(all_sheets))
                
                if not data:
                    error_msg = "No valid sheets found. Please check that your Excel file contains sheets with the required columns: Chave, AssessorReal, Pix_Assessor"
                    st.error(f"❌ {error_msg}")
                    st.warning("⚠️ Skipped sheets details:")
                    for s in skipped_sheets:
                        st.write(f"- {s}")
                    st.stop()
                
                try:
                    df_all = pd.concat(data, ignore_index=True)
                    logger.info(f"Successfully concatenated {len(data)} sheets with {len(df_all)} total rows")
                except Exception as e:
                    st.error(f"❌ Error combining sheets: {str(e)}")
                    logger.error(f"Concatenation error: {str(e)}\n{traceback.format_exc()}")
                    st.stop()
                
                st.session_state["df_all"] = df_all
                st.session_state["skipped_sheets"] = skipped_sheets
                
                # Success message with file info
                st.success("✅ Data successfully loaded! (Sheets Standardized)")
                
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
                error_detail = f"Error reading file: {str(e)}"
                st.session_state["last_error"] = error_detail
                logger.error(f"{error_detail}\n{traceback.format_exc()}")
                
                st.error(f"❌ {error_detail}")
                with st.expander("🔍 Detailed Error Information"):
                    st.code(traceback.format_exc(), language="python")
                
                st.warning("⚠️ Please try uploading the file again or check:")
                st.write("- The file format is .xlsx")
                st.write("- The file contains the required columns: Chave, AssessorReal, Pix_Assessor")
                st.write("- The file is not corrupted")

# --- MACRO VIEW PAGE ---
elif page == "📊 Macro View":
    st.markdown('<div class="main-header"><h1>📊 Macro Summary View</h1></div>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file in the Upload section.")
        st.stop()
    
    try:
        df_all = st.session_state["df_all"]
        
        if df_all is None or df_all.empty:
            st.error("❌ No data available. Please upload a file first.")
            st.stop()
        
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
            df_filtered = df_all[df_all["Chave"].isin(selected_chaves)].copy()
            if selected_assessors:
                df_filtered = df_filtered[df_filtered["AssessorReal"].isin(selected_assessors)]
            
            # Key Performance Indicators
            st.markdown("### 📈 Key Performance Indicators")
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                if "Comissão" in df_filtered.columns:
                    total_commission = safe_operation(lambda: df_filtered["Comissão"].sum(), default=0)
                    st.metric("💰 Total Commission", format_currency(total_commission))
                else:
                    st.metric("💰 Total Commission", "N/A")
            
            with kpi_col2:
                if "Imposto" in df_filtered.columns:
                    total_tax = safe_operation(lambda: df_filtered["Imposto"].sum(), default=0)
                    st.metric("🏛️ Total Tax", format_currency(total_tax))
                else:
                    st.metric("🏛️ Total Tax", "N/A")
            
            with kpi_col3:
                total_pix = safe_operation(lambda: df_filtered["Pix_Assessor"].sum(), default=0)
                st.metric("💳 Total Pix Assessor", format_currency(total_pix))
            
            with kpi_col4:
                if "Lucro_Empresa" in df_filtered.columns:
                    total_profit = safe_operation(lambda: df_filtered["Lucro_Empresa"].sum(), default=0)
                    st.metric("📊 Company Profit", format_currency(total_profit))
                else:
                    st.metric("📊 Company Profit", "N/A")
            
            # Summary Table
            st.markdown("### 📋 Summary Table")
            try:
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
            except Exception as e:
                logger.error(f"Error creating pivot table: {str(e)}")
                st.error(f"❌ Error creating summary table: {str(e)}")
            
            # Charts Section
            st.markdown("### 📊 Performance Analytics")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                try:
                    pix_by_dist = df_filtered.groupby("Distribuidor")["Pix_Assessor"].sum().reset_index()
                    pix_by_dist = pix_by_dist.sort_values("Pix_Assessor", ascending=False)
                    
                    fig_pix = create_performance_chart(pix_by_dist, "Distribuidor", "Pix_Assessor", 
                                                     "Pix Assessor by Distribuidor", "bar")
                    if fig_pix:
                        st.plotly_chart(fig_pix, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating pix chart: {str(e)}")
            
            with chart_col2:
                try:
                    top_assessors = df_filtered.groupby("AssessorReal")["Pix_Assessor"].sum().reset_index()
                    top_assessors = top_assessors.sort_values("Pix_Assessor", ascending=False).head(10)
                    
                    fig_assessors = create_performance_chart(top_assessors, "AssessorReal", "Pix_Assessor", 
                                                           "Top 10 Assessors by Pix", "bar")
                    if fig_assessors:
                        fig_assessors.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_assessors, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating assessors chart: {str(e)}")
            
            # Additional charts if commission data is available
            if "Comissão" in df_filtered.columns:
                chart_col3, chart_col4 = st.columns(2)
                
                with chart_col3:
                    try:
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
                    except Exception as e:
                        logger.error(f"Error creating commission chart: {str(e)}")
                
                with chart_col4:
                    try:
                        if "Lucro_Empresa" in df_filtered.columns:
                            margin_data = df_filtered.groupby("Distribuidor").agg({
                                "Comissão": "sum",
                                "Lucro_Empresa": "sum"
                            }).reset_index()
                            margin_data["Profit_Margin"] = (margin_data["Lucro_Empresa"] / margin_data["Comissão"].replace(0, np.nan) * 100).round(2)
                            
                            fig_margin = px.bar(margin_data, x="Distribuidor", y="Profit_Margin", 
                                              title="Profit Margin % by Distribuidor")
                            st.plotly_chart(fig_margin, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating margin chart: {str(e)}")
            
            # Export options

            st.markdown("### 📥 Export Options")
            
            # Create two columns to place buttons side-by-side
            ex_col1, ex_col2 = st.columns(2)

            # --- EXCEL BUTTON (Column 1) ---
            with ex_col1:
                try:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_filtered.to_excel(writer, sheet_name='Raw_Data', index=False)
                        pivot_df.round(2).to_excel(writer, sheet_name='Summary_Table', index=False)
                    
                    excel_buffer.seek(0)
                    
                    filename_excel = f"Macro_Analysis_{'_'.join(map(str, selected_chaves))}.xlsx"
                    st.download_button(
                        label="📊 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=filename_excel,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    logger.error(f"Error creating excel export: {str(e)}")
                    st.error(f"❌ Error creating Excel file: {str(e)}")

            # --- CSV BUTTON (Column 2) ---
            with ex_col2:
                try:
                    # Convert to CSV
                    # Defaults for to_csv are US format (sep=',' and decimal='.')
                    # .encode('utf-8') ensures the correct encoding
                    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                    
                    filename_csv = f"Macro_Analysis_{'_'.join(map(str, selected_chaves))}.csv"
                    
                    st.download_button(
                        label="📄 Download CSV Report",
                        data=csv_data,
                        file_name=filename_csv,
                        mime="text/csv"
                    )
                except Exception as e:
                    logger.error(f"Error creating csv export: {str(e)}")
                    st.error(f"❌ Error creating CSV file: {str(e)}")

        else:
            st.warning("Please select at least one Chave period.")
    
    except Exception as e:
        logger.error(f"Unexpected error in Macro View: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc(), language="python")

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown(f'<div class="main-header"><h1>👤 Assessor View - UTOR PJ2 </h1></div>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file in the Upload section.")
        st.stop()
    
    try:
        df_all = st.session_state["df_all"]
        
        if df_all is None or df_all.empty:
            st.error("❌ No data available. Please upload a file first.")
            st.stop()
        
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
        df_filtered = df_all[df_all["AssessorReal"] == selected_assessor].copy()
        if selected_months:
            df_filtered = df_filtered[df_filtered["Chave"].isin(selected_months)]
        
        if df_filtered.empty:
            st.warning("No data for selected AssessorReal and/or months.")
        else:
            try:
                # Key Highlights Section
                st.markdown(f"### 🎯  Relatório Gerencial - UTOR PJ2 - NOVEMBRO | {selected_assessor}")
                
                highlight_col1, highlight_col2, highlight_col3, highlight_col4 = st.columns(4)
                
                with highlight_col1:
                    if "Comissão" in df_filtered.columns:
                        total_commission = safe_operation(lambda: df_filtered["Comissão"].sum(), default=0)
                        st.metric("💰 Commission", format_currency(total_commission))
                    else:
                        st.metric("💰 Commission", "N/A")
                
                with highlight_col2:
                    if "Imposto" in df_filtered.columns:
                        total_tax = safe_operation(lambda: df_filtered["Imposto"].sum(), default=0)
                        st.metric("🏛️ Tax", format_currency(total_tax))
                    else:
                        st.metric("🏛️ Tax", "N/A")
                
                with highlight_col3:
                    total_pix = safe_operation(lambda: df_filtered["Pix_Assessor"].sum(), default=0)
                    st.metric("💳 Pix Assessor", format_currency(total_pix))
                
                
                # Main pivot table
                st.markdown("### 📋 Detailed Breakdown")
                try:
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
                except Exception as e:
                    logger.error(f"Error creating pivot table: {str(e)}")
                    st.error(f"❌ Error creating breakdown table: {str(e)}")
                
                # Distribuidor totals
                st.markdown("### 📊 Total by Distribuidor")
                try:
                    sheet_totals = (
                        df_filtered.groupby("Distribuidor")["Pix_Assessor"]
                        .sum()
                        .reset_index()
                        .sort_values("Pix_Assessor", ascending=False)
                    )
                    sheet_totals["Pix_Assessor"] = sheet_totals["Pix_Assessor"].round(2)
                    
                    grand_total = sheet_totals["Pix_Assessor"].sum()
                    total_row = pd.DataFrame({"Distribuidor": ["GRAND TOTAL"], "Pix_Assessor": [grand_total]})
                    sheet_totals_with_total = pd.concat([sheet_totals, total_row], ignore_index=True)
                    
                    st.dataframe(sheet_totals_with_total, use_container_width=True, hide_index=True)
                except Exception as e:
                    logger.error(f"Error creating totals: {str(e)}")
                    st.error(f"❌ Error calculating totals: {str(e)}")
                
                # Performance Charts
                st.markdown("### 📊 Performance Analytics")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    try:
                        if len(selected_months) > 1:
                            monthly_perf = df_filtered.groupby("Chave")["Pix_Assessor"].sum().reset_index()
                            monthly_perf = monthly_perf.sort_values("Chave")
                            
                            fig_monthly = create_performance_chart(monthly_perf, "Chave", "Pix_Assessor", 
                                                                 "Monthly Performance", "line")
                            if fig_monthly:
                                st.plotly_chart(fig_monthly, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating monthly chart: {str(e)}")
                
                with chart_col2:
                    try:
                        dist_perf = df_filtered.groupby("Distribuidor")["Pix_Assessor"].sum().reset_index()
                        dist_perf = dist_perf.sort_values("Pix_Assessor", ascending=False)
                        
                        fig_dist = create_performance_chart(dist_perf, "Distribuidor", "Pix_Assessor", 
                                                          "Performance by Distribuidor", "bar")
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating distribuidor chart: {str(e)}")
                
                # Export section
                st.markdown("### 📥 Export Options")
                
                # Check for columns and inform user
                missing_req_cols = []
                
                # --- UPDATED COLUMN LIST ---
                requested_cols = [
                    "AssessorReal", 
                    "Chave", 
                    "Conta", 
                    "Cliente", 
                    "Ativo", 
                    "Categoria",    # <--- ADDED
                    "Tipo Receita", # <--- ADDED
                    "Comissão",
                    "Pix_Assessor", 
                    "Distribuidor"
                ]
                
                # Because we standardized the data in Upload, this list should ideally be empty now
                for c in requested_cols:
                    if c not in df_filtered.columns:
                        missing_req_cols.append(c)
                
                if missing_req_cols:
                    st.warning(f"⚠️ Warning: The following columns were requested but are missing from data: {', '.join(missing_req_cols)}")
                
                export_info = f"Export will contain: {', '.join([c for c in requested_cols if c in df_filtered.columns])}"
                st.info(export_info)
                
                try:
                    excel_buffer = BytesIO()
                    
                    # --- CUSTOM COLUMN SELECTION LOGIC ---
                    # Now that we guarantee the columns exist (even if empty), we can safely request them
                    final_export_cols = [col for col in requested_cols if col in df_filtered.columns]
                    
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # Only export selected columns
                        if final_export_cols:
                            df_filtered[final_export_cols].to_excel(writer, sheet_name='Raw_Data', index=False)
                        else:
                            df_filtered.to_excel(writer, sheet_name='Raw_Data', index=False)
                        
                        pivot_df.round(2).to_excel(writer, sheet_name='Summary_Table', index=False)
                        sheet_totals_with_total.to_excel(writer, sheet_name='Distribuidor_Totals', index=False)
                    
                    excel_buffer.seek(0)
                    
                    filename_excel = f"Assessor_Report_{selected_assessor}.xlsx"
                    st.download_button(
                        label="📊 Download Complete Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=filename_excel,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    logger.error(f"Error creating export: {str(e)}")
                    st.error(f"❌ Error creating export file: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error in Assessor View: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                with st.expander("🔍 Detailed Error Information"):
                    st.code(traceback.format_exc(), language="python")
    
    except Exception as e:
        logger.error(f"Critical error in Assessor View: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ A critical error occurred: {str(e)}")
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc(), language="python")

# --- PROFIT PAGE ---
elif page == "💰 Profit Analysis":
    st.markdown('<div class="main-header"><h1>💰 Profit Analysis</h1></div>', unsafe_allow_html=True)
    
    if st.session_state["df_all"] is None or st.session_state["uploaded_file_data"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    try:
        uploaded_file = st.session_state["uploaded_file_data"]
        
        with st.spinner("Analyzing profit data..."):
            try:
                logger.info("Starting profit analysis")
                xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                all_sheets = xls.sheet_names
                lucro_data = []
                
                for sheet in all_sheets:
                    try:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        # --- FIX: Clean column names ---
                        df.columns = df.columns.astype(str).str.strip()
                        
                        if {"Chave", "Lucro_Empresa"}.issubset(df.columns):
                            temp = df[["Chave", "Lucro_Empresa"]].copy()
                            # Convert to numeric
                            temp["Lucro_Empresa"] = pd.to_numeric(temp["Lucro_Empresa"], errors='coerce')
                            temp["Distribuidor"] = sheet
                            lucro_data.append(temp)
                    except Exception as e:
                        logger.warning(f"Error processing sheet '{sheet}' for profit analysis: {str(e)}")
                        continue
                
                if not lucro_data:
                    st.error("❌ No sheets contained both 'Chave' and 'Lucro_Empresa' columns.")
                    st.stop()
                
                try:
                    df_lucro = pd.concat(lucro_data, ignore_index=True)
                    logger.info(f"Successfully loaded profit data: {len(df_lucro)} rows")
                except Exception as e:
                    st.error(f"❌ Error combining profit data: {str(e)}")
                    logger.error(f"Profit concatenation error: {str(e)}")
                    st.stop()
                
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
                
                try:
                    total_profit = safe_operation(lambda: df_lucro_filtered["Lucro_Empresa"].sum(), default=0)
                    avg_profit = safe_operation(lambda: df_lucro_filtered["Lucro_Empresa"].mean(), default=0)
                    max_profit = safe_operation(lambda: df_lucro_filtered["Lucro_Empresa"].max(), default=0)
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("💰 Total Profit", format_currency(total_profit))
                    with metric_col2:
                        st.metric("📊 Average Profit", format_currency(avg_profit))
                    with metric_col3:
                        st.metric("🏆 Maximum Profit", format_currency(max_profit))
                except Exception as e:
                    logger.error(f"Error calculating profit metrics: {str(e)}")
                    st.error(f"❌ Error calculating metrics: {str(e)}")
                
                # Charts
                st.markdown("### 📊 Profit Analytics")
                
                try:
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
                        if fig_line:
                            st.plotly_chart(fig_line, use_container_width=True)
                    
                    with chart_col2:
                        fig_bar = create_performance_chart(lucro_summary, "Chave", "Lucro_Empresa", 
                                                         "Profit by Chave", "bar")
                        if fig_bar:
                            st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating profit charts: {str(e)}")
                    st.error(f"❌ Error creating charts: {str(e)}")
                
                # Profit by Distribuidor
                try:
                    dist_profit = (
                        df_lucro_filtered.groupby("Distribuidor")["Lucro_Empresa"]
                        .sum()
                        .reset_index()
                        .sort_values("Lucro_Empresa", ascending=False)
                    )
                    
                    st.markdown("### 📊 Profit by Distribuidor")
                    fig_dist_profit = create_performance_chart(dist_profit, "Distribuidor", "Lucro_Empresa", 
                                                             "Profit by Distribuidor", "bar")
                    if fig_dist_profit:
                        st.plotly_chart(fig_dist_profit, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating distribuidor profit chart: {str(e)}")
                    st.error(f"❌ Error creating distribuidor chart: {str(e)}")
                
                # Summary table
                st.markdown("### 📋 Summary Table")
                try:
                    st.dataframe(lucro_summary.round(2), use_container_width=True)
                except Exception as e:
                    logger.error(f"Error displaying summary table: {str(e)}")
                    st.error(f"❌ Error displaying table: {str(e)}")
                
                # Export
                st.markdown("### 📥 Export Options")
                
                try:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_lucro_filtered.to_excel(writer, sheet_name='Raw_Profit_Data', index=False)
                        lucro_summary.round(2).to_excel(writer, sheet_name='Profit_Summary', index=False)
                    
                    excel_buffer.seek(0)
                    
                    filename_excel = f"Profit_Analysis_{'_'.join(map(str, selected_chaves)) if selected_chaves else 'All'}.xlsx"
                    st.download_button(
                        label="📊 Download Profit Analysis Excel",
                        data=excel_buffer.getvalue(),
                        file_name=filename_excel,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    logger.error(f"Error creating profit export: {str(e)}")
                    st.error(f"❌ Error creating export file: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error in profit analysis: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ Error processing profit data: {str(e)}")
                with st.expander("🔍 Detailed Error Information"):
                    st.code(traceback.format_exc(), language="python")
    
    except Exception as e:
        logger.error(f"Critical error in Profit Analysis: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ A critical error occurred: {str(e)}")
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc(), language="python")

# --- SIDEBAR INFO ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dashboard Info")
st.sidebar.markdown("**Version:** 2.7 (More Columns)")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- 📈 Advanced Analytics")
st.sidebar.markdown("- 🎯 Key Highlights")
st.sidebar.markdown("- 📊 Interactive Charts")
st.sidebar.markdown("- 💰 Profit Analysis")
st.sidebar.markdown("- 📥 Enhanced Exports")
st.sidebar.markdown("- 🛡️ Error Handling & Logging")

if st.session_state["df_all"] is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Current Data")
    try:
        st.sidebar.markdown(f"**Rows:** {len(st.session_state['df_all'])}")
        st.sidebar.markdown(f"**Assessors:** {st.session_state['df_all']['AssessorReal'].nunique()}")
        st.sidebar.markdown(f"**Periods:** {st.session_state['df_all']['Chave'].nunique()}")
    except Exception as e:
        logger.error(f"Error displaying sidebar info: {str(e)}")
        st.sidebar.warning("Error loading data info")



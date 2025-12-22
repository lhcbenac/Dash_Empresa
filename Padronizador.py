import streamlit as st
import pandas as pd
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
    page_title="Utor Data Converter", 
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Custom CSS (From your snippet)
st.markdown("""
<style>
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

# --- ERROR HANDLER CLASS (From your snippet) ---
class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

def safe_operation(func, *args, default=None, error_message="An error occurred", **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ {error_message}: {str(e)}")
        return default

def validate_dataframe(df: pd.DataFrame, required_cols: set, sheet_name: str = "Unknown") -> Tuple[bool, str]:
    """Validate DataFrame structure and content"""
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

# --- CUSTOM HELPER: PT-BR Number Fixer ---
def clean_pt_br_numbers(df, cols):
    """
    Explicitly fixes 1.000,00 format to 1000.00 for SQL compatibility.
    """
    for col in cols:
        if col in df.columns:
            try:
                # Force string, remove 'R$', remove whitespace
                df[col] = df[col].astype(str).str.replace('R$', '', regex=False).str.strip()
                # Remove thousand separator (.)
                df[col] = df[col].str.replace('.', '', regex=False)
                # Replace decimal separator (,) with (.)
                df[col] = df[col].str.replace(',', '.', regex=False)
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not clean column {col}: {e}")
    return df

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 📂 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["🦅 Utor Conversion"])

# --- CONSTANTS ---
TARGET_COLS_UTOR = [
    "Chave", "AssessorReal", "Pix_Assessor", "Cliente", "Conta",
    "Ativo", "Categoria", "Tipo_Receita", "VALOR_LIQUIDO_IR", "Comissao",
    "Imposto", "Lucro_Empresa", "Chave_Interna", "Data_Receita", "Distribuidor"
]

# --- MAIN PAGE ---
if page == "🦅 Utor Conversion":
    st.markdown('<div class="main-header"><h1>🦅 Utor Data Converter</h1></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload the 'Utor_Detalhado.xlsx' file", 
        type=["xlsx"],
        help="Please upload your Excel file containing assessor data"
    )

    if uploaded_file:
        with st.spinner("Processing your file..."):
            try:
                # Load Excel
                logger.info(f"Loading Excel file: {uploaded_file.name}")
                xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                all_sheets = xls.sheet_names
                
                data = []
                skipped_sheets = []
                
                # Progress Bar
                progress_bar = st.progress(0)
                
                for i, sheet in enumerate(all_sheets):
                    try:
                        logger.info(f"Processing sheet: {sheet}")
                        df = pd.read_excel(xls, sheet_name=sheet)
                        
                        # 1. Clean Headers
                        df.columns = df.columns.astype(str).str.strip()
                        
                        # 2. Check strict requirements
                        required_cols = {'Cliente', 'Pix_Assessor'} # Min requirements to detect valid sheet
                        is_valid, error_msg = validate_dataframe(df, required_cols, sheet)
                        
                        if not is_valid:
                            skipped_sheets.append(f"{sheet} - {error_msg}")
                            logger.warning(f"Skipped sheet '{sheet}': {error_msg}")
                            continue

                        # 3. Add Metadata
                        df['Distribuidor'] = sheet

                        # 4. Rename Known Variations
                        rename_map = {
                            "Valor Liquido": "VALOR_LIQUIDO_IR", 
                            "Tipo Receita": "Tipo_Receita",
                            "Comissão": "Comissao"
                        }
                        df.rename(columns=rename_map, inplace=True)

                        # 5. Ensure Target Columns Exist
                        for col in TARGET_COLS_UTOR:
                            if col not in df.columns:
                                df[col] = None

                        # 6. FIX NUMBERS (PT-BR -> US/SQL)
                        # This is the critical step for "Good Conversion"
                        numeric_cols = ["Pix_Assessor", "VALOR_LIQUIDO_IR", "Comissao", "Imposto", "Lucro_Empresa"]
                        df = clean_pt_br_numbers(df, numeric_cols)

                        # 7. Select & Append
                        df_final_slice = df[TARGET_COLS_UTOR].copy()
                        data.append(df_final_slice)
                        
                    except Exception as e:
                        error_detail = f"{sheet} (Error: {str(e)})"
                        skipped_sheets.append(error_detail)
                        logger.error(f"Error processing sheet '{sheet}': {str(e)}")
                    
                    finally:
                        progress_bar.progress((i + 1) / len(all_sheets))

                # Combine Data
                if data:
                    df_all = pd.concat(data, ignore_index=True)
                    
                    # Sort by Chave
                    if "Chave" in df_all.columns:
                        df_all = df_all.sort_values(by="Chave")

                    st.success(f"✅ Successfully processed {len(data)} sheets. Total Rows: {len(df_all)}")
                    
                    # Preview
                    with st.expander("👁️ Preview Data", expanded=True):
                        st.dataframe(df_all.head(10), use_container_width=True)

                    # Export
                    csv_data = df_all.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Download Standardized CSV (UTF-8)",
                        data=csv_data,
                        file_name="Utor_SQL_Converted.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ No valid sheets found (Requires 'Cliente' and 'Pix_Assessor').")
                
                # Show skipped
                if skipped_sheets:
                    with st.expander("⚠️ Skipped Sheets Log"):
                        for s in skipped_sheets:
                            st.write(f"- {s}")

            except Exception as e:
                st.error(f"❌ Critical Error: {str(e)}")
                st.code(traceback.format_exc())

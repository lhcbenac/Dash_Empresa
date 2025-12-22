import streamlit as st
import pandas as pd
from io import BytesIO

# --- Page Setup ---
st.set_page_config(page_title="Data Converter", layout="wide")

# --- Column Mappings ---
# LEFT: How they appear in your Excel File (Input)
# RIGHT: How they must appear in SQL/CSV (Output)
TAURUS_MAPPING = {
    "Data Receita": "Data_Receita",
    "Conta": "Conta",
    "Cliente": "Cliente",
    "Código Assessor": "Codigo_Assessor",
    "Assessor Principal": "Assessor_Principal",
    "Categoria": "Categoria",
    "Produto": "Produto",
    "Ativo": "Ativo",
    "Código/CNPJ": "Codigo_CNPJ",
    "Tipo Receita": "Tipo_Receita",
    "Receita Bruta": "Receita_Bruta",
    "Receita Líquida": "Receita_Liquida",
    "Comissão": "Comissao",
    "Chave": "Chave",
    "Chave_Interna": "Chave_Interna",
    "AssessorReal": "AssessorReal",
    "Exc_Cliente": "Exc_Cliente",
    "Exc_Mes": "Exc_Mes",
    "Standard": "Standard",
    "Repasse": "Repasse",
    "Imposto Retido": "Imposto_Retido",
    "Receita Assessor": "Receita_Assessor",
    "Tributo_Retido": "Tributo_Retido",
    "Pix_Assessor": "Pix_Assessor",
    "Lucro_Empresa": "Lucro_Empresa"
}

# The final list of columns to ensure strictly correct order
ORDER_TAURUS = list(TAURUS_MAPPING.values())

COLS_UTOR = [
    "Chave", "AssessorReal", "Pix_Assessor", "Cliente", "Conta",
    "Ativo", "Categoria", "Tipo_Receita", "VALOR_LIQUIDO_IR", "Comissao",
    "Imposto", "Lucro_Empresa", "Chave_Interna", "Data_Receita", "Distribuidor"
]

def clean_ptbr_numbers(df, columns_to_clean):
    """
    Converts PT-BR format (1.000,00) to SQL/US format (1000.00).
    """
    for col in columns_to_clean:
        if col in df.columns:
            # force to string first, then replace characters
            df[col] = df[col].astype(str)
            # Remove thousand separators (.) and replace decimal (,) with (.)
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            # Convert to float (coercing errors to NaN if something is really wrong)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def process_taurus(df):
    # 1. Strip whitespace from Excel headers (Fixes " Chave " issues)
    df.columns = df.columns.astype(str).str.strip()
    
    # 2. Rename columns using the Mapping
    # We use valid_mapping to avoid errors if your file changes slightly
    # It only renames columns that actually exist in the file
    valid_mapping = {k: v for k, v in TAURUS_MAPPING.items() if k in df.columns}
    df.rename(columns=valid_mapping, inplace=True)
    
    # 3. Add any missing target columns as empty
    for col in ORDER_TAURUS:
        if col not in df.columns:
            df[col] = None
            
    # 4. Clean Number Formats (Define which columns are money/numeric)
    numeric_cols = [
        "Receita_Bruta", "Receita_Liquida", "Comissao", 
        "Repasse", "Imposto_Retido", "Receita_Assessor", 
        "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"
    ]
    df = clean_ptbr_numbers(df, numeric_cols)
    
    # 5. Return ordered dataframe
    return df[ORDER_TAURUS]

def convert_df_to_csv(df):
    # utf-8-sig ensures Excel opens it correctly with special chars
    return df.to_csv(index=False).encode('utf-8-sig')

# --- Sidebar ---
st.sidebar.title("Select Tool")
page = st.sidebar.radio("Go to", ["Taurus Converter", "Utor Converter"])

# ==========================================
# PAGE 1: TAURUS
# ==========================================
if page == "Taurus Converter":
    st.header("🐂 Taurus: Excel to SQL CSV")
    
    uploaded_file = st.file_uploader("Upload Taurus Excel", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            # Read Data
            df = pd.read_excel(uploaded_file)
            
            # Process Data
            df_clean = process_taurus(df)
            
            # Sort by Chave
            if "Chave" in df_clean.columns:
                df_clean = df_clean.sort_values(by="Chave")
            
            st.success(f"Processed {len(df_clean)} rows.")
            st.dataframe(df_clean.head())
            
            # Download
            st.download_button(
                label="Download Taurus CSV (US Format)",
                data=convert_df_to_csv(df_clean),
                file_name="Taurus_SQL_Ready.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# PAGE 2: UTOR
# ==========================================
elif page == "Utor Converter":
    st.header("🦅 Utor: Multi-Sheet Processor")
    
    uploaded_file = st.file_uploader("Upload Utor Excel", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            all_data = []
            
            for sheet_name, df_sheet in xls.items():
                df_sheet.columns = df_sheet.columns.astype(str).str.strip()
                
                # Check for criteria
                if 'Cliente' in df_sheet.columns and 'Pix_Assessor' in df_sheet.columns:
                    
                    df_sheet['Distribuidor'] = sheet_name
                    
                    # Rename Helper for Utor
                    rename_map = {
                        "Valor Liquido": "VALOR_LIQUIDO_IR", 
                        "Tipo Receita": "Tipo_Receita",
                        "Comissão": "Comissao" # Common accent issue
                    }
                    df_sheet.rename(columns=rename_map, inplace=True)

                    # Ensure columns exist
                    for col in COLS_UTOR:
                        if col not in df_sheet.columns:
                            df_sheet[col] = None
                            
                    # Clean Numbers for Utor columns
                    numeric_cols_utor = [
                        "Pix_Assessor", "VALOR_LIQUIDO_IR", 
                        "Comissao", "Imposto", "Lucro_Empresa"
                    ]
                    df_sheet = clean_ptbr_numbers(df_sheet, numeric_cols_utor)

                    # Append
                    all_data.append(df_sheet[COLS_UTOR])
            
            if all_data:
                final_df = pd.concat(all_data, ignore_index=True)
                
                if "Chave" in final_df.columns:
                    final_df = final_df.sort_values(by="Chave")
                
                st.success(f"Merged {len(all_data)} sheets.")
                st.dataframe(final_df.head())
                
                st.download_button(
                    label="Download Utor CSV (US Format)",
                    data=convert_df_to_csv(final_df),
                    file_name="Utor_SQL_Ready.csv",
                    mime="text/csv",
                )
            else:
                st.error("No valid sheets found.")
        except Exception as e:
            st.error(f"Error: {e}")

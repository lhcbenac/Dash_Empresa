import streamlit as st
import pandas as pd
from io import BytesIO

# --- Page Setup ---
st.set_page_config(page_title="Data Converter", layout="wide")

# --- Target Column Definitions ---
COLS_TAURUS = [
    "Data_Receita", "Conta", "Cliente", "Codigo_Assessor", "Assessor_Principal",
    "Categoria", "Produto", "Ativo", "Codigo_CNPJ", "Tipo_Receita",
    "Receita_Bruta", "Receita_Liquida", "Comissao", "Chave", "Chave_Interna",
    "AssessorReal", "Exc_Cliente", "Exc_Mes", "Standard", "Repasse",
    "Imposto_Retido", "Receita_Assessor", "Tributo_Retido", "Pix_Assessor",
    "Lucro_Empresa"
]

COLS_UTOR = [
    "Chave", "AssessorReal", "Pix_Assessor", "Cliente", "Conta",
    "Ativo", "Categoria", "Tipo_Receita", "VALOR_LIQUIDO_IR", "Comissao",
    "Imposto", "Lucro_Empresa", "Chave_Interna", "Data_Receita", "Distribuidor"
]

def format_dataframe(df, target_columns):
    """
    1. Renames common variations (optional helper).
    2. Creates missing columns as empty.
    3. Returns only the requested columns in the exact order.
    """
    # Simple cleanup to remove whitespace from headers
    df.columns = df.columns.astype(str).str.strip()
    
    # Optional: Map common variations if they exist in your raw file
    # This helps if your raw file has "Valor Liquido" but you need "VALOR_LIQUIDO_IR"
    rename_map = {
        "Valor Liquido": "VALOR_LIQUIDO_IR", 
        "Tipo Receita": "Tipo_Receita"
    }
    df.rename(columns=rename_map, inplace=True)

    # Add missing columns
    for col in target_columns:
        if col not in df.columns:
            df[col] = None  # Leave blank/empty
            
    # Select exact columns in order
    return df[target_columns]

def convert_df_to_csv(df):
    """Converts dataframe to CSV UTF-8 bytes."""
    return df.to_csv(index=False).encode('utf-8')

# --- Sidebar ---
st.sidebar.title("Select Tool")
page = st.sidebar.radio("Go to", ["Taurus Converter", "Utor Converter"])

# ==========================================
# PAGE 1: TAURUS
# ==========================================
if page == "Taurus Converter":
    st.header("🐂 Taurus: Excel to Standard CSV")
    
    uploaded_file = st.file_uploader("Upload Taurus Excel", type=["xlsx", "xls"])
    
    if uploaded_file:
        # Read Data
        df = pd.read_excel(uploaded_file)
        
        # Format Data
        df_clean = format_dataframe(df, COLS_TAURUS)
        
        # Show Preview
        st.write(f"Processed {len(df_clean)} rows.")
        st.dataframe(df_clean.head())
        
        # Download Button
        csv = convert_df_to_csv(df_clean)
        st.download_button(
            label="Download Taurus CSV",
            data=csv,
            file_name="Taurus_Standardized.csv",
            mime="text/csv",
        )

# ==========================================
# PAGE 2: UTOR
# ==========================================
elif page == "Utor Converter":
    st.header("🦅 Utor: Multi-Sheet Processor")
    st.write("Loops through all sheets. Merges only those with 'Cliente' and 'Pix_Assessor'.")
    
    uploaded_file = st.file_uploader("Upload Utor Excel", type=["xlsx", "xls"])
    
    if uploaded_file:
        # Read all sheets at once
        xls = pd.read_excel(uploaded_file, sheet_name=None)
        
        all_data = []
        processed_sheets = []
        
        for sheet_name, df_sheet in xls.items():
            # Clean header whitespace
            df_sheet.columns = df_sheet.columns.astype(str).str.strip()
            
            # CHECK CRITERIA: Must have Cliente and Pix_Assessor
            if 'Cliente' in df_sheet.columns and 'Pix_Assessor' in df_sheet.columns:
                
                # 1. Add Distribuidor Column
                df_sheet['Distribuidor'] = sheet_name
                
                # 2. Format columns (add missing ones like Categoria, etc)
                df_sheet_clean = format_dataframe(df_sheet, COLS_UTOR)
                
                all_data.append(df_sheet_clean)
                processed_sheets.append(sheet_name)
        
        if all_data:
            # Merge all valid sheets
            final_df = pd.concat(all_data, ignore_index=True)
            
            st.success(f"Merged {len(processed_sheets)} sheets: {', '.join(processed_sheets)}")
            st.dataframe(final_df.head())
            
            # Download Button
            csv = convert_df_to_csv(final_df)
            st.download_button(
                label="Download Utor CSV",
                data=csv,
                file_name="Utor_Consolidated.csv",
                mime="text/csv",
            )
        else:
            st.error("No sheets found containing both 'Cliente' and 'Pix_Assessor' columns.")

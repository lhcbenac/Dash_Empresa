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
    1. Renames common variations.
    2. Creates missing columns as empty.
    3. Returns only the requested columns in the exact order.
    """
    # Clean header whitespace
    df.columns = df.columns.astype(str).str.strip()
    
    # Map common variations
    rename_map = {
        "Valor Liquido": "VALOR_LIQUIDO_IR", 
        "Tipo Receita": "Tipo_Receita"
    }
    df.rename(columns=rename_map, inplace=True)

    # Add missing columns
    for col in target_columns:
        if col not in df.columns:
            df[col] = None
            
    # Return exact columns
    return df[target_columns]

def convert_df_to_csv(df):
    """
    Converts dataframe to CSV.
    Uses 'utf-8-sig' which is the best UTF-8 version for Excel 
    (it includes the BOM so special characters like 'ç' and 'ã' show correctly).
    """
    return df.to_csv(index=False).encode('utf-8-sig')

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
        try:
            # Read Data
            df = pd.read_excel(uploaded_file)
            
            # Format Data
            df_clean = format_dataframe(df, COLS_TAURUS)
            
            # Sort by Chave
            if "Chave" in df_clean.columns:
                df_clean = df_clean.sort_values(by="Chave")
            
            # Show Preview
            st.success(f"Processed {len(df_clean)} rows.")
            st.dataframe(df_clean.head())
            
            # Download Button
            csv = convert_df_to_csv(df_clean)
            st.download_button(
                label="Download Taurus CSV",
                data=csv,
                file_name="Taurus_Output.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# PAGE 2: UTOR
# ==========================================
elif page == "Utor Converter":
    st.header("🦅 Utor: Multi-Sheet Processor")
    st.write("Loops through all sheets. Merges only those with 'Cliente' and 'Pix_Assessor'.")
    
    uploaded_file = st.file_uploader("Upload Utor Excel", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            # Read all sheets
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            
            all_data = []
            processed_sheets = []
            
            for sheet_name, df_sheet in xls.items():
                # Clean headers
                df_sheet.columns = df_sheet.columns.astype(str).str.strip()
                
                # CHECK CRITERIA
                if 'Cliente' in df_sheet.columns and 'Pix_Assessor' in df_sheet.columns:
                    
                    # 1. Add Distribuidor
                    df_sheet['Distribuidor'] = sheet_name
                    
                    # 2. Format columns
                    df_sheet_clean = format_dataframe(df_sheet, COLS_UTOR)
                    
                    all_data.append(df_sheet_clean)
                    processed_sheets.append(sheet_name)
            
            if all_data:
                # Merge
                final_df = pd.concat(all_data, ignore_index=True)
                
                # Sort by Chave
                if "Chave" in final_df.columns:
                    final_df = final_df.sort_values(by="Chave")
                
                st.success(f"Merged {len(processed_sheets)} sheets: {', '.join(processed_sheets)}")
                st.dataframe(final_df.head())
                
                # Download Button
                csv = convert_df_to_csv(final_df)
                st.download_button(
                    label="Download Utor CSV",
                    data=csv,
                    file_name="Utor_Output.csv",
                    mime="text/csv",
                )
            else:
                st.error("No sheets found containing both 'Cliente' and 'Pix_Assessor' columns.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

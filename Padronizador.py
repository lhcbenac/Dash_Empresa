import streamlit as st
import pandas as pd
from io import BytesIO

# --- Configuration ---
st.set_page_config(page_title="Excel Processor", layout="wide")

# Target Columns Definitions
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

def to_csv_utf8(df):
    """Converts a dataframe to a CSV string encoded in UTF-8."""
    return df.to_csv(index=False).encode('utf-8')

def enforce_schema(df, target_columns):
    """
    Ensures the dataframe has exactly the target columns.
    Missing columns are created as empty strings.
    Extra columns are dropped.
    """
    # Add missing columns
    for col in target_columns:
        if col not in df.columns:
            df[col] = ""  # or 0, or None depending on preference
    
    # Return only the target columns in the specific order
    return df[target_columns]

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Taurus", "Utor"])

# --- Page: Taurus ---
if page == "Taurus":
    st.title("🐂 Taurus File Processor")
    st.write("Upload the Taurus Excel file. The output will follow the standard Taurus schema.")

    uploaded_file = st.file_uploader("Upload Taurus Excel", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            # Read Data
            df = pd.read_excel(uploaded_file)
            
            # Process Data
            df_processed = enforce_schema(df, COLS_TAURUS)
            
            st.success("File processed successfully!")
            st.dataframe(df_processed.head())

            # Download Button
            csv_data = to_csv_utf8(df_processed)
            st.download_button(
                label="Download Taurus CSV",
                data=csv_data,
                file_name="Taurus_Output.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Page: Utor ---
elif page == "Utor":
    st.title("🦅 Utor File Processor")
    st.write("Upload the Utor Excel file (Multiple Sheets). Only sheets containing `Cliente` and `Pix_Assessor` will be processed.")

    uploaded_file = st.file_uploader("Upload Utor Excel", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            # Read all sheets
            # sheet_name=None returns a dictionary {sheet_name: dataframe}
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            
            all_data = []
            processed_sheets = []

            for sheet_name, df_sheet in xls.items():
                # Check criteria: Must have Cliente and Pix_Assessor
                if 'Cliente' in df_sheet.columns and 'Pix_Assessor' in df_sheet.columns:
                    
                    # Add Distributor Name (Sheet Name)
                    df_sheet['Distribuidor'] = sheet_name
                    
                    # Ensure specific columns exist before appending
                    df_sheet_clean = enforce_schema(df_sheet, COLS_UTOR)
                    
                    all_data.append(df_sheet_clean)
                    processed_sheets.append(sheet_name)

            if all_data:
                # Combine all valid sheets
                final_df = pd.concat(all_data, ignore_index=True)
                
                st.success(f"Processed {len(processed_sheets)} sheets: {', '.join(processed_sheets)}")
                st.dataframe(final_df.head())

                # Download Button
                csv_data = to_csv_utf8(final_df)
                st.download_button(
                    label="Download Utor CSV",
                    data=csv_data,
                    file_name="Utor_Output.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No sheets found with columns 'Cliente' and 'Pix_Assessor'.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

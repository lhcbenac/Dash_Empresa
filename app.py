import streamlit as st
import pandas as pd

# --- Streamlit config ---
st.set_page_config(page_title="Pix Assessor Dashboard", layout="wide")
st.title("📊 Pix Assessor Dashboard")

# --- File upload ---
uploaded_file = st.file_uploader("Upload the 'Utor_Detalhado.xlsx' file", type=["xlsx"])

if uploaded_file:
    try:
        # Load Excel file with openpyxl engine
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        all_sheets = xls.sheet_names

        # Consolidate data
        expected_cols = {"Chave", "AssessorReal", "Pix_Assessor"}
        data = []
        skipped_sheets = []

        for sheet in all_sheets:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                if not expected_cols.issubset(df.columns):
                    skipped_sheets.append(sheet)
                    continue
                df = df[["Chave", "AssessorReal", "Pix_Assessor"]]
                df["Distribuidor"] = sheet
                data.append(df)
            except Exception as e:
                skipped_sheets.append(sheet)

        # Check if any valid data was found
        if not data:
            st.error("No valid sheets found. Please check that all distributor sheets contain 'Chave', 'AssessorReal', and 'Pix_Assessor'.")
        else:
            df_all = pd.concat(data, ignore_index=True)

            # Chave filter
            chave_list = sorted(df_all["Chave"].dropna().unique())
            selected_chave = st.selectbox("Select a Chave", chave_list)

            df_filtered = df_all[df_all["Chave"] == selected_chave]

            # Pivot table
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

            st.markdown(f"### Summary for Chave: `{selected_chave}`")
            st.dataframe(pivot_df, use_container_width=True)

            # Download button
            csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"Pix_Summary_{selected_chave}.csv",
                mime="text/csv"
            )

            # Show skipped sheets (if any)
            if skipped_sheets:
                with st.expander("⚠️ Skipped Sheets (Missing Columns)"):
                    for sheet in skipped_sheets:
                        st.write(f"- {sheet}")

    except Exception as e:
        st.error(f"❌ An error occurred while processing the Excel file. Details: {e}")

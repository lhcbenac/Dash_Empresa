import streamlit as st
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="Pix Assessor Dashboard", layout="wide")

# --- UPLOAD EXCEL FILE ---
st.title("📊 Pix Assessor Dashboard")
uploaded_file = st.file_uploader("Upload the Utor_Detalhado Excel file", type=["xlsx"])

if uploaded_file:
    # --- LOAD SHEETS ---
    xls = pd.ExcelFile(uploaded_file)
    all_sheets = xls.sheet_names  # Distributor sheet names

    # --- CONSOLIDATE DATA ---
    data = []
    for sheet in all_sheets:
        df = pd.read_excel(xls, sheet_name=sheet, usecols=["Chave", "AssessorReal", "Pix_Assessor"])
        df["Distribuidor"] = sheet
        data.append(df)

    df_all = pd.concat(data, ignore_index=True)

    # --- FILTER FOR CHAVE SELECTION ---
    chave_list = sorted(df_all["Chave"].dropna().unique())
    selected_chave = st.selectbox("Select a Chave", chave_list)

    df_filtered = df_all[df_all["Chave"] == selected_chave]

    # --- PIVOT TABLE ---
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

    # --- DISPLAY ---
    st.markdown(f"### Summary for Chave: `{selected_chave}`")
    st.dataframe(pivot_df, use_container_width=True)

    # --- OPTIONAL: DOWNLOADABLE OUTPUT ---
    csv = pivot_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Summary CSV", data=csv, file_name=f"Pix_Summary_{selected_chave}.csv", mime='text/csv')

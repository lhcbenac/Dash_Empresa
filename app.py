import streamlit as st
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="Pix Assessor Dashboard", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Macro View", "Assessor View"])

# --- SESSION STORAGE ---
if "df_all" not in st.session_state:
    st.session_state["df_all"] = None
if "skipped_sheets" not in st.session_state:
    st.session_state["skipped_sheets"] = []

# --- UPLOAD PAGE ---
if page == "Upload":
    st.title("📤 Upload Excel File")

    uploaded_file = st.file_uploader("Upload the 'Utor_Detalhado.xlsx' file", type=["xlsx"])

    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            all_sheets = xls.sheet_names

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
                except:
                    skipped_sheets.append(sheet)

            if not data:
                st.error("❌ No valid sheets found. Please check columns.")
            else:
                df_all = pd.concat(data, ignore_index=True)
                st.session_state["df_all"] = df_all
                st.session_state["skipped_sheets"] = skipped_sheets
                st.success("✅ Data successfully loaded!")

                if skipped_sheets:
                    with st.expander("⚠️ Skipped Sheets"):
                        for s in skipped_sheets:
                            st.write(f"- {s}")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# --- MACRO VIEW PAGE ---
elif page == "Macro View":
    st.title("📊 Macro Summary View")

    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file in the Upload section.")
        st.stop()

    df_all = st.session_state["df_all"]
    chave_list = sorted(df_all["Chave"].dropna().unique())
    selected_chaves = st.multiselect("Select one or more Chave periods", chave_list, default=chave_list[:1])

    if selected_chaves:
        df_filtered = df_all[df_all["Chave"].isin(selected_chaves)]
        st.markdown(f"### Summary for Chave(s): `{', '.join(map(str, selected_chaves))}`")

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

        csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "Pix_Summary_Selected_Chaves.csv", "text/csv")
    else:
        st.warning("Please select at least one Chave.")

# --- ASSESSOR VIEW PAGE ---
elif page == "Assessor View":
    st.title("👤 Assessor Breakdown View")

    if st.session_state["df_all"] is None:
        st.warning("Please upload the Excel file in the Upload section.")
        st.stop()

    df_all = st.session_state["df_all"]
    assessor_list = sorted(df_all["AssessorReal"].dropna().unique())
    selected_assessor = st.selectbox("Select AssessorReal", assessor_list)

    df_filtered = df_all[df_all["AssessorReal"] == selected_assessor]

    if df_filtered.empty:
        st.warning("No data for selected AssessorReal.")
    else:
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

        st.markdown(f"### Summary for AssessorReal: `{selected_assessor}`")
        st.dataframe(pivot_df.round(2), use_container_width=True)

        csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, f"{selected_assessor}_Summary.csv", "text/csv")

import streamlit as st
import pandas as pd
# --- CONFIG ---
st.set_page_config(page_title="Pix Assessor Dashboard", layout="wide")
# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Macro View", "Assessor View", "Profit"])
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
        # Export summarized pivot table CSV
        csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "Pix_Summary_Selected_Chaves.csv", "text/csv")
        # Export full filtered raw data CSV
        csv_all = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📦 Download Full Data (Raw Rows)",
            data=csv_all,
            file_name="FullData_Selected_Chaves.csv",
            mime="text/csv"
        )
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
        # Export summarized pivot table CSV
        csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, f"{selected_assessor}_Summary.csv", "text/csv")
        # Export full filtered raw data CSV
        csv_all = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📦 Download Full Data (Raw Rows)",
            data=csv_all,
            file_name=f"{selected_assessor}_FullData.csv",
            mime="text/csv"
        )
# --- PROFIT PAGE ---
elif page == "Profit":
    st.title("💰 Profit Summary (Lucro_Utor)")
    uploaded_file = st.file_uploader("Optional: Re-upload to include Lucro_Utor from all sheets", type=["xlsx"], key="profit_upload")
    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            all_sheets = xls.sheet_names
            lucro_data = []
            for sheet in all_sheets:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    if {"Chave", "Lucro_Utor"}.issubset(df.columns):
                        temp = df[["Chave", "Lucro_Utor"]].copy()
                        temp["Distribuidor"] = sheet
                        lucro_data.append(temp)
                except:
                    continue
            if not lucro_data:
                st.error("❌ No sheets contained both 'Chave' and 'Lucro_Utor' columns.")
            else:
                df_lucro = pd.concat(lucro_data, ignore_index=True)
                
                # --- NEW: CHAVE FILTER ---
                st.markdown("### 🔍 Filter Options")
                chave_list = sorted(df_lucro["Chave"].dropna().unique())
                selected_chaves = st.multiselect(
                    "Select Chave periods to include (leave empty for all)",
                    chave_list,
                    default=chave_list  # Default to all selected
                )
                
                # Filter data based on selection
                if selected_chaves:
                    df_lucro_filtered = df_lucro[df_lucro["Chave"].isin(selected_chaves)]
                else:
                    df_lucro_filtered = df_lucro
                
                lucro_summary = (
                    df_lucro_filtered.groupby("Chave")["Lucro_Utor"]
                    .sum()
                    .reset_index()
                    .sort_values("Chave")
                )
                
                # --- NEW: CALCULATE TOTAL SUM ---
                total_sum = lucro_summary["Lucro_Utor"].sum()
                
                st.markdown("### 📈 Total Lucro_Utor by Chave")
                
                # --- NEW: DISPLAY TOTAL SUM LABEL ---
                st.metric(
                    label="💰 Total Sum of All Values",
                    value=f"{total_sum:,.2f}",
                    help="Sum of all Lucro_Utor values in the chart below"
                )
                
                # Display chart
                st.bar_chart(lucro_summary.set_index("Chave"))
                
                # Show summary table
                st.markdown("### 📊 Summary Table")
                st.dataframe(lucro_summary.round(2), use_container_width=True)
                
                # Download button
                csv = lucro_summary.to_csv(index=False).encode("utf-8")
                filename = f"Lucro_Utor_by_Chave_{'_'.join(map(str, selected_chaves)) if selected_chaves else 'All'}.csv"
                st.download_button("📥 Download Profit CSV", csv, filename, "text/csv")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("Upload the same Excel file or another one to analyze 'Lucro_Utor'.")

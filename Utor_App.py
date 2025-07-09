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
if "uploaded_file_data" not in st.session_state:
    st.session_state["uploaded_file_data"] = None

# --- UPLOAD PAGE ---
if page == "Upload":
    st.title("📤 Upload Excel File")
    uploaded_file = st.file_uploader("Upload the 'Utor_Detalhado.xlsx' file", type=["xlsx"])
    
    if uploaded_file:
        # Store the uploaded file data in session state
        st.session_state["uploaded_file_data"] = uploaded_file
        
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
    
    # Filters section
    col1, col2 = st.columns(2)
    
    with col1:
        chave_list = sorted(df_all["Chave"].dropna().unique())
        selected_chaves = st.multiselect(
            "Select one or more Chave periods", 
            chave_list, 
            default=chave_list[:1]
        )
    
    with col2:
        assessor_list = sorted(df_all["AssessorReal"].dropna().unique())
        selected_assessors = st.multiselect(
            "Select one or more Assessors (leave empty for all)", 
            assessor_list,
            default=[]
        )
    
    if selected_chaves:
        # Filter by Chave
        df_filtered = df_all[df_all["Chave"].isin(selected_chaves)]
        
        # Filter by Assessor if selected
        if selected_assessors:
            df_filtered = df_filtered[df_filtered["AssessorReal"].isin(selected_assessors)]
        
        # Display filter info
        filter_info = f"Chave(s): {', '.join(map(str, selected_chaves))}"
        if selected_assessors:
            filter_info += f" | Assessor(s): {', '.join(selected_assessors)}"
        else:
            filter_info += " | All Assessors"
        
        st.markdown(f"### Summary for {filter_info}")
        
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
        filename = f"Pix_Summary_{'_'.join(map(str, selected_chaves))}"
        if selected_assessors:
            filename += f"_Assessors_{'_'.join(selected_assessors[:3])}"  # Limit filename length
        filename += ".csv"
        st.download_button("📥 Download CSV", csv, filename, "text/csv")
        
        # Export full filtered raw data CSV
        csv_all = df_filtered.to_csv(index=False).encode("utf-8")
        filename_raw = f"FullData_{'_'.join(map(str, selected_chaves))}"
        if selected_assessors:
            filename_raw += f"_Assessors_{'_'.join(selected_assessors[:3])}"
        filename_raw += ".csv"
        st.download_button(
            label="📦 Download Full Data (Raw Rows)",
            data=csv_all,
            file_name=filename_raw,
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
    
    # Filters section
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
    df_filtered = df_all[df_all["AssessorReal"] == selected_assessor]
    
    if selected_months:
        df_filtered = df_filtered[df_filtered["Chave"].isin(selected_months)]
    
    if df_filtered.empty:
        st.warning("No data for selected AssessorReal and/or months.")
    else:
        # Main pivot table (Distribuidor vs Chave)
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
        
        month_info = f"Months: {', '.join(map(str, selected_months))}" if selected_months else "All Months"
        st.markdown(f"### Summary for AssessorReal: {selected_assessor} | {month_info}")
        st.dataframe(pivot_df.round(2), use_container_width=True)
        
        # NEW: Table with totals for all sheets (Distributors)
        st.markdown("### 📊 Total by Distribuidor (All Sheets)")
        sheet_totals = (
            df_filtered.groupby("Distribuidor")["Pix_Assessor"]
            .sum()
            .reset_index()
            .sort_values("Pix_Assessor", ascending=False)
        )
        sheet_totals["Pix_Assessor"] = sheet_totals["Pix_Assessor"].round(2)
        
        # Add grand total row
        grand_total = sheet_totals["Pix_Assessor"].sum()
        total_row = pd.DataFrame({"Distribuidor": ["GRAND TOTAL"], "Pix_Assessor": [grand_total]})
        sheet_totals_with_total = pd.concat([sheet_totals, total_row], ignore_index=True)
        
        st.dataframe(sheet_totals_with_total, use_container_width=True, hide_index=True)
        
        # Export buttons
        csv = pivot_df.round(2).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Main Summary CSV", csv, f"{selected_assessor}_Summary.csv", "text/csv")
        
        csv_totals = sheet_totals_with_total.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Sheet Totals CSV", csv_totals, f"{selected_assessor}_SheetTotals.csv", "text/csv")
                
        # ✅ Assume you have a filtered DataFrame called df_filtered
        # Make sure 'Cliente' is included for export
        export_columns = list(df_filtered.columns)
        
        # If 'Cliente' is missing but should be there, warn the user
        if "Cliente" not in export_columns:
            st.warning("⚠️ 'Cliente' column not found — adding it if possible.")
            if "Cliente" in st.session_state["df_all"].columns:
                # Join with original to bring back Cliente
                df_filtered = df_filtered.merge(
                    st.session_state["df_all"][["Chave", "AssessorReal", "Cliente"]],
                    on=["Chave", "AssessorReal"],
                    how="left"
                )
            else:
                st.error("❌ 'Cliente' column not found in the source data either!")
                
        # Confirm final export columns include Cliente
        if "Cliente" not in df_filtered.columns:
            st.warning("⚠️ 'Cliente' still missing — export will continue without it.")
        else:
            st.success("✅ 'Cliente' column included in export!")
        
        # Export to CSV
        csv_all = df_filtered.to_csv(index=False).encode("utf-8")
        
        # Create dynamic filename
        filename_raw = f"FullData_{'_'.join(map(str, selected_chaves))}"
        if selected_assessors:
            filename_raw += f"_Assessors_{'_'.join(selected_assessors[:3])}"
        filename_raw += ".csv"
        
        # Download button
        st.download_button(
            label="📦 Download Full Data (Raw Rows)",
            data=csv_all,
            file_name=filename_raw,
            mime="text/csv"
        )


# --- PROFIT PAGE ---
elif page == "Profit":
    st.title("💰 Profit Summary (Lucro_Empresa)")
    
    if st.session_state["df_all"] is None or st.session_state["uploaded_file_data"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    st.success("✅ Using previously uploaded file for Lucro_Empresa analysis")
    uploaded_file = st.session_state["uploaded_file_data"]
    
    try:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        all_sheets = xls.sheet_names
        lucro_data = []
        
        for sheet in all_sheets:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                if {"Chave", "Lucro_Empresa"}.issubset(df.columns):
                    temp = df[["Chave", "Lucro_Empresa"]].copy()
                    temp["Distribuidor"] = sheet
                    lucro_data.append(temp)
            except:
                continue
        
        if not lucro_data:
            st.error("❌ No sheets contained both 'Chave' and 'Lucro_Empresa' columns.")
        else:
            df_lucro = pd.concat(lucro_data, ignore_index=True)
            
            st.markdown("### 🔍 Filter Options")
            chave_list = sorted(df_lucro["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "Select Chave periods to include (leave empty for all)",
                chave_list,
                default=chave_list
            )
            
            df_lucro_filtered = df_lucro[df_lucro["Chave"].isin(selected_chaves)] if selected_chaves else df_lucro
            
            lucro_summary = (
                df_lucro_filtered.groupby("Chave")["Lucro_Empresa"]
                .sum()
                .reset_index()
                .sort_values("Chave")
            )
            
            total_sum = lucro_summary["Lucro_Empresa"].sum()
            
            st.markdown("### 📈 Total Lucro_Empresa by Chave")
            st.metric(
                label="💰 Total Sum of All Values",
                value=f"{total_sum:,.2f}",
                help="Sum of all Lucro_Empresa values in the chart below"
            )
            
            st.bar_chart(lucro_summary.set_index("Chave"))
            
            st.markdown("### 📊 Summary Table")
            st.dataframe(lucro_summary.round(2), use_container_width=True)
            
            csv = lucro_summary.to_csv(index=False).encode("utf-8")
            filename = f"Lucro_Empresa_byChave{'_'.join(map(str, selected_chaves)) if selected_chaves else 'All'}.csv"
            st.download_button("📥 Download Profit CSV", csv, filename, "text/csv")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

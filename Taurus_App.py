import streamlit as st
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="Taurus Dashboard", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Macro View", "Assessor View", "Profit"])

# --- SESSION STORAGE ---
if "df_taurus" not in st.session_state:
    st.session_state["df_taurus"] = None

# --- UPLOAD PAGE ---
if page == "Upload":
    st.title("📤 Upload Taurus Excel File")
    uploaded_file = st.file_uploader("Upload the Excel file with 'Taurus' sheet", type=["xlsx"])
    
    if uploaded_file:
        try:
            # Read specifically the 'Taurus' sheet
            df_taurus = pd.read_excel(uploaded_file, sheet_name="Taurus", engine="openpyxl")
            
            # Check if required columns exist
            required_cols = {"Chave", "AssessorReal", "Categoria", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"}
            
            if not required_cols.issubset(df_taurus.columns):
                missing_cols = required_cols - set(df_taurus.columns)
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: Chave, AssessorReal, Categoria, Comissão, Tributo_Retido, Pix_Assessor, Lucro_Empresa")
            else:
                # Store data in session state
                st.session_state["df_taurus"] = df_taurus
                st.success("✅ Taurus data successfully loaded!")
                
                # Show basic info
                st.markdown("### 📊 Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df_taurus))
                with col2:
                    st.metric("Unique Assessors", df_taurus["AssessorReal"].nunique())
                with col3:
                    st.metric("Unique Chaves", df_taurus["Chave"].nunique())
                
                # Show sample data
                st.markdown("### 👀 Sample Data")
                st.dataframe(df_taurus.head(), use_container_width=True )
                
        except ValueError as e:
            if "Worksheet named 'Taurus' not found" in str(e):
                st.error("❌ Sheet named 'Taurus' not found in the uploaded file.")
                st.info("Please make sure your Excel file contains a sheet named 'Taurus'.")
            else:
                st.error(f"❌ Error reading file: {e}")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- MACRO VIEW PAGE ---
elif page == "Macro View":
    st.title("📊 Macro View - Summary by Assessor")
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    df_taurus = st.session_state["df_taurus"]
    
    # Chave filter
    st.markdown("### 🔍 Filter Options")
    chave_list = sorted(df_taurus["Chave"].dropna().unique())
    selected_chaves = st.multiselect(
        "Select Chave periods",
        chave_list,
        default=chave_list  # Default to all selected
    )
    
    if selected_chaves:
        df_filtered = df_taurus[df_taurus["Chave"].isin(selected_chaves)]
        
        st.markdown(f"### Summary for Chave(s): `{', '.join(map(str, selected_chaves))}`")
        
        # Create pivot table with AssessorReal as rows and sum of financial columns
        financial_cols = ["Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        
        # Group by AssessorReal and sum the financial columns
        summary_df = (
            df_filtered.groupby("AssessorReal")[financial_cols]
            .sum()
            .reset_index()
        )
        
        # Sort by Lucro_Empresa (descending - greater to lower)
        summary_df = summary_df.sort_values("Lucro_Empresa", ascending=False)
        
        # Add totals row
        totals_row = pd.DataFrame({
            "AssessorReal": ["TOTAL"],
            "Comissão": [summary_df["Comissão"].sum()],
            "Tributo_Retido": [summary_df["Tributo_Retido"].sum()],
            "Pix_Assessor": [summary_df["Pix_Assessor"].sum()],
            "Lucro_Empresa": [summary_df["Lucro_Empresa"].sum()]
        })
        
        summary_with_totals = pd.concat([summary_df, totals_row], ignore_index=True)
        
        # Display the table
        st.dataframe(summary_with_totals.round(2), use_container_width=True ,height=460)
        
        # Export CSV
        csv = summary_with_totals.round(2).to_csv(index=False).encode("utf-8")
        filename = f"Macro_Summary_{'_'.join(map(str, selected_chaves))}.csv"
        st.download_button("📥 Download Summary CSV", csv, filename, "text/csv")
        
    else:
        st.warning("Please select at least one Chave.")

# --- ASSESSOR VIEW PAGE ---
elif page == "Assessor View":
    st.title("👤 Assessor View - Breakdown by Category")
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    df_taurus = st.session_state["df_taurus"]
    
    # Chave filter
    st.markdown("### 🔍 Filter by Chave")
    chave_list = sorted(df_taurus["Chave"].dropna().unique())
    selected_chaves = st.multiselect(
        "Select Chave period(s)",
        chave_list,
        default=chave_list
    )
    
    # Assessor selection
    assessor_list = sorted(df_taurus["AssessorReal"].dropna().unique())
    selected_assessor = st.selectbox("Select AssessorReal", assessor_list)
    
    # Filter data by selected assessor and Chave
    df_filtered = df_taurus[
        (df_taurus["AssessorReal"] == selected_assessor) &
        (df_taurus["Chave"].isin(selected_chaves))
    ]
    
    if df_filtered.empty:
        st.warning("No data for the selected AssessorReal & Chave combination.")
    else:
        st.markdown(
            f"### Summary for `{selected_assessor}` "
            f"(Chave: {', '.join(map(str, selected_chaves))})"
        )
        
        # Financial columns to sum
        financial_cols = ["Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        
        # Group by Categoria and sum financial columns
        category_summary = (
            df_filtered
            .groupby("Categoria")[financial_cols]
            .sum()
            .reset_index()
        )
        
        # Add totals row
        totals_row = pd.DataFrame({
            "Categoria": ["TOTAL"],
            "Comissão": [category_summary["Comissão"].sum()],
            "Tributo_Retido": [category_summary["Tributo_Retido"].sum()],
            "Pix_Assessor": [category_summary["Pix_Assessor"].sum()],
            "Lucro_Empresa": [category_summary["Lucro_Empresa"].sum()]
        })
        
        category_with_totals = pd.concat([category_summary, totals_row], ignore_index=True)
        
        # Display the table
        st.dataframe(category_with_totals.round(2), use_container_width=True )
        
        # --- DOWNLOAD SECTION ---
        st.markdown("### 📥 Download Options")
        
        # Create two columns for download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Export category summary CSV
            csv_summary = category_with_totals.round(2).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Download Category Summary CSV",
                csv_summary,
                f"{selected_assessor}_Category_Summary_{'_'.join(map(str, selected_chaves))}.csv",
                "text/csv"
            )
        
        with col2:
            # Export detailed transactions Excel
            # Select only the required columns for the detailed export
            detailed_cols = [
                "Data Receita", "Conta", "Cliente", "Comissão", 
                "Receita Assessor", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"
            ]
            
            # Check which columns exist in the dataframe
            available_cols = [col for col in detailed_cols if col in df_filtered.columns]
            
            if available_cols:
                # Create detailed export with available columns
                detailed_export = df_filtered[available_cols].copy()
                
                # Round numeric columns to 2 decimal places
                numeric_cols = detailed_export.select_dtypes(include=['float64', 'int64']).columns
                detailed_export[numeric_cols] = detailed_export[numeric_cols].round(2)
                
                # Convert to Excel format
                from io import BytesIO
                buffer = BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    detailed_export.to_excel(writer, sheet_name='Payment_Details', index=False)
                
                excel_data = buffer.getvalue()
                
                st.download_button(
                    "📋 Download Detailed Payment Info (Excel)",
                    excel_data,
                    f"{selected_assessor}_Payment_Details_{'_'.join(map(str, selected_chaves))}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Required columns for detailed export not found in the data.")
        
        # Show preview of detailed data
        if 'detailed_export' in locals():
            st.markdown("### 👀 Preview of Detailed Payment Information")
            st.dataframe(detailed_export.head(10), use_container_width=True)
            st.info(f"📊 Total transactions for {selected_assessor}: {len(detailed_export)}")

# --- PROFIT PAGE ---
elif page == "Profit":
    st.title("💰 Profit Summary - Lucro_Empresa by Chave")
    
    if st.session_state["df_taurus"] is None:
        st.warning("Please upload the Excel file in the Upload section first.")
        st.stop()
    
    df_taurus = st.session_state["df_taurus"]
    
    # Chave filter
    st.markdown("### 🔍 Filter Options")
    chave_list = sorted(df_taurus["Chave"].dropna().unique())
    selected_chaves = st.multiselect(
        "Select Chave periods to include (leave empty for all)",
        chave_list,
        default=chave_list  # Default to all selected
    )
    
    # Filter data based on selection
    if selected_chaves:
        df_filtered = df_taurus[df_taurus["Chave"].isin(selected_chaves)]
    else:
        df_filtered = df_taurus
    
    # Group by Chave and sum Lucro_Empresa
    profit_summary = (
        df_filtered.groupby("Chave")["Lucro_Empresa"]
        .sum()
        .reset_index()
        .sort_values("Chave")
    )
    
    # Calculate total sum
    total_sum = profit_summary["Lucro_Empresa"].sum()
    
    st.markdown("### 📈 Lucro_Empresa by Chave")
    
    # Display total sum
    st.metric(
        label="💰 Total Lucro_Empresa",
        value=f"{total_sum:,.2f}",
        help="Sum of all Lucro_Empresa values in the chart below"
    )
    
    # Display bar chart
    st.bar_chart(profit_summary.set_index("Chave"))
    
    # Show summary table
    st.markdown("### 📊 Summary Table")
    st.dataframe(profit_summary.round(2), use_container_width=True )
    
    # Download button
    csv = profit_summary.to_csv(index=False).encode("utf-8")
    filename = f"Profit_by_Chave_{'_'.join(map(str, selected_chaves)) if selected_chaves else 'All'}.csv"
    st.download_button("📥 Download Profit CSV", csv, filename, "text/csv")

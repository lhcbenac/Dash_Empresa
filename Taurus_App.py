import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar
from io import BytesIO # Import BytesIO for Excel export

# --- CONFIG ---
st.set_page_config(
    page_title="Taurus Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* General body and container styling */
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 2.8rem; /* Slightly larger for impact */
        font-weight: bold;
        color: #2c3e50; /* Darker blue/grey for professionalism */
        text-align: center;
        margin-bottom: 2.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0; /* Subtle underline */
    }
    .stApp {
        padding-top: 1rem; /* Add some space at the top */
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(45deg, #1f77b4, #2ca02c); /* Blend of blues and greens */
        padding: 1.5rem; /* More padding */
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.75rem 0; /* More margin for separation */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Soft shadow for depth */
        transition: transform 0.2s ease-in-out; /* Smooth hover effect */
    }
    .metric-card:hover {
        transform: translateY(-5px); /* Lift card on hover */
    }
    .metric-card h3 {
        color: white; /* Ensure text is white for contrast */
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        font-size: 2.2rem; /* Larger font for key metrics */
        font-weight: bold;
        margin: 0;
    }

    /* Streamlit widgets styling */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div {
        background-color: #ffffff; /* White background for inputs */
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stSelectbox label, .stMultiSelect label, .stNumberInput label {
        font-weight: bold;
        color: #333;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }

    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
        border-collapse: collapse;
        width: 100%;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-radius: 8px; /* Rounded corners for the table */
        overflow: hidden; /* Ensures content stays within rounded corners */
    }
    .dataframe th {
        background-color: #e0e0e0;
        color: #333;
        font-weight: bold;
        padding: 10px;
        text-align: left;
    }
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f8f8f8;
    }
    .dataframe tr:hover {
        background-color: #f1f1f1;
    }

    /* Profit Text */
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2c8ed6;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Sidebar */
    .css-1d391kg { /* This targets the sidebar directly, might change with Streamlit updates */
        background-color: #ffffff; /* White sidebar background */
        padding: 1.5rem 1rem;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
    }
    .st-emotion-cache-1jmve5q { /* Title in sidebar */
        color: #1f77b4;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
    }
    .st-emotion-cache-vk336y { /* Radio buttons for navigation */
        font-size: 1.1rem;
    }
    .st-emotion-cache-vk336y label div {
        padding: 0.5rem 0.75rem;
        border-radius: 5px;
        transition: background-color 0.2s ease;
    }
    .st-emotion-cache-vk336y label div:hover {
        background-color: #e6f0f8;
    }
    .st-emotion-cache-vk336y label div.st-af.st-ag.st-ah.st-ai.st-aj.st-ak.st-al.st-am.st-an.st-ao.st-ap.st-aq.st-ar.st-as.st-at { /* Selected radio button */
        background-color: #e0eaf3 !important; /* Lighter blue for selected state */
        color: #1f77b4 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 Taurus Analytics")
page = st.sidebar.radio("Navigation", [
    "📤 Upload",
    "📊 Executive Dashboard",
    "🌍 Macro View",
    "👤 Assessor View",
    "📈 Performance Analytics"
])

# --- SESSION STORAGE ---
if "df_taurus" not in st.session_state:
    st.session_state["df_taurus"] = None

# --- HELPER FUNCTIONS ---
def parse_chave_to_date(chave):
    """Convert Chave format (MM_YYYY) to datetime"""
    try:
        month, year = str(chave).split('_') # Ensure chave is string
        return datetime(int(year), int(month), 1)
    except:
        return None

def format_currency(value):
    """Format currency with proper formatting (Brazilian standard)"""
    return f"R\$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def calculate_growth_rate(current, previous):
    """Calculate growth rate between two values, handling zero previous to avoid division by zero"""
    if previous == 0:
        return 0.0 # Or np.nan if you prefer to represent it as Not a Number
    return ((current - previous) / previous) * 100

def create_gauge_chart(value, title, max_val=None, delta_ref=None):
    """Create a gauge chart for KPIs with optional delta"""
    if max_val is None:
        max_val = value * 1.5 if value > 0 else 100 # Default max_val to allow gauge to show progress
    
    delta_args = {}
    if delta_ref is not None:
        delta_args = {
            'reference': delta_ref,
            'relative': True,
            'valueformat': ".2%",
            'increasing': {'color': "#28a745"},
            'decreasing': {'color': "#dc3545"}
        }

    fig = go.Figure(go.Indicator(
        mode = "gauge+number" + ("+delta" if delta_ref is not None else ""),
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f'<b>{title}</b>', 'font': {'size': 24}},
        delta = delta_args,
        gauge = {
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1f77b4"}, # Brighter blue bar
            'steps': [
                {'range': [0, max_val * 0.6], 'color': "lightgray"},
                {'range': [max_val * 0.6, max_val], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9 # Threshold at 90% of max_val
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- UPLOAD PAGE ---
if page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload & Data Management</h1>', unsafe_allow_html=True)
    
    st.info("💡 **Instruções:** Por favor, carregue seu arquivo Excel de dados financeiros. Certifique-se de que ele contenha uma planilha chamada 'Taurus' com todas as colunas necessárias para uma análise completa.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Carregue seu arquivo Excel Taurus",
            type=["xlsx"],
            help="O arquivo deve conter uma planilha 'Taurus' com as colunas necessárias."
        )
    
    with col2:
        # Added "Receita Bruta" to the required columns list here
        st.info("📋 **Colunas Obrigatórias:**\n- Chave\n- AssessorReal\n- Categoria\n- Receita Bruta\n- Comissão\n- Tributo_Retido\n- Pix_Assessor\n- Lucro_Empresa\n- Data Receita (Opcional, mas recomendado para análises avançadas)")
    
    if uploaded_file:
        try:
            # Read specifically the 'Taurus' sheet
            df_taurus = pd.read_excel(uploaded_file, sheet_name="Taurus", engine="openpyxl")
            
            # Check if required columns exist (updated to include "Receita Bruta")
            required_cols = {"Chave", "AssessorReal", "Categoria", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa", "Receita Bruta"}
            
            if not required_cols.issubset(df_taurus.columns):
                missing_cols = required_cols - set(df_taurus.columns)
                st.error(f"❌ Colunas obrigatórias ausentes: {', '.join(missing_cols)}. Por favor, verifique seu arquivo.")
            else:
                # Data preprocessing
                df_taurus['Chave_Date'] = df_taurus['Chave'].apply(parse_chave_to_date)
                df_taurus['Month_Year'] = df_taurus['Chave_Date'].dt.strftime('%Y-%m')
                
                # Store data in session state
                st.session_state["df_taurus"] = df_taurus
                st.success("✅ Dados carregados e processados com sucesso! Você pode agora navegar pelos dashboards.")
                
                # Enhanced data overview
                st.markdown("### 📊 Visão Geral dos Dados")
                
                # Use st.container for better visual grouping
                with st.container(border=True):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.markdown(f"<div class='metric-card'><h3>Total de Transações</h3><p>{len(df_taurus):,}</p></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='metric-card'><h3>Assessores Únicos</h3><p>{df_taurus['AssessorReal'].nunique()}</p></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<div class='metric-card'><h3>Períodos</h3><p>{df_taurus['Chave'].nunique()}</p></div>", unsafe_allow_html=True)
                    with col4:
                        total_revenue = df_taurus["Comissão"].sum()
                        st.markdown(f"<div class='metric-card'><h3>Total Comissão</h3><p>{format_currency(total_revenue)}</p></div>", unsafe_allow_html=True)
                    with col5:
                        total_brute_revenue = df_taurus["Receita Bruta"].sum()
                        st.markdown(f"<div class='metric-card'><h3>Total Receita Bruta</h3><p>{format_currency(total_brute_revenue)}</p></div>", unsafe_allow_html=True)
                
                # Data quality checks
                st.markdown("### 🔍 Avaliação da Qualidade dos Dados")
                with st.expander("Clique para ver detalhes da qualidade dos dados"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        missing_data = df_taurus.isnull().sum()
                        missing_data_cols = missing_data[missing_data > 0]
                        if not missing_data_cols.empty:
                            st.warning("⚠️ Dados Ausentes Detectados nas seguintes colunas:")
                            st.dataframe(missing_data_cols.to_frame(name='Missing Count'))
                        else:
                            st.success("✅ Nenhuma informação ausente detectada em colunas críticas.")
                    
                    with col2:
                        # Date range
                        if 'Data Receita' in df_taurus.columns:
                            min_date = df_taurus['Data Receita'].min()
                            max_date = df_taurus['Data Receita'].max()
                            if pd.isna(min_date) or pd.isna(max_date):
                                st.info("📅 **Período de Datas:** Não disponível ou coluna 'Data Receita' incompleta.")
                            else:
                                st.info(f"📅 **Período de Datas:** {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}")
                        else:
                            st.info("📅 Coluna 'Data Receita' não encontrada para avaliação do período de datas.")
                        
                        # Top categories
                        top_categories = df_taurus['Categoria'].value_counts().head(3)
                        if not top_categories.empty:
                            st.info("🏆 **Principais Categorias:**\n" + "\n".join([f"• {cat}: {count}" for cat, count in top_categories.items()]))
                        else:
                            st.info("🏆 Nenhuma categoria encontrada.")
                
                # Sample data with better formatting
                st.markdown("### 👀 Pré-visualização dos Dados")
                with st.expander("Clique para ver uma amostra dos dados"):
                    # Added "Receita Bruta" to display_cols
                    display_cols = ['Chave', 'AssessorReal', 'Categoria', 'Receita Bruta', 'Comissão', 'Pix_Assessor', 'Lucro_Empresa']
                    sample_data = df_taurus[display_cols].head(10)
                    st.dataframe(sample_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo. Por favor, certifique-se de que o arquivo é um Excel válido e contém uma planilha 'Taurus' com os dados corretos. Detalhes do erro: {e}")

# --- EXECUTIVE DASHBOARD ---
elif page == "📊 Executive Dashboard":
    st.markdown('<h1 class="main-header">📊 Dashboard Executivo</h1>', unsafe_allow_html=True)

    if st.session_state["df_taurus"] is None:
        st.warning("Por favor, carregue o arquivo Excel primeiro para visualizar o dashboard.")
        st.stop()

    df = st.session_state["df_taurus"]

    # Time Period Filter
    with st.container(border=True):
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            chave_list = sorted(df["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "🕐 Selecione Períodos",
                chave_list,
                default=chave_list[-6:] if len(chave_list) >= 6 else chave_list,
                help="Selecione um ou mais períodos para filtrar os dados do dashboard."
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
            show_growth = st.checkbox("Mostrar Crescimento (vs. Período Anterior)", value=True, help="Compara as métricas com o período imediatamente anterior (disponível apenas para um único período selecionado).")

    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]
        
        # Calculate current period metrics
        total_revenue_current = df_filtered["Comissão"].sum()
        total_brute_revenue_current = df_filtered["Receita Bruta"].sum()
        total_pix_current = df_filtered["Pix_Assessor"].sum()
        total_profit_current = df_filtered["Lucro_Empresa"].sum()
        avg_transaction_current = df_filtered["Comissão"].mean()
        active_assessors_current = df_filtered["AssessorReal"].nunique()

        # Calculate previous period metrics for growth comparison
        total_revenue_prev = 0
        total_brute_revenue_prev = 0
        total_pix_prev = 0
        total_profit_prev = 0
        avg_transaction_prev = 0

        if show_growth and len(selected_chaves) == 1:
            current_chave_date = parse_chave_to_date(selected_chaves[0])
            if current_chave_date:
                # Calculate previous month's 'Chave'
                prev_month_date = current_chave_date - timedelta(days=1)
                prev_month_chave = prev_month_date.strftime('%m_%Y')
                
                df_prev_period = df[df["Chave"] == prev_month_chave]

                if not df_prev_period.empty:
                    total_revenue_prev = df_prev_period["Comissão"].sum()
                    total_brute_revenue_prev = df_prev_period["Receita Bruta"].sum()
                    total_pix_prev = df_prev_period["Pix_Assessor"].sum()
                    total_profit_prev = df_prev_period["Lucro_Empresa"].sum()
                    avg_transaction_prev = df_prev_period["Comissão"].mean()

        st.markdown("### 🎯 Indicadores Chave de Performance")
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Comissão", format_currency(total_revenue_current), delta=f"{calculate_growth_rate(total_revenue_current, total_revenue_prev):.2f}%" if show_growth and len(selected_chaves) == 1 else None)
            with col2:
                st.metric("Total Receita Bruta", format_currency(total_brute_revenue_current), delta=f"{calculate_growth_rate(total_brute_revenue_current, total_brute_revenue_prev):.2f}%" if show_growth and len(selected_chaves) == 1 else None)
            with col3:
                st.metric("Total Pix Assessor", format_currency(total_pix_current), delta=f"{calculate_growth_rate(total_pix_current, total_pix_prev):.2f}%" if show_growth and len(selected_chaves) == 1 else None)
            with col4:
                st.metric("Lucro da Empresa", format_currency(total_profit_current), delta=f"{calculate_growth_rate(total_profit_current, total_profit_prev):.2f}%" if show_growth and len(selected_chaves) == 1 else None)
            with col5:
                st.metric("Assessores Ativos", active_assessors_current)
        
        st.markdown("---") # Separator for charts

        # Charts Row 1
        st.markdown("### 📈 Visualizações de Performance")
        col1, col2 = st.columns(2)

        with col1:
            # Revenue Evolution
            # Ensure 'Chave_Date' column is used for sorting the evolution chart
            monthly_revenue = df_filtered.groupby('Chave')['Comissão'].sum().reset_index()
            # If Chave_Date is missing, recalculate for the current filtered data
            if 'Chave_Date' not in monthly_revenue.columns:
                monthly_revenue['Chave_Date'] = monthly_revenue['Chave'].apply(parse_chave_to_date)
            monthly_revenue = monthly_revenue.sort_values('Chave_Date')

            fig_revenue = px.line(
                monthly_revenue,
                x='Chave',
                y='Comissão',
                title='📈 Evolução da Comissão Mensal',
                markers=True,
                line_shape='spline', # Smooth lines
                color_discrete_sequence=px.colors.qualitative.Plotly # Consistent color
            )
            fig_revenue.update_layout(xaxis_title="Período", yaxis_title="Comissão (R\$)", hovermode="x unified")
            fig_revenue.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_revenue, use_container_width=True)

        with col2:
            # Top Assessors
            top_assessors = df_filtered.groupby('AssessorReal')['Comissão'].sum().nlargest(10).sort_values(ascending=True) # Sort for bar chart display
            fig_assessors = px.bar(
                x=top_assessors.values,
                y=top_assessors.index,
                orientation='h',
                title='🏆 Top 10 Assessores por Comissão',
                color_discrete_sequence=px.colors.qualitative.Pastel # Use a different color palette
            )
            fig_assessors.update_layout(xaxis_title="Comissão (R\$)", yaxis_title="Assessor", showlegend=False)
            st.plotly_chart(fig_assessors, use_container_width=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            # Category Distribution
            category_dist = df_filtered.groupby('Categoria')['Comissão'].sum().reset_index()
            fig_pie = px.pie(
                category_dist,
                values='Comissão',
                names='Categoria',
                title='🎯 Distribuição da Comissão por Categoria',
                hole=0.3, # Make it a donut chart
                color_discrete_sequence=px.colors.sequential.RdBu # Sequential colors for segments
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Profit Margin Analysis
            profit_margin = df_filtered.groupby('Chave').agg({
                'Comissão': 'sum',
                'Lucro_Empresa': 'sum'
            }).reset_index()
            profit_margin['Margin_Percent'] = (profit_margin['Lucro_Empresa'] / profit_margin['Comissão']) * 100
            profit_margin.fillna(0, inplace=True) # Handle division by zero if Comissão is 0

            fig_margin = px.bar(
                profit_margin,
                x='Chave',
                y='Margin_Percent',
                title='📊 Margem de Lucro por Período (%)',
                color='Margin_Percent',
                color_continuous_scale='RdYlGn' # Green for higher margin, Red for lower
            )
            fig_margin.update_layout(xaxis_title="Período", yaxis_title="Margem de Lucro (%)")
            st.plotly_chart(fig_margin, use_container_width=True)
    else:
        st.info("Por favor, selecione pelo menos um período para exibir o Dashboard Executivo.")

# --- MACRO VIEW PAGE ---
elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Visão Macro - Performance do Assessor</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Por favor, carregue o arquivo Excel primeiro.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Filters
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            chave_list = sorted(df["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "🕐 Selecione Períodos",
                chave_list,
                default=chave_list,
                help="Filtre os dados pelos meses/anos selecionados."
            )
        
        with col2:
            min_revenue = st.number_input("💰 Filtro de Comissão Mínima (R\$)", min_value=0.0, value=0.0, step=1000.0,
                                          help="Exibe apenas assessores com Comissão acima deste valor.")
    
    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]
        
        # Summary calculations
        # Added "Receita Bruta" to financial_cols
        financial_cols = ["Receita Bruta", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        summary_df = df_filtered.groupby("AssessorReal")[financial_cols].sum().reset_index()
        
        # Add calculated metrics
        # Correctly get counts for Transaction_Count
        transaction_counts = df_filtered.groupby("AssessorReal").size().reset_index(name='Transaction_Count')
        summary_df = pd.merge(summary_df, transaction_counts, on='AssessorReal', how='left')

        summary_df['Avg_Transaction'] = summary_df['Comissão'] / summary_df['Transaction_Count']
        summary_df['Profit_Margin'] = (summary_df['Lucro_Empresa'] / summary_df['Comissão']) * 100
        summary_df.fillna({'Avg_Transaction': 0, 'Profit_Margin': 0}, inplace=True) # Handle division by zero
        
        # Filter by minimum revenue
        summary_df = summary_df[summary_df['Comissão'] >= min_revenue]
        summary_df = summary_df.sort_values("Comissão", ascending=False)
        
        # Display metrics
        st.markdown("### 📊 Métricas de Performance Geral")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Assessores Únicos", len(summary_df))
            with col2:
                st.metric("Comissão Agregada", format_currency(summary_df['Comissão'].sum()))
            with col3:
                st.metric("Comissão Média por Assessor", format_currency(summary_df['Comissão'].mean()))
        
        # Enhanced table with formatting
        st.markdown("### 📋 Resumo da Performance do Assessor")
        if not summary_df.empty:
            # Format the display dataframe
            display_df = summary_df.copy()
            for col in financial_cols:
                display_df[col] = display_df[col].apply(format_currency)
            display_df['Avg_Transaction'] = display_df['Avg_Transaction'].apply(format_currency)
            display_df['Profit_Margin'] = display_df['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.info("Nenhum dado disponível para os filtros selecionados.")
        
        # Visualizations
        st.markdown("### 📈 Visualizações de Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            if not summary_df.empty:
                # Top performers
                top_10 = summary_df.head(10).sort_values("Comissão", ascending=True)
                fig_top = px.bar(
                    top_10,
                    x='Comissão',
                    y='AssessorReal',
                    orientation='h',
                    title='🏆 Top 10 Assessores por Comissão',
                    color='Profit_Margin',
                    color_continuous_scale='RdYlGn',
                    labels={'Comissão': 'Comissão (R\$)', 'AssessorReal': 'Assessor', 'Profit_Margin': 'Margem de Lucro (%)'}
                )
                fig_top.update_layout(showlegend=True, hovermode="y unified")
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info("Nenhum dado para exibir o gráfico dos Top 10 Performers.")
        
        with col2:
            if not summary_df.empty:
                # Scatter plot: Comissão vs Profit Margin
                fig_scatter = px.scatter(
                    summary_df,
                    x='Comissão',
                    y='Profit_Margin',
                    size='Transaction_Count',
                    hover_name='AssessorReal',
                    color='Profit_Margin',
                    color_continuous_scale='RdYlGn',
                    title='💰 Comissão vs Margem de Lucro (Tamanho da Bolha: Transações)',
                    labels={'Comissão': 'Comissão (R\$)', 'Profit_Margin': 'Margem de Lucro (%)', 'Transaction_Count': 'Número de Transações'}
                )
                fig_scatter.update_layout(hovermode="closest")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Nenhum dado para exibir o gráfico Comissão vs Margem de Lucro.")
        
        # Export options
        st.markdown("### 📥 Opções de Exportação")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Baixar Resumo Completo (CSV)",
                csv_data,
                f"Resumo_Macro_{'_'.join(selected_chaves)}.csv",
                "text/csv",
                help="Baixe todos os dados resumidos dos assessores em formato CSV."
            )
        
        with col2:
            # Top performers only
            top_performers_df = summary_df.head(20)
            if not top_performers_df.empty:
                csv_top = top_performers_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🏆 Baixar Top 20 Performers (CSV)",
                    csv_top,
                    f"Top_Performers_{'_'.join(selected_chaves)}.csv",
                    "text/csv",
                    help="Baixe o resumo dos 20 assessores com melhor desempenho por Comissão."
                )
            else:
                st.info("Nenhum top performer para baixar.")

    else:
        st.info("Por favor, selecione pelo menos um período para ver a Visão Macro.")

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Análise de Assessor Individual</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Por favor, carregue o arquivo Excel primeiro.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Filters
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            chave_list = sorted(df["Chave"].dropna().unique())
            selected_chaves = st.multiselect(
                "🕐 Selecione Períodos",
                chave_list,
                default=chave_list,
                help="Filtre os dados para o assessor selecionado por mês/ano."
            )
        
        with col2:
            assessor_list = sorted(df["AssessorReal"].dropna().unique())
            selected_assessor = st.selectbox("👤 Selecione Assessor", assessor_list,
                                             help="Escolha um assessor para ver sua performance detalhada.")
    
    if selected_chaves and selected_assessor:
        df_filtered = df[
            (df["AssessorReal"] == selected_assessor) &
            (df["Chave"].isin(selected_chaves))
        ]
        
        if df_filtered.empty:
            st.warning("Nenhum dado encontrado para o assessor e períodos selecionados. Por favor, ajuste seus filtros.")
        else:
            # Individual KPIs
            st.markdown(f"### 📊 Visão Geral da Performance: {selected_assessor}")
            
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns(5)

                total_brute_revenue = df_filtered["Receita Bruta"].sum() # Added Receita Bruta
                total_comissao = df_filtered["Comissão"].sum()
                total_transactions = len(df_filtered)
                total_pix = df_filtered["Pix_Assessor"].sum()
                total_profit = df_filtered["Lucro_Empresa"].sum()
                
                # FIX: Calculate avg_transaction here for this specific filtered data
                avg_transaction = df_filtered["Comissão"].mean()
                if pd.isna(avg_transaction): # Handle case where Comissão might be empty or 0
                    avg_transaction = 0.0

                with col1:
                    st.metric("Total Receita Bruta", format_currency(total_brute_revenue)) # Added Receita Bruta metric
                with col2:
                    st.metric("Total Comissão", format_currency(total_comissao))
                with col3:
                    st.metric("Total de Transações", total_transactions)
                with col4:
                    st.metric("Total Pix Assessor", format_currency(total_pix))
                with col5:
                    st.metric("Lucro Gerado", format_currency(total_profit))
            
            st.markdown("---") # Separator

            # Performance over time chart
            st.markdown("### 📈 Tendência de Performance Mensal")
            monthly_performance = df_filtered.groupby('Chave').agg(
                Comissão=('Comissão', 'sum'),
                Lucro_Empresa=('Lucro_Empresa', 'sum'),
                Transaction_Count=('Chave', 'count')
            ).reset_index()
            # Ensure 'Chave_Date' column is present for sorting
            monthly_performance['Chave_Date'] = monthly_performance['Chave'].apply(parse_chave_to_date)
            monthly_performance = monthly_performance.sort_values('Chave_Date')

            fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
            fig_monthly.add_trace(go.Scatter(x=monthly_performance['Chave'], y=monthly_performance['Comissão'],
                                             mode='lines+markers', name='Comissão', line=dict(color='blue')),
                                   secondary_y=False)
            fig_monthly.add_trace(go.Scatter(x=monthly_performance['Chave'], y=monthly_performance['Lucro_Empresa'],
                                             mode='lines+markers', name='Lucro Empresa', line=dict(color='green')),
                                   secondary_y=False)
            fig_monthly.add_trace(go.Bar(x=monthly_performance['Chave'], y=monthly_performance['Transaction_Count'],
                                         name='Transações', marker_color='lightgray', opacity=0.5),
                                   secondary_y=True)
            
            fig_monthly.update_layout(title_text=f"<b>Performance de {selected_assessor} ao Longo do Tempo</b>", hovermode="x unified")
            fig_monthly.update_xaxes(title_text="Período")
            fig_monthly.update_yaxes(title_text="Comissão / Lucro (R\$)", secondary_y=False)
            fig_monthly.update_yaxes(title_text="Transações", secondary_y=True)
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Category breakdown
            financial_cols_category = ["Receita Bruta", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"] # Added Receita Bruta
            category_summary = df_filtered.groupby("Categoria")[financial_cols_category].sum().reset_index()
            
            # Add transaction count per category
            transaction_counts_category = df_filtered.groupby("Categoria").size().reset_index(name='Transaction_Count')
            category_summary = pd.merge(category_summary, transaction_counts_category, on='Categoria', how='left')

            st.markdown("### 📋 Performance por Categoria")
            if not category_summary.empty:
                st.dataframe(category_summary.round(2), use_container_width=True)
            else:
                st.info("Nenhum dado de categoria disponível para este assessor e período.")
            
            # Category visualization (Treemap)
            if not category_summary.empty:
                fig_category = px.treemap(
                    category_summary,
                    path=['Categoria'],
                    values='Comissão',
                    title=f'🎯 {selected_assessor} - Comissão por Categoria',
                    color='Lucro_Empresa',
                    color_continuous_scale='RdYlGn',
                    hover_data=['Receita Bruta', 'Tributo_Retido', 'Pix_Assessor', 'Transaction_Count'] # Added hover data
                )
                fig_category.update_layout(margin = dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig_category, use_container_width=True)
            else:
                st.info("Nenhum dado para exibir o Treemap de Categoria.")
            
            # Download section
            st.markdown("### 📥 Opções de Exportação")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Category summary
                if not category_summary.empty:
                    csv_summary = category_summary.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Baixar Resumo por Categoria (CSV)",
                        csv_summary,
                        f"{selected_assessor}_Resumo_Categoria.csv",
                        "text/csv",
                        help="Exporte o resumo de performance por categoria para este assessor."
                    )
                else:
                    st.info("Nenhum resumo de categoria para baixar.")
            
            with col2:
                # Detailed transactions and other summaries in one Excel
                if not df_filtered.empty:
                    # UPDATED: Added "Receita Bruta" to detailed_cols
                    detailed_cols = [
                        "Data Receita", "Conta", "Cliente", "AssessorReal", "Categoria", "Produto",
                        "Receita Bruta", "Comissão", "Receita Assessor", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa", "Chave"
                    ]
                    available_cols = [col for col in detailed_cols if col in df_filtered.columns]
                    
                    buffer = BytesIO()
                    
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_filtered[available_cols].to_excel(writer, sheet_name=f'{selected_assessor}_DadosBrutos', index=False)
                        category_summary.to_excel(writer, sheet_name=f'{selected_assessor}_ResumoCategoria', index=False)
                        monthly_performance.to_excel(writer, sheet_name=f'{selected_assessor}_PerformanceMensal', index=False)
                    
                    st.download_button(
                        "📋 Baixar Relatório Completo (Excel)",
                        buffer.getvalue(),
                        f"{selected_assessor}_Relatorio_Completo.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Baixe um relatório Excel completo incluindo dados brutos, resumo por categoria e performance mensal."
                    )
                else:
                    st.info("Nenhum dado para gerar um relatório completo.")
            
            with col3:
                # Performance summary (FIXED: avg_transaction is now defined)
                performance_summary = pd.DataFrame({
                    'Métrica': ['Total Receita Bruta', 'Total Comissão', 'Total de Transações', 'Valor Médio por Transação', 'Lucro Total'],
                    'Valor': [total_brute_revenue, total_comissao, total_transactions, avg_transaction, total_profit]
                })
                csv_perf = performance_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🎯 Baixar Resumo de Performance (CSV)",
                    csv_perf,
                    f"{selected_assessor}_Resumo_Performance.csv",
                    "text/csv",
                    help="Obtenha um resumo rápido em CSV dos principais indicadores de performance para este assessor."
                )
    else:
        st.info("Por favor, selecione períodos e um assessor para ver sua análise detalhada.")

# --- PERFORMANCE ANALYTICS ---
elif page == "📈 Performance Analytics":
    st.markdown('<h1 class="main-header">📈 Análise Avançada de Performance</h1>', unsafe_allow_html=True)
    
    if st.session_state["df_taurus"] is None:
        st.warning("Por favor, carregue o arquivo Excel primeiro para realizar análises avançadas.")
        st.stop()
    
    df = st.session_state["df_taurus"]
    
    # Analytics options
    analysis_type = st.selectbox(
        "📊 Selecione o Tipo de Análise",
        ["Análise de Tendência", "Análise Comparativa", "Análise Sazonal", "Análise de Crescimento"],
        help="Escolha um tipo de análise avançada para realizar em seus dados."
    )
    
    # Common filter for all analytics types
    with st.container(border=True):
        chave_list_all = sorted(df["Chave"].dropna().unique())
        selected_chaves_analytics = st.multiselect(
            "🕐 Selecione Períodos para Análise",
            chave_list_all,
            default=chave_list_all,
            key="analytics_chave_filter",
            help="Aplique um filtro de período à análise selecionada."
        )

    if not selected_chaves_analytics:
        st.info("Por favor, selecione pelo menos um período para realizar a análise.")
        st.stop()

    df_analytics_filtered = df[df["Chave"].isin(selected_chaves_analytics)]

    if analysis_type == "Análise de Tendência":
        st.markdown("### 📈 Tendências Mensais para Métricas Chave")
        
        # Time series analysis
        monthly_data = df_analytics_filtered.groupby('Chave').agg(
            Comissão=('Comissão', 'sum'),
            Lucro_Empresa=('Lucro_Empresa', 'sum'),
            Pix_Assessor=('Pix_Assessor', 'sum'),
            Active_Assessors=('AssessorReal', 'nunique')
        ).reset_index()
        
        monthly_data['Chave_Date'] = monthly_data['Chave'].apply(parse_chave_to_date)
        monthly_data = monthly_data.sort_values('Chave_Date')
        
        if not monthly_data.empty:
            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Tendência da Comissão', 'Tendência do Lucro', 'Tendência do Pix Assessor', 'Tendência de Assessores Ativos'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Comissão trend
            fig.add_trace(
                go.Scatter(x=monthly_data['Chave'], y=monthly_data['Comissão'],
                          mode='lines+markers', name='Comissão', line=dict(color='#1f77b4')),
                row=1, col=1
            )
            
            # Profit trend
            fig.add_trace(
                go.Scatter(x=monthly_data['Chave'], y=monthly_data['Lucro_Empresa'],
                          mode='lines+markers', name='Lucro', line=dict(color='#2ca02c')),
                row=1, col=2
            )
            
            # Pix trend
            fig.add_trace(
                go.Scatter(x=monthly_data['Chave'], y=monthly_data['Pix_Assessor'],
                          mode='lines+markers', name='Pix Assessor', line=dict(color='#ff7f0e')),
                row=2, col=1
            )
            
            # Active assessors
            fig.add_trace(
                go.Scatter(x=monthly_data['Chave'], y=monthly_data['Active_Assessors'],
                          mode='lines+markers', name='Assessores Ativos', line=dict(color='#9467bd')),
                row=2, col=2
            )
            
            fig.update_layout(height=650, title_text="📊 Análise de Tendência Abrangente entre Métricas", hovermode="x unified",
                              showlegend=False) # Suppress legend as subplots are self-explanatory
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para análise de tendência no período selecionado.")
    
    elif analysis_type == "Análise Comparativa":
        st.markdown("### 🔍 Comparação de Assessores por Métricas Chave")
        
        # Top assessors selector
        top_n_compare = st.slider("Selecione os N assessores para comparação", 3, min(20, df_analytics_filtered['AssessorReal'].nunique()), 5)
        
        top_assessors_by_comissao = df_analytics_filtered.groupby('AssessorReal')['Comissão'].sum().nlargest(top_n_compare).index
        df_top_compare = df_analytics_filtered[df_analytics_filtered['AssessorReal'].isin(top_assessors_by_comissao)]
        
        if not df_top_compare.empty:
            # Comparison metrics
            comparison_data = df_top_compare.groupby('AssessorReal').agg(
                Total_Comissão=('Comissão', 'sum'),
                Total_Lucro=('Lucro_Empresa', 'sum'),
                Total_Pix=('Pix_Assessor', 'sum'),
                Transaction_Count=('Chave', 'count')
            ).reset_index() # .reset_index() to make 'AssessorReal' a column
            
            comparison_data['Avg_Transaction_Value'] = comparison_data['Total_Comissão'] / comparison_data['Transaction_Count']
            comparison_data['Profit_Margin'] = (comparison_data['Total_Lucro'] / comparison_data['Total_Comissão']) * 100
            comparison_data.fillna({'Avg_Transaction_Value': 0, 'Profit_Margin': 0}, inplace=True)

            # Radar chart
            st.markdown("#### Gráfico de Radar: Perfil de Performance dos Principais Assessores")
            st.info("Este gráfico de radar visualiza a performance relativa dos principais assessores selecionados em métricas normalizadas. Uma área maior indica uma performance geral mais forte.")
            
            fig_radar = go.Figure()
            
            metrics = ['Total_Comissão', 'Total_Lucro', 'Total_Pix', 'Avg_Transaction_Value', 'Profit_Margin']
            
            for assessor in comparison_data['AssessorReal']:
                assessor_data = comparison_data[comparison_data['AssessorReal'] == assessor].iloc[0]
                
                normalized_values = []
                for metric in metrics:
                    max_val = comparison_data[metric].max()
                    if max_val > 0:
                        normalized_values.append((assessor_data[metric] / max_val) * 100)
                    else:
                        normalized_values.append(0) # If max_val is 0, metric is 0

                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=metrics,
                    fill='toself',
                    name=assessor,
                    hoverinfo='text',
                    text=[f"{m}: {v:.2f}" for m, v in zip(metrics, [assessor_data[m] for m in metrics])]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="🎯 Comparação de Performance de Assessores Selecionados (Métricas Normalizadas)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("#### Tabela de Comparação Detalhada")
            display_comp_df = comparison_data.copy()
            display_comp_df['Total_Comissão'] = display_comp_df['Total_Comissão'].apply(format_currency)
            display_comp_df['Total_Lucro'] = display_comp_df['Total_Lucro'].apply(format_currency)
            display_comp_df['Total_Pix'] = display_comp_df['Total_Pix'].apply(format_currency)
            display_comp_df['Avg_Transaction_Value'] = display_comp_df['Avg_Transaction_Value'].apply(format_currency)
            display_comp_df['Profit_Margin'] = display_comp_df['Profit_Margin'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(display_comp_df, use_container_width=True)

        else:
            st.info("Nenhum dado disponível para análise comparativa no período selecionado ou para o número de assessores selecionado.")

    elif analysis_type == "Análise Sazonal":
        st.markdown("### 🗓️ Análise de Performance Sazonal")
        st.info("Analise como a Comissão e o Lucro flutuam ao longo dos diferentes meses do ano, fornecendo insights sobre a sazonalidade.")

        # Extract month from Chave_Date
        df_analytics_filtered['Month_Name'] = df_analytics_filtered['Chave_Date'].dt.strftime('%B')
        df_analytics_filtered['Month_Num'] = df_analytics_filtered['Chave_Date'].dt.month

        seasonal_data = df_analytics_filtered.groupby(['Month_Num', 'Month_Name']).agg(
            Comissão_Avg=('Comissão', 'mean'),
            Lucro_Avg=('Lucro_Empresa', 'mean')
        ).reset_index().sort_values('Month_Num')

        if not seasonal_data.empty:
            fig_seasonal = make_subplots(specs=[[{"secondary_y": True}]])

            fig_seasonal.add_trace(go.Bar(x=seasonal_data['Month_Name'], y=seasonal_data['Comissão_Avg'],
                                          name='Comissão Média', marker_color='#1f77b4'), secondary_y=False)
            fig_seasonal.add_trace(go.Scatter(x=seasonal_data['Month_Name'], y=seasonal_data['Lucro_Avg'],
                                              mode='lines+markers', name='Lucro Médio', line=dict(color='#2ca02c', width=3)),
                                   secondary_y=True)

            fig_seasonal.update_layout(title_text="<b>Comissão e Lucro Médios Mensais (Sazonal)</b>", hovermode="x unified")
            fig_seasonal.update_xaxes(title_text="Mês")
            fig_seasonal.update_yaxes(title_text="Comissão Média (R\$)", secondary_y=False)
            fig_seasonal.update_yaxes(title_text="Lucro Médio (R\$)", secondary_y=True)
            st.plotly_chart(fig_seasonal, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para análise sazonal no período selecionado.")

    elif analysis_type == "Análise de Crescimento":
        st.markdown("### 🚀 Análise de Crescimento Período-a-Período")
        st.info("Avalie a taxa de crescimento das principais métricas de um período para o próximo.")

        if df_analytics_filtered['Chave'].nunique() < 2:
            st.warning("Por favor, selecione pelo menos dois períodos ('Chave') distintos para a análise de crescimento.")
            st.stop()

        monthly_data_growth = df_analytics_filtered.groupby('Chave').agg(
            Comissão=('Comissão', 'sum'),
            Lucro_Empresa=('Lucro_Empresa', 'sum'),
            Receita_Bruta=('Receita Bruta', 'sum') # Added Receita Bruta
        ).reset_index()

        monthly_data_growth['Chave_Date'] = monthly_data_growth['Chave'].apply(parse_chave_to_date)
        monthly_data_growth = monthly_data_growth.sort_values('Chave_Date')

        # Calculate growth rates
        monthly_data_growth['Comissão_Growth'] = monthly_data_growth['Comissão'].pct_change() * 100
        monthly_data_growth['Lucro_Growth'] = monthly_data_growth['Lucro_Empresa'].pct_change() * 100
        monthly_data_growth['Receita_Bruta_Growth'] = monthly_data_growth['Receita_Bruta'].pct_change() * 100 # Added Receita Bruta growth

        # Drop the first row which will have NaN for growth
        monthly_data_growth.dropna(inplace=True)

        if not monthly_data_growth.empty:
            fig_growth = px.line(monthly_data_growth, x='Chave', y=['Comissão_Growth', 'Lucro_Growth', 'Receita_Bruta_Growth'], # Added Receita Bruta
                                 title='<b>Taxas de Crescimento Período-a-Período (%)</b>',
                                 labels={'value': 'Taxa de Crescimento (%)', 'variable': 'Métrica'},
                                 markers=True, line_shape='spline',
                                 color_discrete_map={
                                     'Comissão_Growth': '#1f77b4',
                                     'Lucro_Growth': '#2ca02c',
                                     'Receita_Bruta_Growth': '#ff7f0e' # Orange for Receita Bruta
                                 })
            fig_growth.update_layout(hovermode="x unified")
            fig_growth.update_yaxes(suffix="%")
            st.plotly_chart(fig_growth, use_container_width=True)

            st.markdown("#### Tabela de Taxa de Crescimento")
            display_growth_df = monthly_data_growth[['Chave', 'Comissão_Growth', 'Lucro_Growth', 'Receita_Bruta_Growth']].copy() # Added Receita Bruta
            display_growth_df['Comissão_Growth'] = display_growth_df['Comissão_Growth'].apply(lambda x: f"{x:.2f}%")
            display_growth_df['Lucro_Growth'] = display_growth_df['Lucro_Growth'].apply(lambda x: f"{x:.2f}%")
            display_growth_df['Receita_Bruta_Growth'] = display_growth_df['Receita_Bruta_Growth'].apply(lambda x: f"{x:.2f}%") # Added Receita Bruta
            st.dataframe(display_growth_df, use_container_width=True)

        else:
            st.info("Dados insuficientes ou períodos de crescimento não encontrados para análise. Certifique-se de que pelo menos dois períodos foram selecionados.")

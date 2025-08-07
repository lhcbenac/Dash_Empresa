import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import calendar
from io import BytesIO

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Taurus Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    /* Estilos Gerais */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4; /* Azul Taurus */
        text-align: center;
        margin-bottom: 2.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .st-emotion-cache-1jm6gjm { /* Botões */
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.7rem 1.2rem;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .st-emotion-cache-1jm6gjm:hover {
        background-color: #5a6ed1;
    }

    /* Cartões de Métrica */
    div.st-emotion-cache-nahz7x { /* Container do st.metric */
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Gradiente Azul/Roxo */
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    div.st-emotion-cache-nahz7x:hover {
        transform: translateY(-5px);
    }
    div.st-emotion-cache-nahz7x p { /* Título e Valor do st.metric */
        color: white !important;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    div.st-emotion-cache-nahz7x .st-emotion-cache-v06ywu { /* Valor do st.metric */
        font-size: 2rem;
        font-weight: bold;
    }

    /* Outros Elementos */
    .stSelectbox > div > div {
        background-color: #f0f2f6; /* Cor de fundo para selectbox */
        border-radius: 8px;
    }
    .profit-positive {
        color: #28a745; /* Verde */
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545; /* Vermelho */
        font-weight: bold;
    }
    .stAlert { /* Mensagens de Alerta */
        border-radius: 8px;
    }
    h3 {
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    .st-emotion-cache-1c7y2c9 { /* Título da barra lateral */
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# --- ESTADO DA SESSÃO ---
if "df_taurus" not in st.session_state:
    st.session_state["df_taurus"] = None

# --- FUNÇÕES AUXILIARES ---
def parse_chave_to_date(chave):
    """Converte o formato Chave (MM_AAAA) para datetime."""
    try:
        month, year = map(int, chave.split('_'))
        return datetime(year, month, 1)
    except (ValueError, AttributeError):
        return None

def format_currency(value):
    """Formata um valor numérico para o padrão monetário brasileiro."""
    return f"R\$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def calculate_growth_rate(current, previous):
    """Calcula a taxa de crescimento entre dois valores."""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def load_data(uploaded_file):
    """Carrega e pré-processa o arquivo Excel."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Taurus", engine="openpyxl")

        # Colunas obrigatórias, incluindo "Receita Bruta" que é usada mais tarde
        required_cols = {"Chave", "AssessorReal", "Categoria", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa", "Receita Bruta"}

        if not required_cols.issubset(df.columns):
            missing_cols = required_cols - set(df.columns)
            st.error(f"❌ Erro: Colunas obrigatórias ausentes: {', '.join(missing_cols)}. Verifique seu arquivo.")
            return None

        # Pré-processamento
        df['Chave_Date'] = df['Chave'].apply(parse_chave_to_date)
        df['Month_Year'] = df['Chave_Date'].dt.strftime('%Y-%m')

        return df

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}. Certifique-se de que é um arquivo Excel válido e contém a aba 'Taurus'.")
        return None

def display_kpis(df, is_assessor_view=False):
    """Exibe os cards de KPIs de forma padronizada."""
    col1, col2, col3, col4, col5 = st.columns(5)

    if is_assessor_view:
        total_revenue = df["Receita Bruta"].sum()
        total_commission = df["Comissão"].sum()
        total_transactions = len(df)
        total_pix = df["Pix_Assessor"].sum()
        total_profit = df["Lucro_Empresa"].sum()
        avg_transaction = total_commission / total_transactions if total_transactions > 0 else 0

        with col1:
            st.metric("Receita Bruta", format_currency(total_revenue))
        with col2:
            st.metric("Comissão Total", format_currency(total_commission))
        with col3:
            st.metric("Total Transações", total_transactions)
        with col4:
            st.metric("Pix Assessor", format_currency(total_pix))
        with col5:
            st.metric("Lucro Gerado", format_currency(total_profit))
        # Adicionar o Avg Transaction como um 6º KPI, talvez em uma nova linha ou otimizar o espaço
        # Para manter o layout 5 colunas, decidimos não adicionar um 6º card diretamente aqui, mas garantir que esteja no export.

        return total_revenue, total_commission, total_transactions, total_pix, total_profit, avg_transaction

    else: # Executive Dashboard
        total_commission = df["Comissão"].sum()
        total_pix = df["Pix_Assessor"].sum()
        total_profit = df["Lucro_Empresa"].sum()
        total_transactions = len(df)
        avg_transaction = total_commission / total_transactions if total_transactions > 0 else 0
        active_assessors = df["AssessorReal"].nunique()

        with col1:
            st.metric("Comissão Total", format_currency(total_commission))
        with col2:
            st.metric("Total Pix Assessor", format_currency(total_pix))
        with col3:
            st.metric("Lucro da Empresa", format_currency(total_profit))
        with col4:
            st.metric("Média Transação", format_currency(avg_transaction))
        with col5:
            st.metric("Assessores Ativos", active_assessors)
        return None # Não retorna valores para o dashboard executivo

# --- PÁGINA DE UPLOAD ---
if page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload e Gestão de Dados</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Carregue seu arquivo Taurus Excel",
            type=["xlsx"],
            help="O arquivo deve conter uma aba chamada 'Taurus' com as colunas obrigatórias."
        )

    with col2:
        st.info("📋 **Colunas Obrigatórias:**\n- Chave\n- AssessorReal\n- Categoria\n- Comissão\n- Tributo_Retido\n- Pix_Assessor\n- Lucro_Empresa\n- Receita Bruta")

    if uploaded_file:
        df_taurus = load_data(uploaded_file)
        if df_taurus is not None:
            st.session_state["df_taurus"] = df_taurus
            st.success("✅ Dados carregados e processados com sucesso!")

            st.markdown("### 📊 Visão Geral dos Dados")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Transações", f"{len(df_taurus):,}")
            with col2:
                st.metric("Assessores Únicos", df_taurus["AssessorReal"].nunique())
            with col3:
                st.metric("Períodos de Tempo", df_taurus["Chave"].nunique())
            with col4:
                total_commission = df_taurus["Comissão"].sum()
                st.metric("Comissão Total", format_currency(total_commission))

            st.markdown("### 🔍 Avaliação da Qualidade dos Dados")
            col1, col2 = st.columns(2)

            with col1:
                missing_data = df_taurus.isnull().sum()
                if missing_data.sum() > 0:
                    st.warning("⚠️ Dados Ausentes Encontrados:")
                    st.dataframe(missing_data[missing_data > 0])
                else:
                    st.success("✅ Nenhuma dado ausente detectado.")

            with col2:
                if 'Data Receita' in df_taurus.columns:
                    date_range = f"{df_taurus['Data Receita'].min().strftime('%d/%m/%Y')} a {df_taurus['Data Receita'].max().strftime('%d/%m/%Y')}"
                    st.info(f"📅 **Intervalo de Datas:** {date_range}")

                top_categories = df_taurus['Categoria'].value_counts().head(3)
                st.info("🏆 **Principais Categorias:**\n" + "\n".join([f"• {cat}: {count}" for cat, count in top_categories.items()]))

            st.markdown("### 👀 Pré-visualização de Amostra dos Dados")
            display_cols = ['Chave', 'AssessorReal', 'Categoria', 'Comissão', 'Pix_Assessor', 'Lucro_Empresa', 'Receita Bruta']
            # Filtra apenas as colunas que realmente existem no dataframe antes de exibir
            display_cols = [col for col in display_cols if col in df_taurus.columns]
            sample_data = df_taurus[display_cols].head(10)
            st.dataframe(sample_data, use_container_width=True)

# --- DASHBOARD EXECUTIVO ---
elif page == "📊 Executive Dashboard":
    st.markdown('<h1 class="main-header">📊 Dashboard Executivo</h1>', unsafe_allow_html=True)

    if st.session_state["df_taurus"] is None:
        st.warning("🚨 Por favor, carregue o arquivo Excel na página 'Upload' primeiro.")
        st.stop()

    df = st.session_state["df_taurus"]

    col1 = st.columns(1)[0]
    with col1:
        chave_list = sorted(df["Chave"].dropna().unique(), key=parse_chave_to_date)
        selected_chaves = st.multiselect(
            "🕐 Selecione os Períodos de Tempo",
            chave_list,
            default=chave_list[-6:] if len(chave_list) >= 6 else chave_list,
            help="Escolha os meses/anos para analisar."
        )

    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]

        st.markdown("### 🎯 Indicadores Chave de Performance (KPIs)")
        display_kpis(df_filtered)

        st.markdown("### 📈 Análise de Tendências e Distribuição")
        col1, col2 = st.columns(2)

        with col1:
            monthly_revenue = df_filtered.groupby('Chave')['Comissão'].sum().reset_index()
            monthly_revenue['Chave_Date'] = monthly_revenue['Chave'].apply(parse_chave_to_date)
            monthly_revenue = monthly_revenue.sort_values('Chave_Date')

            fig_revenue = px.line(
                monthly_revenue,
                x='Chave',
                y='Comissão',
                title='📈 Evolução da Comissão Total',
                markers=True,
                labels={'Comissão': 'Comissão (R\$)', 'Chave': 'Período'},
                hover_name='Chave'
            )
            fig_revenue.update_layout(hovermode="x unified")
            st.plotly_chart(fig_revenue, use_container_width=True)

        with col2:
            top_assessors = df_filtered.groupby('AssessorReal')['Comissão'].sum().nlargest(10).reset_index()
            fig_assessors = px.bar(
                top_assessors,
                x='Comissão',
                y='AssessorReal',
                orientation='h',
                title='🏆 Top 10 Assessores por Comissão',
                labels={'Comissão': 'Comissão (R\$)', 'AssessorReal': 'Assessor'},
                hover_name='AssessorReal'
            )
            fig_assessors.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_assessors, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            category_dist = df_filtered.groupby('Categoria')['Comissão'].sum().reset_index()
            fig_pie = px.pie(
                category_dist,
                values='Comissão',
                names='Categoria',
                title='🎯 Distribuição da Comissão por Categoria',
                hole=0.3,
                hover_name='Categoria'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            profit_margin = df_filtered.groupby('Chave').agg(
                Comissão_sum=('Comissão', 'sum'),
                Lucro_Empresa_sum=('Lucro_Empresa', 'sum')
            ).reset_index()
            profit_margin['Margem_Lucro_Percentual'] = (profit_margin['Lucro_Empresa_sum'] / profit_margin['Comissão_sum']) * 100

            fig_margin = px.bar(
                profit_margin,
                x='Chave',
                y='Margem_Lucro_Percentual',
                title='📊 Margem de Lucro por Período (%)',
                color='Margem_Lucro_Percentual',
                color_continuous_scale=px.colors.sequential.RdYlGn,
                labels={'Margem_Lucro_Percentual': 'Margem de Lucro (%)', 'Chave': 'Período'},
                hover_name='Chave'
            )
            fig_margin.update_layout(yaxis_range=[0, profit_margin['Margem_Lucro_Percentual'].max() * 1.1])
            st.plotly_chart(fig_margin, use_container_width=True)
    else:
        st.info("ℹ️ Por favor, selecione pelo menos um período de tempo para exibir o dashboard.")

# --- MACRO VIEW PAGE ---
elif page == "🌍 Macro View":
    st.markdown('<h1 class="main-header">🌍 Visão Macro - Performance dos Assessores</h1>', unsafe_allow_html=True)

    if st.session_state["df_taurus"] is None:
        st.warning("🚨 Por favor, carregue o arquivo Excel na página 'Upload' primeiro.")
        st.stop()

    df = st.session_state["df_taurus"]

    col1, col2 = st.columns(2)
    with col1:
        chave_list = sorted(df["Chave"].dropna().unique(), key=parse_chave_to_date)
        selected_chaves = st.multiselect(
            "🕐 Selecione os Períodos de Tempo",
            chave_list,
            default=chave_list,
            help="Escolha os meses/anos para analisar."
        )

    with col2:
        min_revenue = st.number_input("💰 Filtro de Comissão Mínima (R\$)", min_value=0.0, value=0.0, step=1000.0,
                                     help="Exibe apenas assessores com comissão total acima deste valor.")

    if selected_chaves:
        df_filtered = df[df["Chave"].isin(selected_chaves)]

        # Sumarização por assessor
        financial_cols = ["Receita Bruta", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
        summary_df = df_filtered.groupby("AssessorReal")[financial_cols].sum().reset_index()

        # Adicionar métricas calculadas
        summary_df['Total_Transacoes'] = df_filtered.groupby("AssessorReal").size().values
        summary_df['Media_Transacao'] = summary_df['Comissão'] / summary_df['Total_Transacoes']
        summary_df['Margem_Lucro_Percentual'] = (summary_df['Lucro_Empresa'] / summary_df['Comissão']) * 100
        summary_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Lidar com divisões por zero
        summary_df.fillna(0, inplace=True) # Substituir NaNs por 0, se aplicável

        # Filtrar por comissão mínima
        summary_df = summary_df[summary_df['Comissão'] >= min_revenue]
        summary_df = summary_df.sort_values("Comissão", ascending=False)

        st.markdown("### 📊 Métricas Agregadas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessores", len(summary_df))
        with col2:
            st.metric("Comissão Total", format_currency(summary_df['Comissão'].sum()))
        with col3:
            st.metric("Comissão Média/Assessor", format_currency(summary_df['Comissão'].mean()))

        st.markdown("### 📋 Resumo da Performance dos Assessores")

        # Formatar o DataFrame para exibição
        display_df = summary_df.copy()
        for col in financial_cols:
            display_df[col] = display_df[col].apply(format_currency)
        display_df['Media_Transacao'] = display_df['Media_Transacao'].apply(format_currency)
        display_df['Margem_Lucro_Percentual'] = display_df['Margem_Lucro_Percentual'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown("### 📈 Visualizações Comparativas")
        col1, col2 = st.columns(2)

        with col1:
            top_10 = summary_df.head(10)
            fig_top = px.bar(
                top_10,
                x='Comissão',
                y='AssessorReal',
                orientation='h',
                title='🏆 Top 10 Assessores por Comissão',
                color='Margem_Lucro_Percentual',
                color_continuous_scale=px.colors.sequential.RdYlGn,
                labels={'Comissão': 'Comissão (R\$)', 'AssessorReal': 'Assessor', 'Margem_Lucro_Percentual': 'Margem de Lucro (%)'},
                hover_name='AssessorReal'
            )
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                summary_df,
                x='Comissão',
                y='Margem_Lucro_Percentual',
                size='Total_Transacoes',
                hover_data=['AssessorReal', 'Comissão', 'Margem_Lucro_Percentual', 'Total_Transacoes'],
                title='💰 Comissão vs. Margem de Lucro',
                labels={'Comissão': 'Comissão (R\$)', 'Margem_Lucro_Percentual': 'Margem de Lucro (%)', 'Total_Transacoes': 'Número de Transações'},
                size_max=60
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### 📥 Opções de Exportação")
        col1, col2 = st.columns(2)

        with col1:
            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Baixar Resumo Completo (CSV)",
                csv_data,
                f"Taurus_Macro_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                help="Exporta todos os dados sumarizados dos assessores."
            )

        with col2:
            top_performers_export = summary_df.head(20)
            csv_top = top_performers_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🏆 Baixar Top 20 Assessores (CSV)",
                csv_top,
                f"Taurus_Top_Performers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                help="Exporta apenas os 20 melhores assessores."
            )
    else:
        st.info("ℹ️ Por favor, selecione pelo menos um período de tempo para exibir a visão macro.")

# --- ASSESSOR VIEW PAGE ---
elif page == "👤 Assessor View":
    st.markdown('<h1 class="main-header">👤 Análise Individual do Assessor</h1>', unsafe_allow_html=True)

    if st.session_state["df_taurus"] is None:
        st.warning("🚨 Por favor, carregue o arquivo Excel na página 'Upload' primeiro.")
        st.stop()

    df = st.session_state["df_taurus"]

    col1, col2 = st.columns(2)
    with col1:
        chave_list = sorted(df["Chave"].dropna().unique(), key=parse_chave_to_date)
        selected_chaves = st.multiselect(
            "🕐 Selecione os Períodos de Tempo",
            chave_list,
            default=chave_list,
            help="Escolha os meses/anos para analisar o assessor."
        )

    with col2:
        assessor_list = sorted(df["AssessorReal"].dropna().unique())
        selected_assessor = st.selectbox("👤 Selecione o Assessor", assessor_list,
                                         help="Escolha um assessor para ver sua performance detalhada.")

    if selected_chaves and selected_assessor:
        df_filtered = df[
            (df["AssessorReal"] == selected_assessor) &
            (df["Chave"].isin(selected_chaves))
        ]

        if df_filtered.empty:
            st.warning("⚠️ Nenhum dado encontrado para os critérios selecionados. Tente ajustar os filtros.")
        else:
            st.markdown(f"### 📊 Performance de {selected_assessor}")

            # Chamar a função de exibição de KPIs e obter os valores para exportação
            total_revenue, total_commission, total_transactions, total_pix, total_profit, avg_transaction_assessor = display_kpis(df_filtered, is_assessor_view=True)

            # Performance ao longo do tempo
            st.markdown("### 📈 Desempenho Mensal")
            monthly_performance = df_filtered.groupby('Chave').agg(
                Comissão=('Comissão', 'sum'),
                Lucro_Empresa=('Lucro_Empresa', 'sum'),
                Transacoes=('Chave', 'count')
            ).reset_index()
            monthly_performance['Chave_Date'] = monthly_performance['Chave'].apply(parse_chave_to_date)
            monthly_performance = monthly_performance.sort_values('Chave_Date')

            fig_monthly_perf = make_subplots(specs=[[{"secondary_y": True}]])
            fig_monthly_perf.add_trace(go.Bar(x=monthly_performance['Chave'], y=monthly_performance['Comissão'], name='Comissão (R\$)'), secondary_y=False)
            fig_monthly_perf.add_trace(go.Scatter(x=monthly_performance['Chave'], y=monthly_performance['Transacoes'], name='Transações', mode='lines+markers'), secondary_y=True)
            fig_monthly_perf.update_layout(title_text=f"Evolução Mensal de {selected_assessor}", hovermode="x unified")
            fig_monthly_perf.update_xaxes(title_text="Período")
            fig_monthly_perf.update_yaxes(title_text="Comissão (R\$)", secondary_y=False)
            fig_monthly_perf.update_yaxes(title_text="Número de Transações", secondary_y=True)
            st.plotly_chart(fig_monthly_perf, use_container_width=True)

            # Detalhamento por categoria
            st.markdown("### 📋 Performance por Categoria")
            financial_cols_category = ["Receita Bruta", "Comissão", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"]
            category_summary = df_filtered.groupby("Categoria")[financial_cols_category].sum().reset_index()
            category_summary['Total_Transacoes'] = df_filtered.groupby("Categoria").size().values
            st.dataframe(category_summary.round(2), use_container_width=True)

            # Visualização da distribuição por categoria
            fig_category = px.treemap(
                category_summary,
                path=['Categoria'],
                values='Comissão',
                title=f'🎯 {selected_assessor} - Comissão por Categoria',
                color='Lucro_Empresa',
                color_continuous_scale=px.colors.sequential.RdYlGn,
                hover_data=['Receita Bruta', 'Tributo_Retido', 'Pix_Assessor', 'Total_Transacoes']
            )
            st.plotly_chart(fig_category, use_container_width=True)

            st.markdown("### 📥 Opções de Exportação")
            col1, col2, col3 = st.columns(3)

            with col1:
                csv_summary = category_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Baixar Resumo por Categoria (CSV)",
                    csv_summary,
                    f"{selected_assessor}_Category_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    help="Exporta a performance do assessor por categoria."
                )

            with col2:
                # Exportar para Excel com múltiplas abas
                detailed_cols = [
                    "Chave", "Data Receita", "Conta", "Cliente", "AssessorReal", "Categoria", "Produto",
                    "Comissão", "Receita Bruta", "Tributo_Retido", "Pix_Assessor", "Lucro_Empresa"
                ]
                available_cols = [col for col in detailed_cols if col in df_filtered.columns]

                if available_cols:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_filtered[available_cols].to_excel(writer, sheet_name='Transacoes_Detalhadas', index=False)
                        category_summary.to_excel(writer, sheet_name='Resumo_Categorias', index=False)
                        monthly_performance.to_excel(writer, sheet_name='Performance_Mensal', index=False)

                    st.download_button(
                        "📋 Baixar Relatório Completo (Excel)",
                        buffer.getvalue(),
                        f"{selected_assessor}_Complete_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Exporta transações, resumo por categoria e performance mensal."
                    )
                else:
                    st.warning("⚠️ Algumas colunas detalhadas não estão disponíveis para exportação.")

            with col3:
                # Correção: Usar o avg_transaction_assessor calculado
                performance_summary_export = pd.DataFrame({
                    'Métrica': ['Receita Bruta', 'Comissão Total', 'Total Transações', 'Média Transação', 'Pix Assessor', 'Lucro Gerado'],
                    'Valor': [total_revenue, total_commission, total_transactions, avg_transaction_assessor, total_pix, total_profit]
                })
                csv_perf = performance_summary_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🎯 Baixar Resumo de Performance (CSV)",
                    csv_perf,
                    f"{selected_assessor}_Performance_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    help="Exporta os principais KPIs do assessor."
                )

# --- PERFORMANCE ANALYTICS ---
elif page == "📈 Performance Analytics":
    st.markdown('<h1 class="main-header">📈 Análises Avançadas de Performance</h1>', unsafe_allow_html=True)

    if st.session_state["df_taurus"] is None:
        st.warning("🚨 Por favor, carregue o arquivo Excel na página 'Upload' primeiro.")
        st.stop()

    df = st.session_state["df_taurus"]

    analysis_type = st.selectbox(
        "📊 Selecione o Tipo de Análise",
        ["Análise de Tendência", "Análise Comparativa", "Análise de Categoria"] # Removi "Seasonal Analysis" e "Growth Analysis" por não haver implementação no código original e manter o foco na melhoria do que existe.
    )

    if analysis_type == "Análise de Tendência":
        st.markdown("### 📈 Tendências de Comissão e Lucro ao Longo do Tempo")

        monthly_data = df.groupby('Chave').agg(
            Comissão=('Comissão', 'sum'),
            Lucro_Empresa=('Lucro_Empresa', 'sum'),
            Pix_Assessor=('Pix_Assessor', 'sum'),
            Assessores_Ativos=('AssessorReal', 'nunique')
        ).reset_index()

        monthly_data['Chave_Date'] = monthly_data['Chave'].apply(parse_chave_to_date)
        monthly_data = monthly_data.sort_values('Chave_Date')

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Evolução da Comissão', 'Evolução do Lucro', 'Pix para Assessores', 'Número de Assessores Ativos'),
            vertical_spacing=0.15
        )

        fig.add_trace(go.Scatter(x=monthly_data['Chave'], y=monthly_data['Comissão'],
                                 mode='lines+markers', name='Comissão', line=dict(color='#1f77b4')), # Azul Taurus
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=monthly_data['Chave'], y=monthly_data['Lucro_Empresa'],
                                 mode='lines+markers', name='Lucro', line=dict(color='#2ca02c')), # Verde
                      row=1, col=2)

        fig.add_trace(go.Scatter(x=monthly_data['Chave'], y=monthly_data['Pix_Assessor'],
                                 mode='lines+markers', name='Pix', line=dict(color='#ff7f0e')), # Laranja
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=monthly_data['Chave'], y=monthly_data['Assessores_Ativos'],
                                 mode='lines+markers', name='Assessores Ativos', line=dict(color='#9467bd')), # Roxo
                      row=2, col=2)

        fig.update_layout(height=700, title_text="📊 Análise Abrangente de Tendências", showlegend=False)
        fig.update_xaxes(title_text="Período")
        fig.update_yaxes(title_text="Valor (R\$)", row=1, col=1)
        fig.update_yaxes(title_text="Valor (R\$)", row=1, col=2)
        fig.update_yaxes(title_text="Valor (R\$)", row=2, col=1)
        fig.update_yaxes(title_text="Contagem", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Análise Comparativa":
        st.markdown("### 🔍 Comparativo de Performance de Assessores")

        top_n = st.slider("Selecione o Top N Assessores para Comparação", 3, 20, 10,
                          help="Número de assessores para incluir na comparação.")

        top_assessors_names = df.groupby('AssessorReal')['Comissão'].sum().nlargest(top_n).index
        df_top = df[df['AssessorReal'].isin(top_assessors_names)]

        comparison_data = df_top.groupby('AssessorReal').agg(
            Comissao_Total=('Comissão', 'sum'),
            Lucro_Total=('Lucro_Empresa', 'sum'),
            Pix_Total=('Pix_Assessor', 'sum'),
            Total_Transacoes=('Chave', 'count')
        ).reset_index()

        comparison_data['Media_Transacao'] = comparison_data['Comissao_Total'] / comparison_data['Total_Transacoes']
        comparison_data['Margem_Lucro_Percentual'] = (comparison_data['Lucro_Total'] / comparison_data['Comissao_Total']) * 100
        comparison_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        comparison_data.fillna(0, inplace=True)

        st.dataframe(comparison_data.sort_values('Comissao_Total', ascending=False).round(2), use_container_width=True)

        # Radar chart para os 5 melhores (para não sobrecarregar o gráfico)
        st.markdown("### 🕸️ Comparação Detalhada (Top 5 Assessores)")
        fig_radar = go.Figure()

        # Seleciona apenas os 5 primeiros para o radar chart
        top_5_for_radar = comparison_data.sort_values('Comissao_Total', ascending=False).head(5)

        if not top_5_for_radar.empty:
            metrics_for_radar = ['Comissao_Total', 'Lucro_Total', 'Pix_Total', 'Media_Transacao', 'Margem_Lucro_Percentual']
            metrics_labels = ['Comissão Total', 'Lucro Total', 'Pix Total', 'Média Transação', 'Margem de Lucro (%)']

            # Normalizar métricas para o radar chart (0 a 100)
            normalized_df = pd.DataFrame()
            for metric in metrics_for_radar:
                max_val = top_5_for_radar[metric].max()
                min_val = top_5_for_radar[metric].min()
                if max_val == min_val: # Evitar divisão por zero se todos os valores forem iguais
                    normalized_df[metric] = 50 # Valor médio
                else:
                    normalized_df[metric] = ((top_5_for_radar[metric] - min_val) / (max_val - min_val)) * 100

            for i, row in top_5_for_radar.iterrows():
                assessor = row['AssessorReal']
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_df.loc[i, metrics_for_radar].values,
                    theta=metrics_labels,
                    fill='toself',
                    name=assessor,
                    hoverinfo='text',
                    text=[f"{label}: {row[original_metric]:.2f}" for label, original_metric in zip(metrics_labels, metrics_for_radar)]
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="🎯 Performance Comparativa Normalizada dos Assessores"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("ℹ️ Não há dados suficientes para gerar o gráfico de radar para os principais assessores.")

    elif analysis_type == "Análise de Categoria":
        st.markdown("### 📊 Análise Detalhada por Categoria")

        col1, col2 = st.columns(2)
        with col1:
            category_selection = st.selectbox("Selecione uma Categoria", df["Categoria"].dropna().unique(),
                                              help="Escolha uma categoria para analisar sua performance ao longo do tempo.")

        if category_selection:
            df_category = df[df["Categoria"] == category_selection]

            monthly_category_data = df_category.groupby('Chave').agg(
                Comissão=('Comissão', 'sum'),
                Lucro_Empresa=('Lucro_Empresa', 'sum'),
                Total_Transacoes=('Chave', 'count')
            ).reset_index()
            monthly_category_data['Chave_Date'] = monthly_category_data['Chave'].apply(parse_chave_to_date)
            monthly_category_data = monthly_category_data.sort_values('Chave_Date')

            if not monthly_category_data.empty:
                fig_category_trend = make_subplots(specs=[[{"secondary_y": True}]])
                fig_category_trend.add_trace(go.Bar(x=monthly_category_data['Chave'], y=monthly_category_data['Comissão'], name='Comissão (R\$)', marker_color='#1f77b4'), secondary_y=False)
                fig_category_trend.add_trace(go.Scatter(x=monthly_category_data['Chave'], y=monthly_category_data['Lucro_Empresa'], name='Lucro (R\$)', mode='lines+markers', line=dict(color='#2ca02c')), secondary_y=False)
                fig_category_trend.add_trace(go.Scatter(x=monthly_category_data['Chave'], y=monthly_category_data['Total_Transacoes'], name='Transações', mode='lines', line=dict(color='#ff7f0e', dash='dot')), secondary_y=True)

                fig_category_trend.update_layout(title_text=f"Tendência da Categoria: {category_selection}", hovermode="x unified")
                fig_category_trend.update_xaxes(title_text="Período")
                fig_category_trend.update_yaxes(title_text="Valor (R\$)", secondary_y=False)
                fig_category_trend.update_yaxes(title_text="Número de Transações", secondary_y=True)
                st.plotly_chart(fig_category_trend, use_container_width=True)
            else:
                st.info(f"ℹ️ Não há dados para a categoria '{category_selection}' nos períodos carregados.")

            st.markdown("### 🏆 Top Assessores na Categoria")
            top_assessors_in_category = df_category.groupby('AssessorReal')['Comissão'].sum().nlargest(10).reset_index()
            if not top_assessors_in_category.empty:
                fig_top_cat_assessors = px.bar(
                    top_assessors_in_category,
                    x='Comissão',
                    y='AssessorReal',
                    orientation='h',
                    title=f'Top 10 Assessores em {category_selection}',
                    labels={'Comissão': 'Comissão (R\$)', 'AssessorReal': 'Assessor'}
                )
                fig_top_cat_assessors.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top_cat_assessors, use_container_width=True)
            else:
                st.info(f"ℹ️ Não há assessores com comissão para a categoria '{category_selection}'.")

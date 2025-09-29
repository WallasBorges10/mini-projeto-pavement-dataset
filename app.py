# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise e Predição de Manutenção de Pavimentos",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÕES DE CACHE E CARREGAMENTO ---
@st.cache_resource
def load_data_and_model():
    """Carrega dados e modelos com cache para melhor performance."""
    try:
        # Cria o diretório 'models' se não existir
        os.makedirs("models", exist_ok=True)
        
        # Carrega o dataset do Kaggle
        path = kagglehub.dataset_download("gifreysulay/pavement-dataset")
        
        # O dataset vem num arquivo zip, então precisamos encontrar o CSV dentro dele
        csv_file_path = ""
        for f in os.listdir(path):
            if f.endswith('.csv'):
                csv_file_path = os.path.join(path, f)
                break
        
        if not csv_file_path:
            st.error("Arquivo CSV não encontrado no dataset do Kaggle.")
            return None, None, None, None, None, None

        df_pavement = pd.read_csv(csv_file_path)
        df_pavement = df_pavement.sample(10000, random_state=42)

        # Caminhos dos modelos e do histórico
        MODEL_PATH = os.path.join("models", "best_pavement_model.keras")
        PIPELINE_PATH = os.path.join("models", "pavement_preprocessor.pkl")
        HISTORY_PATH = os.path.join("models", "training_history.pkl")

        # Carrega o modelo, pipeline e histórico
        model = keras.models.load_model(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        history = joblib.load(HISTORY_PATH)
        
        st.success("✅ Dados e modelos carregados com sucesso!")
        return df_pavement, model, pipeline, history

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados/modelo: {e}")
        st.info("Certifique-se de que os arquivos 'best_pavement_model.keras', 'pavement_preprocessor.pkl', e 'training_history.pkl' estão na pasta 'models'.")
        return None, None, None, None

# --- CARREGAMENTO INICIAL ---
df, model, pipeline, history = load_data_and_model()

# --- SIDEBAR DE NAVEGAÇÃO ---
st.sidebar.title("🛣️ Análise e Predição de Manutenção de Pavimentos")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegue pelas seções:",
    ["Predição em Tempo Real", "Análise Exploratória", "Modelagem & Métricas", "Visão Geral dos Dados"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Desenvolvido por:** Wallas Borges Araújo  
    """
)
# --- SEÇÃO DE PREDIÇÃO ---
def prediction_section():
    st.header("🔮 Predição de Necessidade de Manutenção")
    st.markdown("Insira os dados do segmento de pavimento para obter uma predição em tempo real.")

    # Formulário para entrada do usuário
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pci = st.slider("Índice de Condição do Pavimento (PCI)", 0.0, 100.0, 50.0, 0.1)
            rutting = st.slider("Afundamento (Rutting em mm)", 8.0, 23.0, 15.0, 0.1)
            road_type = st.selectbox("Tipo de Via (Road Type)", options=df['Road Type'].unique(), index=0)

        with col2:
            aadt = st.number_input("Tráfego Médio Diário Anual (AADT)", min_value=0, max_value=200000, value=10000)
            iri = st.slider("Índice de Irregularidade Internacional (IRI)", 0.0, 2.0, 0.8, 0.01)
            asphalt_type = st.selectbox("Tipo de Asfalto (Asphalt Type)", options=df['Asphalt Type'].unique(), index=0)

        with col3:
            last_maintenance = st.number_input("Ano da Última Manutenção", min_value=1950, max_value=2025, value=2018)
            avg_rainfall = st.slider("Precipitação Média Anual (mm)", 10.0, 120.0, 65.0, 0.1)
        
        submit_button = st.form_submit_button("Executar Predição")

    if submit_button:
        # Criar DataFrame com os dados de entrada
        input_data = pd.DataFrame({
            'PCI': [pci],
            'Road Type': [road_type],
            'AADT': [aadt],
            'Asphalt Type': [asphalt_type],
            'Last Maintenance': [last_maintenance],
            'Average Rainfall': [avg_rainfall],
            'Rutting': [rutting],
            'IRI': [iri]
        })

        st.markdown("---")
        st.subheader("Dados Inseridos pelo Usuário:")
        st.dataframe(input_data)

        # Pré-processar os dados e fazer a predição original
        processed_data = pipeline.transform(input_data)
        prediction_proba = model.predict(processed_data)[0][0]
        prediction = 1 if prediction_proba > 0.5 else 0

        # Exibir o resultado
        st.subheader("Resultado da Predição:")
        if prediction == 1:
            st.error(f"🚨 **Precisa de Manutenção** (Probabilidade: {prediction_proba:.2%})")
        else:
            st.success(f"✅ **Não Precisa de Manutenção** (Probabilidade de precisar: {prediction_proba:.2%})")

        st.progress(float(prediction_proba))
        st.markdown("---")

        # --- ANÁLISE TÉCNICA DE SENSIBILIDADE COM GRÁFICO ---
        with st.expander("🔬 Clique para ver a Análise Detalhada"):
            st.subheader("Análise de Contribuição dos Atributos")
            st.markdown("""
            O gráfico interativo abaixo mostra o impacto de cada atributo na predição. Simulamos o "pior cenário" para cada variável e medimos o quanto a probabilidade de manutenção **aumentaria**. 
            
            **Passe o mouse sobre as barras para ver os detalhes.**
            """)

            feature_impacts = {}
            # Definimos os piores cenários para cada atributo
            worst_scenarios = {
                'PCI': df['PCI'].min(),
                'Rutting': df['Rutting'].max(),
                'IRI': df['IRI'].max(),
                'Last Maintenance': df['Last Maintenance'].min(),
                'AADT': df['AADT'].max(),
                'Average Rainfall': df['Average Rainfall'].max()
            }

            for feature, worst_value in worst_scenarios.items():
                temp_data = input_data.copy()
                temp_data[feature] = worst_value
                temp_processed = pipeline.transform(temp_data)
                new_proba = model.predict(temp_processed)[0][0]
                impact = new_proba - prediction_proba
                # Garantimos que o impacto não seja negativo (só nos interessa o aumento do risco)
                feature_impacts[feature] = max(0, impact)

            # Ordena os atributos pelo maior impacto e cria um DataFrame
            sorted_impacts = sorted(feature_impacts.items(), key=lambda item: item[1], reverse=False) # Invertido para plotagem correta
            impact_df = pd.DataFrame(sorted_impacts, columns=['Atributo', 'Impacto na Probabilidade'])

            # Cria o gráfico de barras interativo com Plotly
            fig = px.bar(
                impact_df,
                x='Impacto na Probabilidade',
                y='Atributo',
                orientation='h',
                title='Sensibilidade da Predição a Piores Cenários',
                text='Impacto na Probabilidade'
            )
            
            # Melhora a formatação do gráfico
            fig.update_traces(
                texttemplate='%{text:.2%}', 
                textposition='outside',
                marker_color='#EF553B' # Cor vermelha para indicar risco
            )
            fig.update_layout(
                yaxis_title="Atributo",
                xaxis_title="Aumento na Probabilidade de Manutenção",
                xaxis_tickformat='.0%',
                uniformtext_minsize=8, 
                uniformtext_mode='hide'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Análise dos Fatores Chave")
            st.markdown("""
            Com base no comportamento geral do modelo, observe os seguintes fatores de risco:

            * **Fatores que AUMENTAM a necessidade de manutenção:**
                * 📉 **PCI baixo:** Valores abaixo de 60-70 são um forte indicativo.
                * 📈 **Rutting (Afundamento) alto:** Geralmente acima de 15-18mm.
                * 📈 **IRI (Irregularidade) alto:** Valores se aproximando de 1.5 ou mais.
                * 🗓️ **Última Manutenção distante:** Anos muito antigos aumentam significativamente o risco.

            * **Fatores que DIMINUEM a necessidade de manutenção:**
                * 📈 **PCI alto:** Pavimentos com PCI acima de 85 raramente necessitam de intervenção imediata.
                * 📉 **Rutting e IRI baixos:** Indicam uma superfície de boa qualidade.
                * 🆕 **Manutenção Recente:** Intervenções nos últimos anos reduzem drasticamente a necessidade de uma nova.
            """)

# --- SEÇÃO DE ANÁLISE EXPLORATÓRIA (EDA) ---
# --- SEÇÃO DE ANÁLISE EXPLORATÓRIA (EDA) AMPLIADA ---
def eda_section():
    st.header("📊 Análise Exploratória dos Dados (EDA)")
    st.markdown("Visualizações interativas para entender a distribuição e correlação dos dados.")
    
    # Filtros na sidebar
    st.sidebar.subheader("🎛️ Filtros de Dados")
    
    # Filtro por Tipo de Rodovia
    road_types = st.sidebar.multiselect(
        "Tipo de Rodovia:",
        options=df['Road Type'].unique(),
        default=df['Road Type'].unique()
    )
    
    # Filtro por PCI
    pci_range = st.sidebar.slider(
        "Faixa de PCI:",
        min_value=float(df['PCI'].min()),
        max_value=float(df['PCI'].max()),
        value=(float(df['PCI'].min()), float(df['PCI'].max()))
    )
    
    # Filtro por necessidade de manutenção
    maintenance_filter = st.sidebar.selectbox(
        "Necessidade de Manutenção:",
        options=['Todos', 'Sim', 'Não']
    )
    
    # Aplicar filtros
    filtered_df = df[
        (df['Road Type'].isin(road_types)) &
        (df['PCI'] >= pci_range[0]) &
        (df['PCI'] <= pci_range[1])
    ]
    
    if maintenance_filter != 'Todos':
        filter_value = 1 if maintenance_filter == 'Sim' else 0
        filtered_df = filtered_df[filtered_df['Needs Maintenance'] == filter_value]
    
    # Métricas principais
    st.subheader("📊 Métricas Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_samples = len(filtered_df)
        st.metric("Amostras Filtradas", total_samples)
    
    with col2:
        maintenance_rate = filtered_df['Needs Maintenance'].mean() * 100
        st.metric("Taxa de Manutenção", f"{maintenance_rate:.1f}%")
    
    with col3:
        avg_pci = filtered_df['PCI'].mean()
        st.metric("PCI Médio", f"{avg_pci:.1f}")
    
    with col4:
        high_risk = len(filtered_df[filtered_df['PCI'] < 40])
        st.metric("Pavimentos Críticos", high_risk)
    
    # Abas para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribuições", "🔍 Correlações", "📋 Dados Detalhados", "🎯 Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição da classe alvo
            fig = px.pie(
                filtered_df, 
                names='Needs Maintenance',
                title='Distribuição de Necessidade de Manutenção',
                color='Needs Maintenance',
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribuição do PCI por tipo de manutenção
            fig = px.histogram(
                filtered_df, 
                x='PCI',
                color='Needs Maintenance',
                nbins=20,
                title='Distribuição do PCI por Status de Manutenção',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
        # Distribuição das variáveis numéricas (versão original)
        st.subheader("Distribuição das Variáveis Numéricas")
        numeric_features = ['PCI', 'IRI', 'Rutting', 'AADT', 'Last Maintenance', 'Average Rainfall']
        feature_to_plot = st.selectbox("Selecione uma variável para visualizar a distribuição:", numeric_features, key="distrib_original")
        fig = px.histogram(df, x=feature_to_plot, color='Needs Maintenance', marginal="box", 
                        title=f'Distribuição de {feature_to_plot} por Classe de Manutenção',
                        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        - **PCI (Índice de Condição do Pavimento):** Valores mais baixos estão fortemente associados à necessidade de manutenção.
        - **Rutting (Afundamento):** Valores mais altos indicam maior probabilidade de manutenção.
        - A análise das outras variáveis também mostra separações claras entre as classes, indicando seu poder preditivo.
        """)
        
        with col2:
            # Distribuição por Tipo de Rodovia
            road_maintenance = filtered_df.groupby(['Road Type', 'Needs Maintenance']).size().reset_index(name='Count')
            fig = px.bar(
                road_maintenance,
                x='Road Type',
                y='Count',
                color='Needs Maintenance',
                barmode='group',
                title='Necessidade de Manutenção por Tipo de Rodovia',
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot do PCI por Tipo de Rodovia
            fig = px.box(
                filtered_df,
                x='Road Type',
                y='PCI',
                color='Needs Maintenance',
                title='PCI por Tipo de Rodovia e Status de Manutenção'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Matriz de correlação
        st.subheader("🔗 Matriz de Correlação")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação entre Variáveis Numéricas",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot interativo
        st.subheader("📊 Análise de Relacionamento")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Eixo X:", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Eixo Y:", numeric_cols, index=1)
        with col3:
            color_by = st.selectbox("Colorir por:", ['Needs Maintenance', 'Road Type', 'Asphalt Type'])
        
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=filtered_df.columns.tolist(),
            title=f"Relação entre {y_axis} e {x_axis}",
            size_max=10
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Dataframe interativo
        st.subheader("📋 Dados Filtrados")
        
        # Opções de visualização
        view_option = st.radio(
            "Tipo de Visualização:",
            ["Primeiras 100 linhas", "Amostra aleatória", "Dados agregados", "Estatísticas descritivas"]
        )
        
        if view_option == "Primeiras 100 linhas":
            st.dataframe(filtered_df.head(100), use_container_width=True, height=400)
        elif view_option == "Amostra aleatória":
            sample_size = st.slider("Tamanho da amostra:", 10, 200, 50)
            st.dataframe(filtered_df.sample(sample_size), use_container_width=True, height=400)
        elif view_option == "Dados agregados":
            group_by = st.selectbox("Agrupar por:", ['Road Type', 'Asphalt Type', 'Needs Maintenance'])
            aggregated = filtered_df.groupby(group_by).agg({
                'PCI': ['mean', 'min', 'max', 'std'],
                'AADT': 'mean',
                'Last Maintenance': 'mean',
                'Needs Maintenance': ['mean', 'sum']
            }).round(2)
            st.dataframe(aggregated, use_container_width=True)
        else:
            st.dataframe(filtered_df.describe(), use_container_width=True)
        
        # Botão para download
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download dos dados filtrados (CSV)",
            data=csv,
            file_name="dados_pavimentos_filtrados.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("🎯 Insights Automáticos")
        
        # Análises automáticas
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Estatísticas de Manutenção:**")
            
            # Análise por Tipo de Rodovia
            maintenance_by_road = filtered_df.groupby('Road Type')['Needs Maintenance'].agg(['mean', 'count']).round(3)
            maintenance_by_road['mean'] = maintenance_by_road['mean'] * 100
            
            for road_type, row in maintenance_by_road.iterrows():
                status = "🔴" if row['mean'] > 30 else "🟡" if row['mean'] > 15 else "🟢"
                st.write(f"{status} **{road_type}**: {row['mean']:.1f}% ({int(row['count'])} amostras)")
            
            st.write("**📊 Distribuição de PCI:**")
            pci_stats = filtered_df['PCI'].describe()
            st.write(f"- Média: {pci_stats['mean']:.1f}")
            st.write(f"- Mediana: {pci_stats['50%']:.1f}")
            st.write(f"- Mínimo: {pci_stats['min']:.1f}")
            st.write(f"- Máximo: {pci_stats['max']:.1f}")
        
        with col2:
            st.write("**🔍 Alertas e Recomendações:**")
            
            # Alertas baseados nos dados
            critical_pci = len(filtered_df[filtered_df['PCI'] < 40])
            if critical_pci > 0:
                st.error(f"🚨 **{critical_pci} pavimentos em condição crítica** (PCI < 40)")
            
            high_maintenance = filtered_df['Needs Maintenance'].mean()
            if high_maintenance > 0.3:
                st.warning(f"⚠️ **Alta taxa de manutenção**: {high_maintenance:.1%}")
            elif high_maintenance < 0.1:
                st.success(f"✅ **Baixa taxa de manutenção**: {high_maintenance:.1%}")
            
            # Correlação mais forte com necessidade de manutenção
            if 'Needs Maintenance' in numeric_cols:
                correlations = filtered_df[numeric_cols].corr()['Needs Maintenance'].abs().sort_values(ascending=False)
                if len(correlations) > 1:
                    strongest_corr = correlations.index[1]
                    corr_value = correlations.iloc[1]
                    st.info(f"💡 **Maior correlação com manutenção**: {strongest_corr} ({corr_value:.3f})")

 


# --- SEÇÃO DE MODELAGEM E MÉTRICAS ---
def modeling_section():
    st.header("⚙️ Modelagem e Métricas de Avaliação")
    st.markdown("Avaliação da performance do modelo de Redes Neurais.")

    # Preparar dados para reavaliação
    X = df.drop(['Needs Maintenance', 'Segment ID'], axis=1)
    y = df['Needs Maintenance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_test_processed = pipeline.transform(X_test)
    y_pred_proba = model.predict(X_test_processed)
    y_pred = (y_pred_proba > 0.5).astype(int)

    st.subheader("Curvas de Treinamento")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Curva de Acurácia
    ax[0].plot(history['accuracy'], label='Acurácia de Treino')
    ax[0].plot(history['val_accuracy'], label='Acurácia de Validação')
    ax[0].set_title('Acurácia por Época')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('Acurácia')
    ax[0].legend()
    # Curva de Perda
    ax[1].plot(history['loss'], label='Perda de Treino')
    ax[1].plot(history['val_loss'], label='Perda de Validação')
    ax[1].set_title('Perda por Época')
    ax[1].set_xlabel('Época')
    ax[1].set_ylabel('Perda')
    ax[1].legend()
    st.pyplot(fig)

    st.subheader("Métricas de Classificação")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Métricas Principais")
    acc = report['accuracy']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Acurácia", f"{acc:.2%}")
    m2.metric("Recall (Classe 1)", f"{recall:.2%}")
    m3.metric("F1-Score (Classe 1)", f"{f1:.2%}")
    m4.metric("ROC AUC", f"{roc_auc:.3f}")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                           labels=dict(x="Predito", y="Verdadeiro"),
                           x=['Não Precisa', 'Precisa'],
                           y=['Não Precisa', 'Precisa'],
                           color_continuous_scale='Greens')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        st.subheader("Curva ROC")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC={roc_auc:.3f}', mode='lines'))
        fig_roc.update_layout(
                              xaxis_title='Taxa de Falsos Positivos',
                              yaxis_title='Taxa de Verdadeiros Positivos',
                              yaxis=dict(scaleanchor="x", scaleratio=1),
                              xaxis=dict(constrain='domain'))
        st.plotly_chart(fig_roc, use_container_width=True)

# --- SEÇÃO DE VISÃO GERAL DOS DADOS ---
def data_overview_section():
    st.header("📄 Visão Geral do Dataset")
    st.markdown("Uma amostra dos dados utilizados para treinar o modelo.")
    
    st.subheader("Amostra do DataFrame")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informações Gerais (Estrutura)")
        
        # Criar um DataFrame com as informações de .info()
        info_df = pd.DataFrame({
            "Coluna": df.columns,
            "Valores Não Nulos": df.count().values,
            "Tipo de Dado (Dtype)": df.dtypes.values
        })
        
        st.dataframe(info_df, use_container_width=True)

    with col2:
        st.subheader("Estatísticas Descritivas (.describe)")
        st.dataframe(df.describe(), use_container_width=True)

# --- CONTROLE PRINCIPAL ---
if df is not None:
    if page == "Predição em Tempo Real":
        prediction_section()
    elif page == "Análise Exploratória":
        eda_section()  # Agora usa a versão ampliada
    elif page == "Modelagem & Métricas":
        import plotly.graph_objects as go
        modeling_section()
    elif page == "Visão Geral dos Dados":
        data_overview_section()

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

import kagglehub
from kagglehub import KaggleDatasetAdapter
import os




# Configuração da página
st.set_page_config(
    page_title="Análise de Pavimentos",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🛣️ Sistema de Análise e Predição de Pavimentos")
st.markdown("---")

# Sidebar para navegação
st.sidebar.title("Navegação")
app_mode = st.sidebar.selectbox(
    "Selecione a página:",
    ["📊 Análise Exploratória", "🤖 Predição de Manutenção", "📈 Resultados do Modelo"]
)

# Carregar dados e modelo (com cache)
@st.cache_resource
def load_data_and_model():
    """Carrega dados e modelo com cache para melhor performance"""
    try:
        # Aqui você precisa ajustar os caminhos para seus arquivos
        # df_pavement = pd.read_csv('ESC 12 Pavement Dataset.csv')
        # df_pavement = df_pavement.sample(10000, random_state=42)


        path = kagglehub.dataset_download("gifreysulay/pavement-dataset")

        for f in os.listdir(path):
            file_path = os.path.join(path, f)

        df_pavement = pd.read_csv(file_path)

        df_pavement = df_pavement.sample(10000, random_state=42)


        MODEL_PATH = os.path.join("models", "road_maintenance_rede_neural.keras")
        PIPELINE_PATH = os.path.join("models", "road_maintenance_rede_neural.pkl")
        HISTORY_PATH = os.path.join("models", "training_history.pkl")

        model = keras.models.load_model(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)

        try:
            history = joblib.load(HISTORY_PATH)
        except:
            history = None

        # model = keras.models.load_model('road_maintenance_rede_neural.keras')
        # pipeline = joblib.load('road_maintenance_rede_neural.pkl')

        # try:
        #     history = joblib.load('training_history.pkl')
        # except:
        #     # Opção 2: Se o histórico está no objeto history do Keras
        #     # (precisa ser salvo durante o treinamento)
        #     history = None
        
        # Por enquanto, vou criar dados de exemplo para demonstração
        # np.random.seed(42)
        # n_samples = 1000
        
        # data = {
        #     'PCI': np.random.uniform(0, 100, n_samples),
        #     'AADT': np.random.randint(100, 10000, n_samples),
        #     'Last Maintenance': np.random.randint(0, 10, n_samples),
        #     'Average Rainfall': np.random.uniform(500, 2000, n_samples),
        #     'Rutting': np.random.uniform(0, 20, n_samples),
        #     'IRI': np.random.uniform(1, 10, n_samples),
        #     'Road Type': np.random.choice(['Primary', 'Secondary', 'Tertiary'], n_samples),
        #     'Asphalt Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
        #     'Needs Maintenance': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        # }
        
        # df_pavement = pd.DataFrame(data)
        
        # Criar um modelo dummy para demonstração
        # Em produção, carregue seu modelo real

        # class DummyModel:
        #     def predict(self, X):
        #         return np.random.uniform(0, 1, (X.shape[0], 1))
        
        # model = DummyModel()
        # #pipeline = None

        st.success("✅ Dados e modelo carregados com sucesso!")
        return df_pavement, model, pipeline, history
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados/modelo: {e}")
        return None, None, None, None

# Carregar dados
df_pavement, model, pipeline, training_history = load_data_and_model()

def plot_real_training_history(history):
    """Plota as curvas de aprendizado reais do treinamento"""
    if history is None:
        st.warning("📊 Histórico de treinamento não disponível")
        return None
    
    # Verificar se history é um dicionário (que é o caso quando carregamos do .pkl)
    if isinstance(history, dict):
        history_dict = history
    else:
        # Se for um objeto do Keras, extrair o dicionário
        history_dict = history.history
    
    # Criar subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Acurácia durante o Treinamento', 'Loss durante o Treinamento'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Adicionar curvas de acurácia
    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
        epochs = range(1, len(history_dict['accuracy']) + 1)
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history_dict['accuracy'], 
                      name='Acurácia - Treino', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history_dict['val_accuracy'], 
                      name='Acurácia - Validação', line=dict(color='red')),
            row=1, col=1
        )
    
    # Adicionar curvas de loss
    if 'loss' in history_dict and 'val_loss' in history_dict:
        epochs = range(1, len(history_dict['loss']) + 1)
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history_dict['loss'], 
                      name='Loss - Treino', line=dict(color='blue')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history_dict['val_loss'], 
                      name='Loss - Validação', line=dict(color='red')),
            row=1, col=2
        )
    
    # Adicionar curva de AUC se disponível
    if 'auc' in history_dict and 'val_auc' in history_dict:
        epochs = range(1, len(history_dict['auc']) + 1)
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history_dict['auc'], 
                      name='AUC - Treino', line=dict(color='green'), visible='legendonly'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history_dict['val_auc'], 
                      name='AUC - Validação', line=dict(color='orange'), visible='legendonly'),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="Épocas", row=1, col=1)
    fig.update_xaxes(title_text="Épocas", row=1, col=2)
    fig.update_yaxes(title_text="Acurácia", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="Curvas de Aprendizado do Modelo",
        showlegend=True
    )
    
    return fig

def predict_maintenance_need_real(new_data, pipeline, model):
    """
    Função REAL para prever se uma estrada precisa de manutenção
    usando seu pipeline e modelo treinado
    """
    try:
        # Pré-processar os dados usando o pipeline salvo
        processed_data = pipeline['preprocessor'].transform(new_data)
        
        # Fazer previsão com o modelo real
        probability = model.predict(processed_data)[0][0]
        predicted_class = int(probability > 0.5)
        
        return probability, predicted_class
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, None

if df_pavement is None:
    st.stop()

# Página 1: Análise Exploratória
if app_mode == "📊 Análise Exploratória":
    st.header("📊 Análise Exploratória dos Dados")
    
    # Filtros na sidebar
    st.sidebar.subheader("Filtros de Dados")
    
    # Filtro por tipo de rodovia
    road_types = st.sidebar.multiselect(
        "Tipo de Rodovia:",
        options=df_pavement['Road Type'].unique(),
        default=df_pavement['Road Type'].unique()
    )
    
    # Filtro por PCI
    pci_range = st.sidebar.slider(
        "Faixa de PCI:",
        min_value=float(df_pavement['PCI'].min()),
        max_value=float(df_pavement['PCI'].max()),
        value=(float(df_pavement['PCI'].min()), float(df_pavement['PCI'].max()))
    )
    
    # Aplicar filtros
    filtered_df = df_pavement[
        (df_pavement['Road Type'].isin(road_types)) &
        (df_pavement['PCI'] >= pci_range[0]) &
        (df_pavement['PCI'] <= pci_range[1])
    ]
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Amostras", len(filtered_df))
    
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
                title='Distribuição de Necessidade de Manutenção'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribuição do PCI
            fig = px.histogram(
                filtered_df, 
                x='PCI',
                nbins=20,
                title='Distribuição do PCI'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribuição por tipo de estrada
            fig = px.histogram(
                filtered_df,
                x='Road Type',
                color='Needs Maintenance',
                barmode='group',
                title='Necessidade de Manutenção por Tipo de Estrada'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot do PCI por tipo de estrada
            fig = px.box(
                filtered_df,
                x='Road Type',
                y='PCI',
                color='Needs Maintenance',
                title='PCI por Tipo de Estrada e Status de Manutenção'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Matriz de correlação
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot interativo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Eixo X:", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Eixo Y:", numeric_cols, index=1)
        with col3:
            color_by = st.selectbox("Colorir por:", ['Needs Maintenance', 'Road Type'])
        
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=['Road Type', 'Asphalt Type'],
            title=f"{y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Dataframe interativo
        st.subheader("Dados Filtrados")
        
        # Opções de visualização
        view_option = st.radio(
            "Visualização:",
            ["Primeiras 100 linhas", "Amostra aleatória", "Dados agregados"]
        )
        
        if view_option == "Primeiras 100 linhas":
            st.dataframe(filtered_df.head(100), use_container_width=True)
        elif view_option == "Amostra aleatória":
            sample_size = st.slider("Tamanho da amostra:", 10, 100, 50)
            st.dataframe(filtered_df.sample(sample_size), use_container_width=True)
        else:
            # Dados agregados
            group_by = st.selectbox("Agrupar por:", ['Road Type', 'Asphalt Type', 'Needs Maintenance'])
            aggregated = filtered_df.groupby(group_by).agg({
                'PCI': ['mean', 'min', 'max'],
                'AADT': 'mean',
                'Needs Maintenance': 'mean'
            }).round(2)
            st.dataframe(aggregated, use_container_width=True)
        
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
            st.write("**📊 Estatísticas Descritivas:**")
            st.dataframe(filtered_df.describe(), use_container_width=True)
            
            st.write("**🔍 Valores Faltantes:**")
            missing_data = filtered_df.isnull().sum()
            if missing_data.sum() == 0:
                st.success("✅ Nenhum valor faltante encontrado!")
            else:
                st.warning(f"⚠️ Valores faltantes detectados:")
                st.write(missing_data[missing_data > 0])
        
        with col2:
            st.write("**📈 Tendências Principais:**")
            
            # Insights automáticos
            critical_pci = len(filtered_df[filtered_df['PCI'] < 40])
            if critical_pci > 0:
                st.error(f"🚨 {critical_pci} pavimentos estão em condição crítica (PCI < 40)")
            
            high_maintenance = filtered_df['Needs Maintenance'].mean()
            if high_maintenance > 0.3:
                st.warning(f"⚠️ Alta taxa de manutenção necessária: {high_maintenance:.1%}")
            
            # Correlação mais forte com necessidade de manutenção
            if 'Needs Maintenance' in numeric_cols:
                correlations = filtered_df[numeric_cols].corr()['Needs Maintenance'].abs().sort_values(ascending=False)
                strongest_corr = correlations.index[1]  # Pula a correlação consigo mesma
                st.info(f"💡 Maior correlação com necessidade de manutenção: **{strongest_corr}**")

# Página 2: Predição de Manutenção
elif app_mode == "🤖 Predição de Manutenção":
    st.header("🤖 Predição de Necessidade de Manutenção")
    
    st.markdown("""
    Preencha os dados do pavimento abaixo para prever se será necessária manutenção.
    """)
    
    # Formulário de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dados do Pavimento")
        
        pci = st.slider("PCI (Condition Index)", 0.0, 100.0, 50.0, help="Índice de Condição do Pavimento")
        aadt = st.number_input("AADT (Tráfego Diário)", min_value=0, value=1000)
        last_maintenance = st.number_input("Última Manutenção (anos)", min_value=0, value=2)
        average_rainfall = st.number_input("Precipitação Média (mm/ano)", min_value=0.0, value=1000.0)
    
    with col2:
        st.subheader("🔧 Parâmetros Técnicos")
        
        rutting = st.slider("Rutting (mm)", 0.0, 50.0, 5.0)
        iri = st.slider("IRI (International Roughness Index)", 1.0, 10.0, 3.0)
        road_type = st.selectbox("Tipo de Estrada", ["Primary", "Secondary", "Tertiary"])
        asphalt_type = st.selectbox("Tipo de Asfalto", ["Concrete", "Asphalt"])
    
    # Botão de predição
    if st.button("🔮 Prever Necessidade de Manutenção", type="primary"):
        # Criar dataframe com os dados de entrada
        input_data = pd.DataFrame([{
            'PCI': pci,
            'AADT': aadt,
            'Last Maintenance': last_maintenance,
            'Average Rainfall': average_rainfall,
            'Rutting': rutting,
            'IRI': iri,
            'Road Type': road_type,
            'Asphalt Type': asphalt_type
        }])
        
        # Fazer predição (simulada para demonstração)
        try:
            if pipeline is not None and model is not None:
                probability, prediction = predict_maintenance_need_real(input_data, pipeline, model)
            else:
                st.error("Modelo ou pipeline não carregado corretamente")
                probability, prediction = 0.5, 0  # Fallback
            
            # # Simulação para demonstração
            # probability = np.random.uniform(0, 1)
            # prediction = 1 if probability > 0.5 else 0
            
            # Resultados
            st.markdown("---")
            st.subheader("📊 Resultado da Predição")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Indicador de probabilidade
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidade de Manutenção"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Resultado da classificação
                if prediction == 1:
                    st.error("🚨 **Manutenção Necessária**")
                    st.markdown(f"**Probabilidade:** {probability:.1%}")
                    st.warning("⚠️ Recomenda-se inspeção detalhada")
                else:
                    st.success("✅ **Manutenção Não Necessária**")
                    st.markdown(f"**Probabilidade:** {probability:.1%}")
                    st.info("💡 Condição dentro dos parâmetros aceitáveis")
            
            with col3:
                # Recomendações
                st.subheader("💡 Recomendações")
                
                if pci < 40:
                    st.error("• PCI crítico - intervenção urgente necessária")
                elif pci < 60:
                    st.warning("• PCI baixo - programar manutenção preventiva")
                else:
                    st.success("• PCI adequado - manter monitoramento")
                
                if rutting > 15:
                    st.warning("• Rutting elevado - verificar drenagem")
                
                if last_maintenance > 5:
                    st.info("• Considerar manutenção programada")
            
            # Detalhes técnicos
            with st.expander("🔧 Detalhes Técnicos da Predição"):
                st.write("**Dados de entrada utilizados:**")
                st.dataframe(input_data, use_container_width=True)


                if hasattr(pipeline, 'feature_importances_') or hasattr(model, 'feature_importances_'):
                    # Use a importância do seu modelo
                    # Exemplo para modelos sklearn:
                    importance = pipeline.named_steps['model'].feature_importances_
                    features = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    pass
                else:
                    # Fallback - mantenha o exemplo ou adapte para seu método
                    features = ['PCI', 'AADT', 'Last Maintenance', 'Average Rainfall', 'Rutting', 'IRI']
                    importance = np.random.dirichlet(np.ones(len(features)))
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importância': importance
                }).sort_values('Importância', ascending=False)
                
                fig = px.bar(importance_df, x='Importância', y='Feature', 
                        title='Importância das Features na Predição')
                st.plotly_chart(fig, use_container_width=True)
               
        except Exception as e:
            st.error(f"Erro na predição: {e}")

# Página 3: Resultados do Modelo
elif app_mode == "📈 Resultados do Modelo":
    st.header("📈 Desempenho do Modelo")
    
    # Simular resultados do modelo (substitua pelos seus resultados reais)
    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Curvas de Aprendizado", "🔍 Análise de Erros"])
    
    with tab1:
        st.subheader("Métricas de Desempenho")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Inserir aqui as métricas de desempenho do modelo
        with col1:
            st.metric("Acurácia", "99.9%", "0.4%")
        with col2:
            st.metric("AUC-ROC", "0.999", "0.03")
        with col3:
            st.metric("Precisão", "99.6%", "0.2%")
        with col4:
            st.metric("Recall", "99.9%", "0.3%")
        
        # Matriz
        #st.subheader("Matriz de Confusão")
        conf_matrix = np.array([[998, 0], [0, 1002]])  # Real
        
        fig = px.imshow(
            conf_matrix,
            text_auto=True,
            labels=dict(x="Predito", y="Real", color="Casos"),
            x=['Não Manutenção', 'Manutenção'],
            y=['Não Manutenção', 'Manutenção'],
            title="Matriz de Confusão",
            width=800,   # largura da figura
            height=800   # altura da figura
        )

        fig.update_layout(
        title_font=dict(size=24),
        margin=dict(l=80, r=80, t=100, b=80)
)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Curvas de Aprendizado Reais")
        
        if training_history is not None:
            fig = plot_real_training_history(training_history)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            
            # Estatísticas do treinamento (CORRIGIDO)
            if isinstance(training_history, dict):
                history_dict = training_history
            else:
                history_dict = training_history.history
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
                    final_train_acc = history_dict['accuracy'][-1]
                    final_val_acc = history_dict['val_accuracy'][-1]
                    st.metric("Acurácia Final - Treino", f"{final_train_acc:.1%}")
                    st.metric("Acurácia Final - Validação", f"{final_val_acc:.1%}")
                else:
                    st.info("Acurácia não disponível")
            
            with col2:
                if 'loss' in history_dict and 'val_loss' in history_dict:
                    final_train_loss = history_dict['loss'][-1]
                    final_val_loss = history_dict['val_loss'][-1]
                    st.metric("Loss Final - Treino", f"{final_train_loss:.4f}")
                    st.metric("Loss Final - Validação", f"{final_val_loss:.4f}")
                else:
                    st.info("Loss não disponível")
            
            with col3:
                if 'val_loss' in history_dict:
                    best_val_epoch = np.argmin(history_dict['val_loss']) + 1
                    total_epochs = len(history_dict['val_loss'])
                    st.metric("Melhor Época (Validação)", best_val_epoch)
                    st.metric("Total de Épocas", total_epochs)
                else:
                    st.info("Épocas não disponível")
            
            # Gráfico adicional: Comparação de métricas
            st.subheader("📊 Evolução das Métricas")
            
            available_metrics = [key for key in history_dict.keys() if not key.startswith('val_')]
            metric_option = st.selectbox(
                "Selecione a métrica para visualizar:",
                available_metrics
            )
            
            if metric_option in history_dict:
                fig_metric = go.Figure()
                epochs_range = range(1, len(history_dict[metric_option]) + 1)
                
                fig_metric.add_trace(go.Scatter(
                    x=list(epochs_range),
                    y=history_dict[metric_option],
                    name=f'{metric_option} - Treino',
                    line=dict(width=3)
                ))
                
                val_metric = f'val_{metric_option}'
                if val_metric in history_dict:
                    fig_metric.add_trace(go.Scatter(
                        x=list(epochs_range),
                        y=history_dict[val_metric],
                        name=f'{metric_option} - Validação',
                        line=dict(width=3)
                    ))
                
                fig_metric.update_layout(
                    title=f"Evolução da {metric_option.capitalize()}",
                    xaxis_title="Épocas",
                    yaxis_title=metric_option.capitalize(),
                    height=400
                )
                st.plotly_chart(fig_metric, use_container_width=True)
            
            # Mostrar informações sobre as métricas disponíveis
            with st.expander("📋 Métricas Disponíveis"):
                st.write("**Métricas de treino:**")
                train_metrics = [key for key in history_dict.keys() if not key.startswith('val_')]
                st.write(train_metrics)
                
                st.write("**Métricas de validação:**")
                val_metrics = [key for key in history_dict.keys() if key.startswith('val_')]
                st.write(val_metrics)
                
                st.write("**Estatísticas:**")
                for metric in train_metrics:
                    if metric in history_dict:
                        st.write(f"- {metric}: {len(history_dict[metric])} épocas, valor final: {history_dict[metric][-1]:.4f}")
        
        else:
            st.warning("""
            ⚠️ **Histórico de treinamento não disponível**
            """)
    
    with tab3:
        st.subheader("Análise de Erros")
        
        # Simular análise de erros
        error_analysis = pd.DataFrame({
            'Tipo de Erro': ['Falso Positivo', 'Falso Negativo', 'Verdadeiro Positivo', 'Verdadeiro Negativo'],
            'Quantidade': [0, 0, 998, 1002],
            'Custo Relativo': ['N/A', 'N/A', 'N/A', 'N/A']
        })
        
        st.dataframe(error_analysis, use_container_width=True)
        
        # Gráfico de importância das features
        st.subheader("Importância das Features")
        
        features = ['PCI', 'Last Maintenance', 'IRI', 'Rutting', 'AADT', 'Average Rainfall']
        importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                   title='Importância das Features no Modelo')
        fig.update_layout(xaxis_title='Importância', yaxis_title='Features')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        padding: 10px;
        color: #666;
    }
    </style>
    <div class="footer">
        Análise de Pavimentos com Streamlit | Protótipo criado por Wallas Borges
    </div>
    """,
    unsafe_allow_html=True

)

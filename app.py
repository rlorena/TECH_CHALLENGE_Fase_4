# Bibliotecas
import streamlit as st  
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from pandas import to_datetime, date_range
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import joblib
import datetime


# Configuração da página
st.set_page_config(
    page_title="Tech Challenge fase 4", layout="wide",
    initial_sidebar_state="auto", page_icon="📈")

st.image("combustivel.png", width=80)
st.subheader('Pojeção de Preço do Petróleo Brent', divider='rainbow' )

# Barra de menus
tab0, tab1, tab2, tab3 = st.tabs(['###### Previsões', '###### Validação','###### Fatores Históricos', '###### Comentários'],)


# Importação da base
df_anp = pd.read_csv("ipeadata[07-05-2024-09-41].csv", sep=";")


# Removendo coluna Unnamed: 2
df_anp = df_anp.drop(columns=["Unnamed: 2"])


# transformando a coluna data em datetime
df_anp['Data'] = pd.to_datetime(df_anp['Data'], format='%d/%m/%Y')
df_anp['Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366'] = df_anp[
    'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366'].str.replace(',', '.').astype('float64')


# Renomeando colunas
df_anp.rename(columns={'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'Preço petróleo Brent'
                       }, inplace=True)


# Atenção: # removendo valores nulos(constatei que são em sua grande maioria sabado e domingo, dias sem cotação)
df_anp = df_anp.dropna()


df_anp_index = df_anp.set_index('Data')  # definindo a coluna Data como índice
# df_anp_index.drop(columns=['Dia', 'Mês', 'Ano'], inplace=True) # removendo as colunas Dia, Mês e Ano
# df_anp_index.head()


# Gráfico1
# plotando o gráfico de linha com o preço do petróleo Brent
with tab2:
    #col1, col2 = st.columns(2)
    #with col1:
        fig=px.line(df_anp, x='Data', y='Preço petróleo Brent', title='Histórico de preço Brent',
            labels={'Preço petróleo Brent': 'Preço do petróleo Brent (US$)'})
        fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0="2008-01-01",
        y0=0,
        x1="2008-12-31",
        y1=1,
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
)
        fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0="2011-01-01",
        y0=0,
        x1="2011-12-31",
        y1=1,
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
)
        fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0="2016-01-01",
        y0=0,
        x1="2016-12-31",
        y1=1,
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
)
        fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0="2020-01-01",
        y0=0,
        x1="2020-12-31",
        y1=1,
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
)
        fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0="2022-01-01",
        y0=0,
        x1="2022-12-31",
        y1=1,
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
)
        fig.show()
        st.plotly_chart(fig, use_container_width=True)


############ Construindo o modelo##############
# selecionando o intervalo de 10 anos
df_anp_range_10_anos = df_anp[(df_anp['Data'] >= '2014-04-29')]
# df_anp_range_10_anos.head()


# selecionando as colunas Data e Preço petróleo Brent
df_anp_range_10_anos_prophet = df_anp_range_10_anos[[
    'Data', 'Preço petróleo Brent']]
# df_anp_range_10_anos_prophet.head()


df_anp_range_10_anos_prophet.rename(columns={
                                    "Data": "ds", "Preço petróleo Brent": "y"}, inplace=True)  # renomeando as colunas
# df_anp_range_10_anos_prophet.head()


m2 = Prophet(changepoint_prior_scale=0.08, seasonality_prior_scale=0.08,
             seasonality_mode="multiplicative", changepoint_range=0.95)
m2.fit(df_anp_range_10_anos_prophet)


# criando um dataframe com 90 dias no futuro (pode ser alterado para tentar uma previsão melhor)
future2 = m2.make_future_dataframe(periods=30)
# future2.tail()


with tab0:
    col1, col2=st.columns(2)
    with col1:
        st.write('##### Projeção de preço - 30 dias')
        forecast2 = m2.predict(future2)
        forecast3=forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].sort_values(by="ds",ascending=False)    
        forecast3['ds'] = pd.to_datetime(forecast3['ds']).dt.date
        forecast3 = forecast3.head(30)
        forecast3 = forecast3.reset_index(drop=True)
        pd.set_option('display.max_columns', 4)
        pd.set_option('display.max_rows', 30)
        st.dataframe(forecast3, width=650)    


# Gráfico2
with tab0:
    with col2:
        fig1 = m2.plot(forecast2, xlabel='Data', ylabel='Preço do petróleo Brent (US$)')
        st.write('##### Real x Previsto')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        st.pyplot(fig1,use_container_width=True)


with tab0:
    col3, col4=st.columns((1,1))
    with col4:
        st.write('##### Range previsto (30 dias)')
        fig2 = px.line(forecast2.tail(30), x='ds', y='yhat', labels={'yhat': 'Preço do petróleo Brent (US$)'})
        fig2.show()
        st.plotly_chart(fig2)
       

with tab0:
    with col3:
        st.write('##### Leitura gráfica ')
        st.write('''
                <br>                
    ➡️ Podemos observar no gráfico "Previsto x Real" (acima), que o modelo prevê um aumento no preço do petróleo Brent para os próximos 30 dias, com um intervalo de confiança de 80%.  
                 A linha azul representa a média prevista, enquanto as linhas pontilhadas representam o intervalo de confiança.  
                <br>
                <br>  
                                
    ➡️ O gráfico ao lado é um recorte que contempla o período de 30 dias previsto pelo modelo para o preço do petróleo Brent, com base nos dados históricos dos últimos 10 anos.  
''', unsafe_allow_html=True, use_container_width=True)


# Validações == cross_validation e performance_metrics / verificar se é viavel mostrar no Streamlit
df_anp_range_10_anos_prophet_cv = cross_validation(m2, initial='730 days', period='180 days', horizon = '30 days')
df_anp_range_10_anos_prophet_cv.sort_values("ds", ascending=False)


# Validação
with tab1:
    col3, col4=st.columns(2)
    with col3:
        st.write("##### Métricas de erro para a validação do modelo")
        performance2 = performance_metrics(df_anp_range_10_anos_prophet_cv)
        limited_df = performance2.head(30)
        st.dataframe(limited_df, use_container_width=True)

with tab1:
    with col4:
        st.write('''
    #### Validação do modelo
        
     ➡️ mse: média dos quadrados dos erros (quanto mais próximo de 0, melhor é o modelo).  
                 * Muito útil quando temos outliers, mas não é o caso neste dataset.
                 
     ➡️ rmse: raiz quadrada do MSE (quanto mais próximo de 0, melhor).
                 
     ➡️ mae: média dos valores absolutos dos erros.
       
     ➡️ mape: média dos valores absolutos dos erros percentuais.  
                 * No modelo apresentado o erro percentual médio está entre 8 e 17%.
                
     ➡️ mdape: mediana dos valores absolutos dos erros percentuais.
                 
     ➡️ smape: erro percentual absoluto simétrico médio.
                 
     ➡️ cavarege: proporção de vezes que o intervalo de confiança real contém o valor real (quanto mais próximo de 1, melhor).

        ''')

# Fatores históricos
with tab2:
        st.write('''
    #### Alguns fatores histórico e crises que influenciaram o preço do petróleo Brent:
        
    ➡️ 2008: Bolha imobiliária nos Estados Unidos, ao atingir preços bem acima do mercado, o setor acabou entrando em colapso, a crise financeira e a recessão frearam a demanda por petróleo.

    ➡️ 2011: Maior alta em comparação aos últimos dois anos, ocasionado grande parte devida à crise na Líbia, um dos maiores produtores de petróleo do mundo, conflitos no país levaram a uma interrupção significativa na produção. Como resultado, a oferta global de petróleo diminuiu, levando a um aumento nos preços.  
                 
    ➡️ 2016: Excesso de oferta no mercado resultou em uma queda significativa nos preços do petróleo, durante esse período, vários fatores contribuíram para o excesso de oferta:  
    * aumentou significativo da produção de petróleo nos Estados Unidos.
    * a OPEP manteve sua produção em níveis elevados para manter sua participação de mercado e pressionar os produtores de petróleo de xisto dos EUA.  
    * desaceleração econômica em várias partes do mundo, incluindo a China, reduziu a demanda por petróleo.  
    
    ➡️ 2020: COVID -19, a crise do coronavírus e as medidas de lockdown provocaram queda brusca na demanda por combustivel, provocando uma queda acentuada em todo o periodo.  
    Com as restrições de viagem e o fechamento de muitas indústrias, a demanda por combustível caiu drasticamente.  
    Isso levou a um excesso de oferta de petróleo no mercado, o que resultou em uma queda acentuada nos preços, além disso, a incerteza sobre a duração da pandemia e a velocidade da recuperação econômica também afetaram os preços do petróleo.  
                 
    ➡️ 2022: Devido aos reflexos da guerra entre Rússia e Ucrânia, a produção teve seu escoamento e por consequência exportação prejudicados, seja por motivos de sanções ou pelo jogo geopolítico.
    ''')


with tab3:
     st.write('''
    #### Características do modelo   
              
    🔷 Modelo construido com a biblioteca Prophet, que é uma ferramenta de previsão de séries temporais publicada pelo Facebook.  
            
    🔷 O Prophet é projetado para análise de séries temporais em larga escala e prevê para períodos sazonais.  
            
    🔷 Ele lida com lacunas nos dados, mudanças nas tendências e sazonalidade, e é robusto a outliers.  
            
    🔷 Modelo treinado com dados históricos dos últimos 10 anos, e prevê o preço do petróleo Brent para os próximos 30 dias.  
            
    🔷 O modelo apresenta um erro percentual médio entre 8 e 17%.  
            
    🔷 Ao todo foram testados 3 modelos, todos utilizando o Prophet, porém, cada um com um ajuste de parâmetros diferente,  
    aquele que melhor se ajustou foi o modelo que continha um um range do últimos 10 anos.  
            
    🔷 Para que a previsão se mantenha atualizada, é necessário que o modelo seja re-treinado periodicamente, com novos dados.  
            
    🔷 Após os testes de hiperparametros, foi possivel constatar que o comportamento melhora a medida que o range de previsão diminui, ou seja,  
    inicialmente foi testado com 90 dias, depois 30 e por fim 120 dias, sendo que o modelo que apresentou melhor desempenho foi o de 30 dias.  
            
    🔷 O modelo é uma ferramenta de auxílio à tomada de decisão, e não deve ser utilizado como única fonte de informação.  
    ''')
     

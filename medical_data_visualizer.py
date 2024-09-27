import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Importando os dados
df = pd.read_csv('medical_examination.csv')

# 2 - Criando a coluna 'overweight'
# O IMC é calculado como peso (kg) / altura² (m²)
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0)

# 3 - Normalizando os dados
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4 - Função para plotar o gráfico categórico
def draw_cat_plot():
    # 5 - Convertendo os dados para o formato longo
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Agrupando e reformulando os dados
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7 - Plotando o gráfico categórico
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 9 - Salvando o gráfico
    fig.savefig('catplot.png')
    return fig

# 10 - Função para plotando o heatmap
def draw_heat_map():
    # 11 - Limpando os dados
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12 - Calculando a matriz de correlação
    corr = df_heat.corr()

    # 13 - Gerando uma máscara para a parte superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Configurando a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 - Plotando o heatmap
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True, cbar_kws={'shrink': 0.5}, ax=ax)

    # 16 - Salvando o gráfico
    fig.savefig('heatmap.png')
    return fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

caminho = r'C:\Temp\vs-code-projetos\Material_case_mottu.xlsx'

df_base = pd.read_excel(caminho, sheet_name='Base Total')
df_chip = pd.read_excel(caminho, sheet_name='Clientes chip')

# BLOCO 1: ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
print("="*40)
print("INFORMAÇÕES: BASE TOTAL")
print("="*40)
print(df_base.info())
print(df_base.head())
print(df_base.nunique())
print(df_base.describe())

print("\n" + "="*40)
print("INFORMAÇÕES: CLIENTES CHIP")
print("="*40)
print(df_chip.info())
print(df_chip.head())
print(df_chip.nunique())
print(df_chip.describe())

# BLOCO 2: MERGE (JUNÇÃO) E DIAGNÓSTICO INICIAL

# dados para agrupamento : cidade   modelo_moto plano_moto  forma_pagamento
df = pd.merge(df_base, df_chip, on='client_id', how='left')

# Verificando a quantidade de valores vazios e duplicados após a junção
percentual_vazios = (df.isnull().sum() / len(df)) * 100
duplicados = df['client_id'].duplicated().sum()

print("\nPercentual de vazios:\n", percentual_vazios)
print("\nDuplicados:", duplicados) 
# um pedido tem mais de um cliente, o que pode ser um erro de cadastro ou um cliente com mais de um pedido.
print(f"Total de linhas antes da limpeza: {len(df)}")  

# BLOCO 4: LIMPEZA E TRATAMENTO DE DADOS (BÁSICO)
# após análise percentual dos vazios, alguns dados são relevantes para serem removidos,
# tais: modelo_moto, como um cliente pode ter um registro de retirada da moto sem o modelo?
# cidade, como um cliente pode ter um registro de retirada da moto sem a cidade?
# data_entrada, como um cliente pode ter um registro de retirada da moto sem a data de entrada?
df = df.dropna(subset=['modelo_moto', 'cidade', 'data_entrada'])

# Conversão de datas
colunas_datas = ['data_entrada', 'data_saida', 'data_retirada', 'data_ativacao_chip', 'data_cancelamento_chip']
df[colunas_datas] = df[colunas_datas].apply(pd.to_datetime, errors='coerce')

# farei a análise de outliers para as datas de saidas < data de entradas, para remocao (erros cronológicos)
outliers_moto = df['data_saida'] < df['data_entrada']
outliers_chip = df['data_cancelamento_chip'] < df['data_ativacao_chip']
print(f"\nRemovendo {outliers_moto.sum()} erros nas datas da moto (Saída antes da Entrada).")
print(f"Removendo {outliers_chip.sum()} erros nas datas do chip (Cancelamento antes da Ativação).")
df = df.loc[~outliers_moto & ~outliers_chip]

# BLOCO 5: MAPEAMENTO DE OUTLIERS DE NEGÓCIO
print("\n--- MAPEAMENTO DE NEGÓCIO ---")

# Cenário 1.1: Chip ativado antes de o cliente alugar a moto
chip_antes_moto = df['data_ativacao_chip'] < df['data_entrada']
print(f"Cenário 1.1 (Chip antes da Moto): {chip_antes_moto.sum()} casos.")

# Cenário 1.2: Cliente devolveu a moto, mas não cancelou o chip
moto_devolvida_chip_ativo = (df['data_saida'].notna()) & (df['data_ativacao_chip'].notna()) & (df['data_cancelamento_chip'].isna())
print(f"Cenário 1.2 (Moto cancelada, Chip ativo): {moto_devolvida_chip_ativo.sum()} casos.")

# Cenário 2: Flash Churn (Cancelamento super rápido - ex: até 1 dia)
flash_churn = (df['data_saida'].notna()) & ((df['data_saida'] - df['data_entrada']).dt.days <= 1)
print(f"Cenário 2 (Flash Churn <= 1 dia): {flash_churn.sum()} casos.")

# Cenário 3: Datas no Futuro (Descobrindo a data de corte dinamicamente)
# basicamente estou usando a data de entrada mais recente como referência 
# para identificar contratos que estão ativos (data de saída no futuro) 
# sendo outliers de negócios, pois não fazem sentido para o churn já que o cliente ainda está ativo.

data_corte_dinamica = df['data_entrada'].max() # pegando a data de entrada mais recente como referência para identificar contratos ativos no futuro
datas_futuro = df['data_saida'] > data_corte_dinamica
print(f"Cenário 3 (Data de Saída no Futuro > {data_corte_dinamica.date()}): {datas_futuro.sum()} casos.")

# Cenário 4: Clientes Duplicados (Visualização)
df_duplicados = df[df.duplicated(subset=['client_id'], keep=False)].sort_values(by=['client_id'])
print(f"Cenário 4 (Inspecionando os clientes duplicados): Encontrados {duplicados} clientes.")

# BLOCO 5: APLICANDO REGRAS FINAIS E CRIANDO VARIÁVEIS ALVO
# 1. removendo os cliente duplicados 
df = df.drop_duplicates(subset=['client_id'])

# 2. corrigindo as Datas no Futuro (Transformando contratos ativos em vazios/NaT)
df.loc[df['data_saida'] > data_corte_dinamica, 'data_saida'] = pd.NaT

# 3. corrigindo Chips Fantasmas (Se devolveu a moto, consideramos o chip cancelado na mesma data)
# atualizando a máscara pois remove duplicados
moto_devolvida_chip_ativo = (df['data_saida'].notna()) & (df['data_ativacao_chip'].notna()) & (df['data_cancelamento_chip'].isna())
df.loc[moto_devolvida_chip_ativo, 'data_cancelamento_chip'] = df['data_saida']

# 4. CRIANDO AS VARIÁVEIS ALVO DO CASE
df['churn'] = df['data_saida'].notna()
df['aderiu_chip'] = df['data_ativacao_chip'].notna()
df['ano_entrada'] = df['data_entrada'].dt.year 

# Calculando Tempo de Casa (Ativos contam até a data de corte, Churns contam até a data de saída)
df['tempo_de_casa_dias'] = df.apply(
    lambda row: (row['data_saida'] - row['data_entrada']).days 
    if pd.notna(row['data_saida']) 
    else (data_corte_dinamica - row['data_entrada']).days, 
    axis=1
)

print("="*40)
print(f"Total de Clientes únicos válidos: {len(df)}")
print(f"Total de Churns da base: {df['churn'].sum()}")
print(f"Total de Clientes que aderiram ao Chip: {df['aderiu_chip'].sum()}")


# BLOCO 7: ANÁLISE DE TEMPO DE VIDA (IQR)
# usei o IQR para identificar os outliers e calcular a média real do tempo de casa, ex tempo de vida do cliente, sem a distorção dos extremos (clientes que ficaram muito pouco ou muito tempo, o que pode ser um erro de cadastro ou casos atípicos).
# útil para entender o churn real, sem a distorção dos extremos, e para tomar decisões mais informadas sobre estratégias de retenção e aquisição de clientes.

# Calculando a média original (sujeita à distorção dos extremos)
media_original = df['tempo_de_casa_dias'].mean()

# Encontrando os Quartis e calculando o IQR
Q1 = df['tempo_de_casa_dias'].quantile(0.25)
Q3 = df['tempo_de_casa_dias'].quantile(0.75)
IQR = Q3 - Q1

# tudo fora disso é outlier
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df_limpo_iqr = df[(df['tempo_de_casa_dias'] >= limite_inferior) & (df['tempo_de_casa_dias'] <= limite_superior)]

# Calculando a nova média e pegando o volume de outliers
media_real = df_limpo_iqr['tempo_de_casa_dias'].mean()
qtd_outliers = len(df) - len(df_limpo_iqr)

# etapa final garantindo a média real do tempo de vida do cliente
print("\n" + "="*40)
print("ANÁLISE DE TEMPO DE PERMANÊNCIA (MÉTODO IQR)")
print("="*40)
print(f"Limites de normalidade: de {limite_inferior:.0f} a {limite_superior:.0f} dias de casa.")
print(f"Total de clientes identificados como outliers estatísticos: {qtd_outliers}")
print("-" * 40)
print(f"A média de permanência original é de {media_original:.0f} dias, mas removendo os outliers pelo método IQR, o tempo de vida real do cliente Mottu é de {media_real:.0f} dias.")
print("="*40)

# BLOCO 8: MAPA DE CORRELAÇÃO AVANÇADO
# visualização para entender as relações entre as variáveis, 
# especialmente o impacto do chip e do tempo de casa no churn, 
# além de outras variáveis categóricas como plano e forma de pagamento. 
df_final = df_limpo_iqr.copy()

df_final['churn'] = df_final['churn'].astype(int)
df_final['aderiu_chip'] = df_final['aderiu_chip'].astype(int)

colunas_categoricas = ['plano_moto', 'forma_pagamento']

df_para_correlacao = pd.get_dummies(df_final, columns=colunas_categoricas, drop_first=False)
colunas_selecionadas = ['tempo_de_casa_dias', 'churn', 'aderiu_chip'] + \
                       [col for col in df_para_correlacao.columns if 'plano_moto_' in col or 'forma_pagamento_' in col]

df_corr = df_para_correlacao[colunas_selecionadas].corr()

# Plotando o Heatmap
plt.figure(figsize=(12, 10)) 
sns.heatmap(
    df_corr, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=.5,
    cbar_kws={"shrink": .8}
)
plt.title('Mapa de Correlação - Comportamento do Cliente Mottu', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# BLOCO 9: VISUALIZAÇÕES FINAIS (RESULTADOS DO CASE)

# --- Gráfico 1: Churn Global ---
plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")

# Calculando o percentual de churn
taxas_churn = df_final.groupby('aderiu_chip')['churn'].mean() * 100

# Criando o gráfico de barras
ax = sns.barplot(
    x=['Sem Chip', 'Com Chip'], 
    y=taxas_churn.values, 
    palette=['#e74c3c', '#2ecc71'] # Vermelho e Verde
)

plt.title('Taxa de Churn Global: Com Chip vs Sem Chip', fontsize=14, pad=15)
plt.ylabel('Taxa de Churn (%)', fontsize=12)

# Adicionando os percentuais em cima das barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=12, fontweight='bold', 
                xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()


# --- Gráfico 2: Churn por Safra (A Descoberta Principal) ---
plt.figure(figsize=(10, 6))

# Calculando o percentual de churn quebrado por Ano e Adesão
df_cohort = df_final.groupby(['ano_entrada', 'aderiu_chip'])['churn'].mean().reset_index()
df_cohort['churn'] *= 100

# Criando o gráfico de barras agrupadas
ax2 = sns.barplot(
    data=df_cohort, 
    x='ano_entrada', 
    y='churn', 
    hue='aderiu_chip', 
    palette=['#e74c3c', '#2ecc71']
)

plt.title('Taxa de Churn por Safra (Ano de Entrada)', fontsize=14, pad=15)
plt.ylabel('Taxa de Churn (%)', fontsize=12)
plt.xlabel('Ano de Entrada', fontsize=12)

# Ajustando a legenda
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=['Sem Chip', 'Com Chip'], title='Adesão ao Chip', loc='upper right')

for p in ax2.patches:
    altura = p.get_height()
    if altura > 0: 
        ax2.annotate(f'{altura:.1f}%', 
                     (p.get_x() + p.get_width() / 2., altura), 
                     ha='center', va='bottom', 
                     fontsize=10, fontweight='bold', 
                     xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()
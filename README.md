# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

Modelo supervisionado de classificação binária que prevê se um município brasileiro atinge ou
não a própria meta oficial de alfabetização (INEP), a partir de dados educacionais, territoriais,
populacionais e socioeconômicos.

Este README é um guia de apresentação do projeto: contexto, principais conclusões e como executar.
O desenvolvimento completo — código, gráficos, hipóteses testadas e decisões passo a passo — está
nos notebooks (`01_eda.ipynb`, `02_modelagem.ipynb`, `03_clustering.ipynb`); os links abaixo
apontam para onde cada resultado foi construído.

## Estrutura do projeto

```
data/               dataset bruto (CSV) e processado (Parquet)
notebooks/          01_eda.ipynb (EDA), 02_modelagem.ipynb (pipeline de ML), 03_clustering.ipynb (K-Means)
src/preprocessing/  extração e junção dos dados (BigQuery)
src/modeling/       pré-processador, modelos candidatos, treino/seleção do campeão e clusterização
src/evaluation/     métricas de classificação e resumo de validação cruzada
src/visualization/  gráficos usados na EDA e na modelagem
models/             pipeline final treinado (joblib)
reports/            métricas do modelo campeão e perfil dos clusters (JSON)
images/             gráficos gerados pelos notebooks
```

## Contexto do problema

A alfabetização na idade certa é uma das metas educacionais mais monitoradas no Brasil, com meta
nacional de 100% dos municípios até 2030. Gestores públicos precisam identificar, com antecedência,
quais municípios correm risco de não atingir suas metas, para priorizar recursos e intervenções.

Os dados de alfabetização e metas vêm da camada Silver de um projeto anterior de engenharia de
dados (BigQuery), no grão município × ano, enriquecidos com fontes públicas do IBGE e do INEP.

## Objetivo analítico

Prever, para cada município e ano, se a taxa de alfabetização atinge ou não a meta oficial do
próprio município, e usar o modelo para responder cinco perguntas de negócio: quais fatores mais
impactam a alfabetização, quais municípios estão em maior risco, quais regiões têm padrões
semelhantes, como prever municípios que podem não atingir metas futuras, e quais variáveis mais
pesam na decisão do modelo.

## Descrição da base utilizada

**Unidade de análise: município × ano** (não aluno individual) — é o único grão em que existem
variáveis territoriais e socioeconômicas reais associadas ao resultado de alfabetização.

- **9.921 linhas** (4.689 municípios em 2023 + 5.232 em 2024), 21 colunas.
- **Fontes:** taxa/meta de alfabetização (rede pública) + população (IBGE), PIB (IBGE),
  território/região (IBGE) e indicadores educacionais complementares (INEP), todos unidos por
  `id_municipio`.
- **Alvo (`atingiu_meta`):** `taxa_alfabetizacao >= meta_alfabetizacao_2024`.
- **Vazamento de dado tratado:** `taxa_alfabetizacao`, `media_portugues` e
  `meta_alfabetizacao_2024` ficam no dataset só para referência/reconstrução do alvo — nunca
  entram como features do modelo.

Extração, junção, tratamento de nulos e a lista completa de colunas estão em
`src/preprocessing/extract_data.py` e na primeira metade de `notebooks/01_eda.ipynb`.

![Distribuição do alvo por ano](images/01_target_por_ano.png)

## Etapas de modelagem

Extração (BigQuery) → EDA (`01_eda.ipynb`) → pré-processamento com `ColumnTransformer`
(imputação, `log1p`+padronização para população/PIB, one-hot para UF) → comparação de 3 modelos
por validação cruzada → `RandomizedSearchCV` no campeão → validação em holdout estratificado e em
validação temporal (treina 2023, testa 2024) → interpretabilidade (coeficientes + SHAP).

Código completo em `src/modeling/pipeline.py`, `src/modeling/train_model.py` e
`notebooks/02_modelagem.ipynb`.

## Escolha do algoritmo

| Modelo | ROC-AUC médio (CV) |
|---|---|
| **Regressão Logística (campeã)** | **0,6960** (após tuning, `C=0,1`) |
| Gradient Boosting | 0,6891 |
| Random Forest | 0,6575 |

A Regressão Logística venceu a comparação por validação cruzada — resultado que contraria a
expectativa inicial da EDA (que sugeria árvores, pelas correlações lineares fracas), reportado
como está. Detalhes da comparação em `notebooks/02_modelagem.ipynb`.

![Comparação dos modelos candidatos](images/09_comparacao_modelos.png)

## Métricas de avaliação

| Métrica | Holdout (20%) | Validação temporal (2023→2024) |
|---|---|---|
| ROC-AUC | 0,6745 | 0,5535 |
| Acurácia | 0,6408 | 0,4641 |
| Precisão | 0,5630 | 0,5131 |
| Recall | 0,3892 | 0,0902 |
| F1 | 0,4603 | 0,1534 |

Com limiar de decisão ajustado para ≈0,38 (em vez do padrão 0,5), o recall no holdout sobe para
≈0,70 mantendo precisão ≈0,50 — ponto de operação recomendado para uso como ferramenta de
triagem. A queda na validação temporal é discutida em "Limitações". Curvas e matriz de confusão em
`notebooks/02_modelagem.ipynb`.

![Curva ROC](images/11_curva_roc.png)

## Interpretação dos resultados

Coeficientes e SHAP convergem: o **sinal mais forte é geográfico** (Bahia, Sergipe e Tocantins
negativos; Ceará fortemente positivo). Fora da geografia, `dsu_ef_anos_iniciais` (% de docentes
com curso superior) é a variável de maior impacto médio, com efeito negativo — um padrão
contraintuitivo, consistente nas 5 regiões do país, sem explicação causal identificada até agora.

![Importância das variáveis (SHAP)](images/14_shap_summary.png)

## Regiões com padrões semelhantes (clusterização)

Além do modelo supervisionado, os municípios de 2024 foram agrupados por perfil territorial,
populacional, socioeconômico e educacional (K-Means, sem usar UF/região como feature de
agrupamento, só para caracterizar os grupos depois de formados). O melhor agrupamento (silhouette
score) foi **k=2**:

| Cluster | Municípios | % atingiu meta | Taxa de alfabetização média | Composição regional dominante | `dsu_ef_anos_iniciais` médio |
|---|---|---|---|---|---|
| 0 | 3.931 (75%) | 56,1% | 66,4% | Sudeste (38%), Sul (26%), Nordeste (21%) | 15,4% |
| 1 | 1.301 (25%) | 47,1% | 54,9% | Nordeste (69%), Norte (17%), 31% Amazônia Legal | 27,5% |

Os dois grupos têm desfecho de alfabetização bem diferente, mas o cluster 1 (pior desfecho) tem
quase o dobro do percentual de docentes com curso superior do cluster 0 — o mesmo padrão
contraintuitivo já visto na interpretação do modelo supervisionado, agora associado a um perfil
regional inteiro (Nordeste/Norte/Amazônia Legal), não a uma variável isolada. Os silhouette scores
foram modestos em todos os k testados (0,16-0,23), indicando um contínuo de perfis municipais em
vez de fronteiras rígidas. Desenvolvimento completo em `notebooks/03_clustering.ipynb` e
`src/modeling/clustering.py`.

![Municípios por cluster (PCA)](images/15_clusters_pca.png)
![Composição regional de cada cluster](images/16_clusters_por_regiao.png)

## Insights encontrados

- A melhora de 2023 para 2024 (23,2% → 53,8% de municípios na meta) é uma melhora real na taxa de
  alfabetização, não uma meta mais fácil de bater.
- O Sul foi a única região que piorou (RS caiu para 8,3% de municípios na meta em 2024, ante 53,8%
  nacional) — coincide no tempo com as enchentes de abril-maio de 2024 no estado, hipótese
  plausível mas não comprovável com as variáveis disponíveis.
- População e PIB são fortemente correlacionados entre si (0,95) e quase não têm correlação
  individual com o alvo.
- Dos 6.018 municípios que não atingiram a meta em 2024, 1.225 (20,4%) estavam a menos de 2 pontos
  percentuais dela.

Análise completa, incluindo hipóteses testadas e não confirmadas, em `notebooks/01_eda.ipynb`.

## Limitações do projeto

- Dataset pequeno para o grão escolhido: apenas 2 anos (2023-2024).
- Validação temporal (treina 2023, testa 2024) mostra ROC-AUC caindo para 0,55 — o modelo serve
  para ranquear risco relativo dentro do mesmo ciclo de avaliação, não para prever o patamar
  nacional de um ano futuro sem recalibração.
- PIB de 2024 usa 2023 como proxy (defasagem normal da fonte).
- Possível choque exógeno no RS (enchentes de 2024) não capturado por nenhuma variável do dataset.
- A clusterização teve silhouette scores modestos (0,16-0,23) em todos os valores de k testados —
  os dois grupos encontrados são uma tendência real, não uma fronteira rígida entre perfis de
  município.

## Aplicação prática para políticas públicas

- Usado como ferramenta de triagem (limiar ≈0,38), o modelo identifica ≈70% dos municípios que de
  fato não atingirão a meta — adequado para priorizar visitas técnicas e recursos.
- Peso forte de UF permite priorização geográfica objetiva (ex.: acompanhamento reforçado em
  Bahia, Sergipe e Tocantins).
- O grupo de 1.225 municípios a menos de 2 pontos da própria meta é um alvo natural para
  intervenções pontuais de curto prazo e baixo custo.
- O modelo deve ser recalibrado a cada novo ciclo de avaliação, dado o salto real de patamar
  observado entre 2023 e 2024.

![Gap até a meta entre os municípios que não atingiram](images/08_gap_ate_meta.png)

## Possíveis evoluções futuras

- Investigar a causa do perfil regional Nordeste/Norte/Amazônia Legal ter pior desfecho de
  alfabetização apesar de maior percentual de docentes com curso superior (achado da
  clusterização).
- Investigar a hipótese das enchentes no RS com uma fonte externa de eventos climáticos.
- Ampliar a série histórica conforme novos ciclos do INEP forem publicados.
- Investigar a causa da relação negativa entre `dsu_ef_anos_iniciais` e o alvo.
- Reavaliar a necessidade de manter população e PIB simultaneamente, dada a multicolinearidade.

## Como reproduzir

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/preprocessing/extract_data.py
python src/modeling/train_model.py
python src/modeling/clustering.py
```

A extração requer `gcloud auth login` já configurado com acesso de leitura ao projeto BigQuery de
origem dos dados e às tabelas públicas do Base dos Dados. Os notebooks podem ser reexecutados a
partir de `data/processed/*.parquet`, sem credenciais, desde que a extração já tenha rodado uma
vez.

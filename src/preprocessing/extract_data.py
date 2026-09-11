"""
Extrai o dataset município x ano usado no Tech Challenge Fase 3, combinando:
  - techchallenge2-afabetizacao.silver.municipio (camada Silver da Fase 2, INEP)
  - 4 datasets públicos do Base dos Dados (BigQuery), joináveis por id_municipio:
      br_ibge_populacao.municipio        -> indicador populacional
      br_ibge_pib.municipio              -> dado socioeconômico
      br_bd_diretorios_brasil.municipio  -> dado territorial (estático, sem ano)
      br_inep_indicadores_educacionais.municipio -> dado educacional complementar

Decisões incorporadas nesta query:
  - Unidade de análise = município x ano (não aluno individual): é o único grão em que existe
    sinal territorial/socioeconômico real nos dados disponíveis.
  - Recorte rede='5' (Pública, Estadual+Municipal) em silver.municipio: mesmo recorte já usado na
    camada Gold da Fase 2, e o único sem duplicidade de id_municipio por ano.
  - Target = taxa_alfabetizacao >= meta_alfabetizacao_2024 (município atingiu a própria meta
    oficial do INEP), não um corte fixo de 80%: mais balanceado entre os dois anos disponíveis e
    responde diretamente à pergunta de negócio "quais municípios podem não atingir metas futuras".
  - Linhas sem meta_alfabetizacao_2024 preenchida são descartadas (WHERE ... IS NOT NULL): usar
    esse valor pra construir o próprio alvo do modelo, então não dá pra imputar sem fabricar o
    rótulo.
  - PIB: só a coluna `pib` (total) é usada. As colunas de valor adicionado por setor
    (va_agropecuaria/va_industria/va_servicos) vêm 100% nulas em 2022-2023 (o IBGE publica a
    quebra setorial com defasagem maior que o total) e foram removidas da extração.
  - PIB é sempre do ano de 2023 (o mais recente publicado), aplicado também às linhas de 2024
    como melhor proxy disponível — é uma defasagem normal do indicador, documentar no README.
  - Indicadores educacionais filtrados em rede='Pública', localizacao='Total' (mesma lógica de
    recorte usada em silver.municipio) para evitar duplicidade no join.

⚠️ AVISO DE DATA LEAKAGE — importante para quem for montar o pipeline de modelagem:
  As colunas `taxa_alfabetizacao`, `media_portugues` e `meta_alfabetizacao_2024` ficam no arquivo
  de saída só para referência/EDA e para permitir reconstruir o target. NÃO devem entrar como
  features (X) do modelo: `meta_alfabetizacao_2024` define o próprio target por construção, e
  `taxa_alfabetizacao`/`media_portugues` vêm do mesmo processo de avaliação que originou o target
  (correlação quase determinística com `atingiu_meta`).

Requer autenticação já configurada via `gcloud auth login` + acesso de leitura ao projeto
`techchallenge2-afabetizacao` (ou ajuste BIGQUERY_PROJECT abaixo para outro projeto de billing).

Uso:
    python src/preprocessing/extract_data.py
"""

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

BIGQUERY_PROJECT = "techchallenge2-afabetizacao"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "municipio_alfabetizacao_enriquecido.csv"
PROCESSED_PARQUET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "municipio_alfabetizacao_enriquecido.parquet"
)

EXTRACTION_QUERY = """
WITH base AS (
  SELECT
    ano, id_municipio,
    taxa_alfabetizacao, media_portugues,
    meta_alfabetizacao_2024,
    CASE WHEN taxa_alfabetizacao >= meta_alfabetizacao_2024 THEN 1 ELSE 0 END AS atingiu_meta
  FROM `techchallenge2-afabetizacao.silver.municipio`
  WHERE rede = '5' AND meta_alfabetizacao_2024 IS NOT NULL
),
pop AS (
  SELECT ano, id_municipio, populacao
  FROM `basedosdados.br_ibge_populacao.municipio`
  WHERE ano IN (2023, 2024)
),
pib AS (
  SELECT id_municipio, pib
  FROM `basedosdados.br_ibge_pib.municipio`
  WHERE ano = 2023
),
territorio AS (
  SELECT id_municipio, nome, sigla_uf, nome_uf, nome_regiao, nome_mesorregiao, capital_uf, amazonia_legal
  FROM `basedosdados.br_bd_diretorios_brasil.municipio`
),
educ AS (
  SELECT ano, id_municipio,
    taxa_aprovacao_ef_anos_iniciais, taxa_reprovacao_ef_anos_iniciais, taxa_abandono_ef_anos_iniciais,
    tdi_ef_anos_iniciais, atu_ef_anos_iniciais, dsu_ef_anos_iniciais
  FROM `basedosdados.br_inep_indicadores_educacionais.municipio`
  WHERE rede = 'Pública' AND localizacao = 'Total' AND ano IN (2023, 2024)
)
SELECT
  b.ano, b.id_municipio,
  t.nome AS nome_municipio, t.sigla_uf, t.nome_uf, t.nome_regiao, t.nome_mesorregiao,
  t.capital_uf, t.amazonia_legal,
  p.populacao,
  pb.pib,
  e.taxa_aprovacao_ef_anos_iniciais, e.taxa_reprovacao_ef_anos_iniciais, e.taxa_abandono_ef_anos_iniciais,
  e.tdi_ef_anos_iniciais, e.atu_ef_anos_iniciais, e.dsu_ef_anos_iniciais,
  b.taxa_alfabetizacao, b.media_portugues, b.meta_alfabetizacao_2024,
  b.atingiu_meta
FROM base b
LEFT JOIN pop p ON p.ano = b.ano AND p.id_municipio = b.id_municipio
LEFT JOIN pib pb ON pb.id_municipio = b.id_municipio
LEFT JOIN territorio t ON t.id_municipio = b.id_municipio
LEFT JOIN educ e ON e.ano = b.ano AND e.id_municipio = b.id_municipio
ORDER BY b.ano, b.id_municipio
"""


def extract() -> pd.DataFrame:
    client = bigquery.Client(project=BIGQUERY_PROJECT)
    return client.query(EXTRACTION_QUERY).result().to_dataframe()


def main() -> None:
    df = extract()

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False)

    print(f"Linhas extraídas: {len(df)}")
    print(f"Salvo em: {RAW_CSV_PATH}")
    print(f"Salvo em: {PROCESSED_PARQUET_PATH}")


if __name__ == "__main__":
    main()

"""
Monta o pré-processamento e os modelos candidatos para o problema de classificação binária
(atingiu_meta) no grão município x ano.

Features usadas (X): sigla_uf, capital_uf, amazonia_legal, populacao, pib e os 6 indicadores
educacionais complementares (taxa_aprovacao/reprovacao/abandono, tdi, atu, dsu — todos referentes
aos anos iniciais do Ensino Fundamental, mesmo recorte de série da avaliação de alfabetização).

Colunas explicitamente fora do X:
  - id_municipio, nome_municipio, nome_uf, nome_mesorregiao: identificadores/redundantes.
  - nome_regiao: redundante com sigla_uf (região é determinada pela UF) e menos granular — a UF
    sozinha já captura o padrão regional com mais detalhe (ex.: RS isolado dentro do Sul).
  - ano: teria só 2 valores observados (2023/2024); usar como feature faria o modelo aprender
    "efeito do ano" em vez de padrão estrutural, o que não generaliza para anos futuros — objetivo
    do desafio é justamente prever risco futuro, não replicar o efeito histórico de um ano
    específico. `ano` é usado só para particionar dados numa validação temporal (fora deste
    módulo), não como feature.
  - taxa_alfabetizacao, media_portugues, meta_alfabetizacao_2024: vazamento de dado — a primeira e
    a meta definem o target por construção; a segunda tem correlação de 0,93 com a primeira.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "atingiu_meta"

LOG_NUMERIC_FEATURES = ["populacao", "pib"]
PLAIN_NUMERIC_FEATURES = [
    "taxa_aprovacao_ef_anos_iniciais",
    "taxa_reprovacao_ef_anos_iniciais",
    "taxa_abandono_ef_anos_iniciais",
    "tdi_ef_anos_iniciais",
    "atu_ef_anos_iniciais",
    "dsu_ef_anos_iniciais",
    "capital_uf",
    "amazonia_legal",
]
CATEGORICAL_FEATURES = ["sigla_uf"]

FEATURE_COLUMNS = LOG_NUMERIC_FEATURES + PLAIN_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Colunas mantidas no dataset só para referência/auditoria — nunca entram em X.
LEAKAGE_COLUMNS = ["taxa_alfabetizacao", "media_portugues", "meta_alfabetizacao_2024"]


def build_preprocessor() -> ColumnTransformer:
    log_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    plain_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log_numeric", log_numeric_pipeline, LOG_NUMERIC_FEATURES),
            ("plain_numeric", plain_numeric_pipeline, PLAIN_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }
    return {
        name: Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
        for name, model in candidates.items()
    }


# Espaços de busca para RandomizedSearchCV, usados no ajuste do modelo campeão.
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    },
    "random_forest": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [None, 6, 10, 14, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.1, 0.2],
        "model__max_depth": [2, 3, 4, 5],
        "model__subsample": [0.7, 0.85, 1.0],
    },
}

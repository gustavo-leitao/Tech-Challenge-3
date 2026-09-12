"""
Treina e seleciona o modelo campeão para prever `atingiu_meta` (município x ano).

Validação em duas frentes, ambas exigidas pelo objetivo de generalização do problema:
  1. Split treino/teste estratificado (80/20) sobre os dois anos agrupados, com StratifiedKFold
     de 5 dobras usado para comparar os candidatos e para o RandomizedSearchCV do campeão —
     validação estatística padrão.
  2. Validação temporal fora da amostra: treina só com 2023 e avalia só em 2024 (município nunca
     visto nesse recorte pelo modelo nesse ano). Isso testa diretamente a capacidade real do
     desafio — prever risco futuro a partir de padrões estruturais — e não só a validação
     estatística cruzada de uma amostra embaralhada.

Uso:
    python src/modeling/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import classification_metrics, cross_val_summary  # noqa: E402
from src.modeling.pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    PARAM_DISTRIBUTIONS,
    TARGET,
    build_candidate_models,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "municipio_alfabetizacao_enriquecido.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def main() -> None:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = build_candidate_models(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("=== Comparação de candidatos (5-fold CV, ROC-AUC no treino) ===")
    cv_scores = {}
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = scores
        print(f"{name:>20}: ROC-AUC médio = {scores.mean():.4f} (+/- {scores.std():.4f})")

    summary = cross_val_summary(cv_scores)
    champion_name = summary.iloc[0]["modelo"]
    print(f"\nModelo campeão (maior ROC-AUC médio): {champion_name}")

    print(f"\n=== RandomizedSearchCV no campeão ({champion_name}) ===")
    search = RandomizedSearchCV(
        candidates[champion_name],
        param_distributions=PARAM_DISTRIBUTIONS[champion_name],
        n_iter=20,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Melhor ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"Melhores hiperparâmetros: {search.best_params_}")

    best_pipeline = search.best_estimator_

    print("\n=== Avaliação no conjunto de teste (holdout, nunca visto no treino/tuning) ===")
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, y_pred, y_proba)
    for k, v in test_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Validação temporal: treina em 2023, avalia em 2024 ===")
    train_2023 = df[df["ano"] == 2023]
    test_2024 = df[df["ano"] == 2024]
    temporal_pipeline = build_candidate_models(random_state=RANDOM_STATE)[champion_name]
    temporal_pipeline.set_params(**{k: v for k, v in search.best_params_.items()})
    temporal_pipeline.fit(train_2023[FEATURE_COLUMNS], train_2023[TARGET].astype(int))
    y_pred_temporal = temporal_pipeline.predict(test_2024[FEATURE_COLUMNS])
    y_proba_temporal = temporal_pipeline.predict_proba(test_2024[FEATURE_COLUMNS])[:, 1]
    temporal_metrics = classification_metrics(
        test_2024[TARGET].astype(int), y_pred_temporal, y_proba_temporal
    )
    for k, v in temporal_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "pipeline_final.joblib")

    report = {
        "modelo_campeao": champion_name,
        "melhores_hiperparametros": search.best_params_,
        "cv_roc_auc_treino": search.best_score_,
        "cv_comparacao_candidatos": summary.to_dict(orient="records"),
        "metricas_teste_holdout": test_metrics,
        "metricas_validacao_temporal_2023_para_2024": temporal_metrics,
    }
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)

    print(f"\nPipeline final salvo em {MODELS_DIR / 'pipeline_final.joblib'}")
    print(f"Relatório de métricas salvo em {REPORTS_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()

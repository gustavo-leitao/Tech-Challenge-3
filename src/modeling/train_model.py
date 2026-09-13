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

O ponto de operação usado como ferramenta de triagem (recall/precisão da classe "risco" —
município que NÃO atinge a meta, `atingiu_meta=0`) é escolhido só com o conjunto de treino, via
probabilidade fora-da-dobra (`cross_val_predict`) — o conjunto de teste nunca entra na escolha do
threshold, só na confirmação final do ponto já escolhido. Precisão/recall/F1 são sempre reportados
com `pos_label` explícito, porque a classe "risco" (0) e a classe "atingiu" (1) não são espelhos
uma da outra e um threshold bom para uma pode ser ruim para a outra.

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
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    cross_val_summary,
    pick_threshold_for_recall,
)
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
TARGET_RECALL_RISCO = 0.70


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

    print(f"\n=== Threshold de triagem para recall>={TARGET_RECALL_RISCO:.0%} na classe risco (CV no treino) ===")
    threshold_pipeline = build_candidate_models(random_state=RANDOM_STATE)[champion_name]
    threshold_pipeline.set_params(**search.best_params_)
    oof_proba_atingir = cross_val_predict(
        threshold_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold_info = pick_threshold_for_recall(
        y_train, oof_proba_atingir, target_recall=TARGET_RECALL_RISCO, pos_label=0
    )
    print(f"Threshold escolhido (sobre P(atingir a meta)): {threshold_info['threshold_atingir_meta']:.4f}")
    print(f"Recall classe risco fora da amostra: {threshold_info['recall_pos_label']:.4f}")
    print(f"Precisão classe risco fora da amostra: {threshold_info['precision_pos_label']:.4f}")

    print("\n=== Avaliação no conjunto de teste (holdout, nunca visto no treino/tuning) ===")
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, y_pred, y_proba)
    for k, v in test_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Teste, classe risco (não atingiu a meta), threshold padrão 0,5 ===")
    test_metrics_risco_padrao = classification_metrics(y_test, y_pred, y_proba, pos_label=0)
    for k, v in test_metrics_risco_padrao.items():
        print(f"{k:>10}: {v:.4f}")

    print(f"\n=== Teste, classe risco, threshold de triagem ({threshold_info['threshold_atingir_meta']:.4f}) ===")
    y_pred_triagem = (y_proba >= threshold_info["threshold_atingir_meta"]).astype(int)
    test_metrics_risco_triagem = classification_metrics(y_test, y_pred_triagem, y_proba, pos_label=0)
    for k, v in test_metrics_risco_triagem.items():
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

    print("\n=== Validação temporal, classe risco (não atingiu a meta), threshold padrão 0,5 ===")
    temporal_metrics_risco = classification_metrics(
        test_2024[TARGET].astype(int), y_pred_temporal, y_proba_temporal, pos_label=0
    )
    for k, v in temporal_metrics_risco.items():
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
        "metricas_teste_holdout_classe_risco_threshold_padrao": test_metrics_risco_padrao,
        "threshold_triagem_risco": threshold_info,
        "metricas_teste_holdout_classe_risco_threshold_triagem": test_metrics_risco_triagem,
        "metricas_validacao_temporal_2023_para_2024": temporal_metrics,
        "metricas_validacao_temporal_2023_para_2024_classe_risco": temporal_metrics_risco,
    }
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)

    print(f"\nPipeline final salvo em {MODELS_DIR / 'pipeline_final.joblib'}")
    print(f"Relatório de métricas salvo em {REPORTS_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()

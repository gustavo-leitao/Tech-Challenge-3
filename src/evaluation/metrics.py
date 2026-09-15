"""Funções de avaliação reutilizáveis pelo script de treino e pelo notebook de modelagem."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba, pos_label: int = 1) -> dict:
    """Métricas de classificação para a classe `pos_label`.

    `y_proba` é sempre a probabilidade da classe 1 (`atingiu_meta`), como sai de
    `predict_proba(...)[:, 1]`. Precisão/recall/F1 da classe "risco" (município não
    atingiu a meta) não são o espelho dos da classe "atingiu" — por isso `pos_label`
    precisa ser explícito sempre que o número for usado para decidir algo (ex.: qual
    threshold de triagem usar), em vez de assumir o default do sklearn (`pos_label=1`)
    sem checar qual classe interessa para a pergunta de negócio.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def cross_val_summary(cv_results: dict[str, np.ndarray]) -> pd.DataFrame:
    """Recebe {nome_modelo: array_de_scores_por_fold} e devolve média/desvio-padrão."""
    rows = []
    for nome, scores in cv_results.items():
        rows.append({"modelo": nome, "media": scores.mean(), "desvio_padrao": scores.std()})
    return pd.DataFrame(rows).sort_values("media", ascending=False).reset_index(drop=True)


def pick_threshold_for_recall(y_true, y_proba, target_recall: float, pos_label: int = 0) -> dict:
    """Escolhe o maior threshold que ainda garante recall >= `target_recall` para a
    classe `pos_label`.

    Deve ser chamado com probabilidades fora da amostra de teste (ex.: saída de
    `cross_val_predict` no conjunto de treino) — nunca com o conjunto de teste, para
    o teste continuar reservado só para a avaliação final, e não para escolher o
    ponto de operação do modelo.

    `y_proba` é sempre a probabilidade da classe 1 (`atingiu_meta`). O threshold
    retornado já vem convertido para essa mesma escala (`threshold_atingir_meta`),
    que é o array que o pipeline realmente produz em uso — quem for aplicar o corte
    não precisa lembrar de inverter a probabilidade toda vez.
    """
    proba_pos_label = y_proba if pos_label == 1 else 1 - y_proba
    precision, recall, thresholds = precision_recall_curve(y_true, proba_pos_label, pos_label=pos_label)
    idx = np.where(recall[:-1] >= target_recall)[0]
    if len(idx) == 0:
        raise ValueError(f"Nenhum threshold atinge recall >= {target_recall} para pos_label={pos_label}")
    best_idx = idx[np.argmax(thresholds[idx])]
    threshold_pos_label = float(thresholds[best_idx])
    threshold_atingir_meta = threshold_pos_label if pos_label == 1 else 1 - threshold_pos_label
    return {
        "pos_label_otimizado": pos_label,
        "threshold_atingir_meta": threshold_atingir_meta,
        "recall_pos_label": float(recall[best_idx]),
        "precision_pos_label": float(precision[best_idx]),
    }

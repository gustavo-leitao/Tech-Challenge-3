"""Funções de avaliação reutilizáveis pelo script de treino e pelo notebook de modelagem."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def cross_val_summary(cv_results: dict[str, np.ndarray]) -> pd.DataFrame:
    """Recebe {nome_modelo: array_de_scores_por_fold} e devolve média/desvio-padrão."""
    rows = []
    for nome, scores in cv_results.items():
        rows.append({"modelo": nome, "media": scores.mean(), "desvio_padrao": scores.std()})
    return pd.DataFrame(rows).sort_values("media", ascending=False).reset_index(drop=True)

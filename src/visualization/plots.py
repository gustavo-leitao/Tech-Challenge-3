"""Funções de plotagem reutilizáveis pelos notebooks de EDA e modelagem."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_confusion_matrix(y_true, y_pred, ax=None, labels=("não atingiu", "atingiu")):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Matriz de confusão")
    return ax


def plot_roc_curve(y_true, y_proba, ax=None, label="modelo"):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="aleatório")
    ax.set_title("Curva ROC")
    ax.legend()
    return ax


def plot_feature_importance(feature_names, importances, ax=None, top_n=15):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    order = np.argsort(importances)[-top_n:]
    ax.barh(np.array(feature_names)[order], np.array(importances)[order], color="#4C72B0")
    ax.set_title(f"Top {top_n} features por importância")
    ax.set_xlabel("Importância")
    return ax

"""
Agrupa municípios por perfil territorial, populacional, socioeconômico e educacional
(K-Means), para responder a pergunta de negócio "quais regiões apresentam padrões semelhantes".

Diferente do modelo supervisionado (`pipeline.py`/`train_model.py`), aqui `sigla_uf`/`nome_regiao`
NÃO entram como feature de agrupamento — são usados só depois, para caracterizar os clusters
encontrados e verificar se eles coincidem com as regiões geográficas oficiais ou as atravessam.

Uso:
    python src/modeling/clustering.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "municipio_alfabetizacao_enriquecido.parquet"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
SNAPSHOT_YEAR = 2024  # ano mais recente e com mais municípios cobertos (5.232 vs. 4.689 em 2023)

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
CLUSTER_FEATURES = LOG_NUMERIC_FEATURES + PLAIN_NUMERIC_FEATURES

# Colunas mantidas só para caracterizar os clusters depois de formados — nunca entram como
# feature de agrupamento (região/UF são o que queremos comparar contra os clusters; taxa de
# alfabetização e meta são desfecho, não perfil).
PROFILE_ONLY_COLUMNS = [
    "sigla_uf",
    "nome_regiao",
    "taxa_alfabetizacao",
    "atingiu_meta",
]


def load_snapshot() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    return df[df["ano"] == SNAPSHOT_YEAR].reset_index(drop=True)


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
    return ColumnTransformer(
        transformers=[
            ("log_numeric", log_numeric_pipeline, LOG_NUMERIC_FEATURES),
            ("plain_numeric", plain_numeric_pipeline, PLAIN_NUMERIC_FEATURES),
        ]
    )


def choose_k(X_transformed: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(
            X_transformed
        )
        rows.append({"k": k, "silhouette": silhouette_score(X_transformed, labels)})
    return pd.DataFrame(rows)


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    df = df.copy()
    df["cluster"] = labels
    profiles = []
    for cluster_id, group in df.groupby("cluster"):
        regiao_pct = (
            group["nome_regiao"].value_counts(normalize=True).mul(100).round(1).to_dict()
        )
        profiles.append(
            {
                "cluster": int(cluster_id),
                "n_municipios": int(len(group)),
                "pct_atingiu_meta": round(float(group["atingiu_meta"].mean()) * 100, 1),
                "taxa_alfabetizacao_media": round(float(group["taxa_alfabetizacao"].mean()), 1),
                "populacao_mediana": round(float(group["populacao"].median()), 0),
                "pib_mediano": round(float(group["pib"].median()), 0),
                "pct_capital_uf": round(float(group["capital_uf"].mean()) * 100, 1),
                "pct_amazonia_legal": round(float(group["amazonia_legal"].mean()) * 100, 1),
                "dsu_ef_anos_iniciais_media": round(
                    float(group["dsu_ef_anos_iniciais"].mean()), 1
                ),
                "composicao_regional_pct": regiao_pct,
            }
        )
    return sorted(profiles, key=lambda p: p["pct_atingiu_meta"], reverse=True)


def plot_pca_scatter(X_transformed: np.ndarray, labels: np.ndarray, path: Path) -> None:
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_transformed)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    ax.set_title(f"Municípios ({SNAPSHOT_YEAR}) por cluster — projeção PCA")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cluster_by_region(df: pd.DataFrame, labels: np.ndarray, path: Path) -> None:
    df = df.copy()
    df["cluster"] = labels
    composicao = pd.crosstab(df["cluster"], df["nome_regiao"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    composicao.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Composição regional de cada cluster")
    ax.set_ylabel("% de municípios do cluster")
    ax.set_xlabel("Cluster")
    ax.legend(title="Região", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_snapshot()
    needed_columns = CLUSTER_FEATURES + PROFILE_ONLY_COLUMNS

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(df[CLUSTER_FEATURES])

    print(f"Municípios no snapshot {SNAPSHOT_YEAR}: {len(df)}")
    print("\n=== Silhouette score por número de clusters ===")
    k_scores = choose_k(X_transformed)
    for _, row in k_scores.iterrows():
        print(f"k={int(row['k'])}: silhouette={row['silhouette']:.4f}")

    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])
    print(f"\nMelhor k pelo silhouette score: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_transformed)

    profiles = profile_clusters(df[needed_columns], labels)

    IMAGES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    plot_pca_scatter(X_transformed, labels, IMAGES_DIR / "15_clusters_pca.png")
    plot_cluster_by_region(df, labels, IMAGES_DIR / "16_clusters_por_regiao.png")

    report = {
        "ano_snapshot": SNAPSHOT_YEAR,
        "n_municipios": len(df),
        "features_agrupamento": CLUSTER_FEATURES,
        "silhouette_por_k": k_scores.to_dict(orient="records"),
        "k_escolhido": best_k,
        "silhouette_k_escolhido": float(k_scores.loc[k_scores["k"] == best_k, "silhouette"].iloc[0]),
        "perfil_dos_clusters": profiles,
    }
    with open(REPORTS_DIR / "clustering_profile.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPerfil dos {best_k} clusters salvo em {REPORTS_DIR / 'clustering_profile.json'}")
    print(f"Gráficos salvos em {IMAGES_DIR}")


if __name__ == "__main__":
    main()

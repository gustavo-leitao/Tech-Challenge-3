"""
Aplica o pipeline campeão já treinado (`models/pipeline_final.joblib`) ao snapshot mais recente
de municípios para responder diretamente a pergunta de negócio "quais municípios estão em maior
risco de não atingir a meta de alfabetização".

O modelo é aplicado à mesma safra de dados usada no treino/teste (2024) porque é a mais recente
disponível — a probabilidade prevista aqui simula como o modelo seria usado de forma prospectiva
(a partir do perfil do município, sem olhar a taxa de alfabetização real), e o resultado real de
2024 é mantido ao lado só para dar transparência de quão bem o ranking bateu com o que de fato
aconteceu, não como informação usada pelo modelo.

Uso:
    python src/modeling/predict_risk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.pipeline import FEATURE_COLUMNS  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "municipio_alfabetizacao_enriquecido.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline_final.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
SNAPSHOT_YEAR = 2024
TOP_N = 20

IDENTIFICATION_COLUMNS = [
    "id_municipio",
    "nome_municipio",
    "sigla_uf",
    "nome_regiao",
    "taxa_alfabetizacao",
    "meta_alfabetizacao_2024",
    "atingiu_meta",
]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    snapshot = df[df["ano"] == SNAPSHOT_YEAR].reset_index(drop=True)

    pipeline = joblib.load(MODEL_PATH)
    proba_atingir_meta = pipeline.predict_proba(snapshot[FEATURE_COLUMNS])[:, 1]

    ranking = snapshot[IDENTIFICATION_COLUMNS].copy()
    ranking["probabilidade_prevista_atingir_meta"] = proba_atingir_meta.round(4)
    ranking["gap_ate_meta_pp"] = (
        ranking["meta_alfabetizacao_2024"] - ranking["taxa_alfabetizacao"]
    ).round(2)
    ranking = ranking.sort_values("probabilidade_prevista_atingir_meta").reset_index(drop=True)
    ranking.insert(0, "ranking_risco", ranking.index + 1)

    REPORTS_DIR.mkdir(exist_ok=True)
    full_path = REPORTS_DIR / "ranking_risco_municipios.csv"
    ranking.to_csv(full_path, index=False, encoding="utf-8")

    top_n = ranking.head(TOP_N)
    print(f"=== Top {TOP_N} municípios de maior risco previsto (safra {SNAPSHOT_YEAR}) ===")
    print(
        top_n[
            [
                "ranking_risco",
                "nome_municipio",
                "sigla_uf",
                "probabilidade_prevista_atingir_meta",
                "atingiu_meta",
            ]
        ].to_string(index=False)
    )

    acerto_top_n = (top_n["atingiu_meta"] == 0).mean()
    print(f"\n% do Top {TOP_N} que de fato não atingiu a meta em {SNAPSHOT_YEAR}: {acerto_top_n:.0%}")
    print(f"\nRanking completo ({len(ranking)} municípios) salvo em {full_path}")


if __name__ == "__main__":
    main()

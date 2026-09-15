"""
Aplica o pipeline treinado só com 2023 (`models/pipeline_temporal_2023.joblib`) ao snapshot de
2024 completo para responder diretamente a pergunta de negócio "quais municípios estão em maior
risco de não atingir a meta de alfabetização".

Usa o pipeline temporal, não o campeão (`models/pipeline_final.joblib`), de propósito: o campeão é
treinado num split 80/20 que mistura linhas de 2023 e 2024, então a maior parte do snapshot de
2024 já teria sido vista por ele durante o treino — o "acerto" do ranking incluiria município que
o modelo já conhecia, não uma previsão de verdade. O pipeline temporal nunca viu nenhuma linha de
2024 (foi treinado só com 2023, ver `src/modeling/train_model.py`), então aplicá-lo ao snapshot de
2024 inteiro é uma previsão genuinamente fora da amostra — a mesma lógica já usada na validação
temporal, reaproveitada aqui para gerar o ranking.

O resultado real de 2024 é mantido ao lado só para dar transparência de quão bem o ranking bateu
com o que de fato aconteceu, não como informação usada pelo modelo.

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
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline_temporal_2023.joblib"
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

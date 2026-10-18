"""
BLOCO 2 — Inteligência Artificial: treina/usa a CNN e roda o classificador
híbrido, a partir das saídas do bloco de Processamento de Imagens.

Antes de tudo, verifica se o bloco de PI já rodou (data_processed/ e
data_split/split_manifest.csv); se não, para e pede para rodar run_pi.py.

Etapas:
    1. train_model         -> só se models/plant_disease_model.keras não existir,
                              ou com --treinar (leva ~1h30 em CPU)
    2. extract_features    -> features_cache/ (pulado se já estiver completo e a
                              CNN não tiver sido treinada de novo)
    3. hybrid_classifier   -> reports/ia/hibrido/ + models/svm_hybrid.joblib
    4. hybrid_controle     -> reports/ia/hibrido/controle_refit_treino/ (SVMs só no treino)
Etapas 2 a 4 são puladas se as saídas já existirem (exceto com --force ou se
uma etapa anterior tiver rodado de novo).

Uso:
    python src/run_ia.py                 -> roda o que falta
    python src/run_ia.py --treinar       -> treina a CNN de novo (e refaz as etapas seguintes)
    python src/run_ia.py --force         -> refaz extração e classificador híbrido
                                            (a CNN só é treinada de novo com --treinar)
    python src/run_ia.py --sem-hibrido   -> para depois da CNN

Gera:
    models/plant_disease_model.keras, models/class_names.json, reports/ia/cnn/,
    features_cache/ (handcrafted, embeddings, softmax), reports/ia/hibrido/ e
    models/svm_hybrid.joblib
"""

import argparse
import os
import sys

from config import (
    CNN_EMBEDDINGS_PATH, CNN_TEST_PRED_PATH, HANDCRAFTED_PATH, HYBRID_MODEL_PATH,
    HYBRID_CONTROL_REPORTS_DIR, HYBRID_REPORTS_DIR, IMAGE_EXTENSIONS, MODEL_PATH, PROCESSED_DIR,
    SPLIT_MANIFEST,
)
from orchestration import Step, go_to_project_root, run_steps

FEATURE_OUTPUTS = [HANDCRAFTED_PATH, CNN_EMBEDDINGS_PATH, CNN_TEST_PRED_PATH]
HYBRID_OUTPUTS = [os.path.join(HYBRID_REPORTS_DIR, "comparativo.csv"), HYBRID_MODEL_PATH]
CONTROL_OUTPUTS = [os.path.join(HYBRID_CONTROL_REPORTS_DIR, "comparativo.csv")]


def pi_outputs_exist():
    has_processed = os.path.isdir(PROCESSED_DIR) and any(
        f.lower().endswith(IMAGE_EXTENSIONS)
        for _, _, files in os.walk(PROCESSED_DIR) for f in files)
    return has_processed and os.path.exists(SPLIT_MANIFEST)


def build_steps(force, retrain, without_hybrid):
    train = retrain or not os.path.exists(MODEL_PATH)
    steps = [Step("train_model", "src/train_model.py",
                  skip_reason=None if train
                  else f"já existe ({MODEL_PATH}; use --treinar para treinar de novo)",
                  note="o treino da CNN leva ~1h30 em CPU")]
    if without_hybrid:
        return steps

    # Uma etapa só é pulada se as saídas existirem E nada antes dela tiver rodado de novo
    extract = force or train or not all(os.path.exists(p) for p in FEATURE_OUTPUTS)
    hybrid = extract or not all(os.path.exists(p) for p in HYBRID_OUTPUTS)
    control = hybrid or not all(os.path.exists(p) for p in CONTROL_OUTPUTS)
    steps += [
        Step("extract_features", "src/extract_features.py",
             skip_reason=None if extract else "já existe (features_cache/ completo)"),
        Step("hybrid_classifier", "src/hybrid_classifier.py",
             skip_reason=None if hybrid else f"já existe ({HYBRID_OUTPUTS[0]} e {HYBRID_MODEL_PATH})"),
        Step("hybrid_controle", "src/hybrid_classifier.py", ["--control-train-only"],
             skip_reason=None if control else f"já existe ({CONTROL_OUTPUTS[0]})"),
    ]
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--treinar", action="store_true",
                        help="treina a CNN mesmo que o modelo já exista (~1h30 em CPU)")
    parser.add_argument("--force", action="store_true",
                        help="refaz extração de features e classificador híbrido")
    parser.add_argument("--sem-hibrido", action="store_true",
                        help="para depois da CNN (não extrai features nem roda os SVMs)")
    args = parser.parse_args()
    go_to_project_root()

    if not pi_outputs_exist():
        print("Rode primeiro o bloco de Processamento de Imagens: python src/run_pi.py")
        sys.exit(1)

    sys.exit(run_steps("Bloco 2: Inteligência Artificial",
                       build_steps(args.force, args.treinar, args.sem_hibrido)))


if __name__ == "__main__":
    main()

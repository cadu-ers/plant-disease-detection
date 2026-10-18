"""
BLOCO 1 — Processamento de Imagens: roda todas as etapas de PI em ordem.
Este bloco é independente: não usa nada do bloco de IA (não importa tensorflow,
não lê models/ nem features_cache/, exceto o fft_features.csv que o próprio
fft_analysis.py gera).

Etapas:
    1. download_dataset      (só se data/ não tiver imagens)
    2. preprocessing         -> data_processed/
    3. split_dataset         -> data_split/ + split_manifest.csv
    4. data_overview         -> reports/pi/dados/
    5. segmentation_report   -> reports/pi/segmentacao/ (etapas, qualidade, suspeitos)
    6. segmentation_report --comparacao -> reports/pi/segmentacao/segmentation_comparison.png
    7. fft_analysis          -> reports/pi/fft/ + features_cache/fft_features.csv
    8. histogram_analysis    -> reports/pi/p1_histograma/ (material da P1)
Cada etapa é pulada se as suas saídas já existirem ("[pulado] já existe").

Uso:
    python src/run_pi.py                -> roda o que falta
    python src/run_pi.py --force        -> refaz tudo (o download continua só com data/ vazia)
    python src/run_pi.py --so-figuras   -> regera só as figuras da apresentação
                                           (data_overview, segmentation_report, fft_analysis);
                                           exige data_processed/ e data_split/

Gera:
    data/, data_processed/, data_split/, reports/pi/ (dados, segmentacao, fft,
    p1_histograma) e features_cache/fft_features.csv
"""

import argparse
import os
import sys

from config import (
    DATA_DIR, DATA_REPORTS_DIR, FFT_FEATURES_PATH, FFT_REPORTS_DIR, IMAGE_EXTENSIONS,
    P1_REPORTS_DIR, PROCESSED_DIR, SEGMENTATION_QUALITY_PATH, SEGMENTATION_REPORTS_DIR,
    SPLIT_MANIFEST,
)
from orchestration import Step, go_to_project_root, run_steps

SEGMENTATION_OUTPUTS = [os.path.join(SEGMENTATION_REPORTS_DIR, f) for f in (
    "etapas_segmentacao.png", "qualidade_segmentacao.png", "casos_suspeitos.png")] + [
    SEGMENTATION_QUALITY_PATH]
COMPARISON_OUTPUT = os.path.join(SEGMENTATION_REPORTS_DIR, "segmentation_comparison.png")
DATA_OUTPUTS = [os.path.join(DATA_REPORTS_DIR, f) for f in (
    "exemplos_por_classe.png", "resumo_dados.csv")]
FFT_OUTPUTS = [os.path.join(FFT_REPORTS_DIR, f) for f in (
    "fft_saudavel_vs_doente.png", "fft_magnitude_vs_fase.png",
    "blur_espaco_vs_frequencia.png", "espectro_radial_por_classe.png")] + [FFT_FEATURES_PATH]
HISTOGRAM_OUTPUT = os.path.join(P1_REPORTS_DIR, "histograma_exemplo.png")


def has_images(root):
    """True se alguma subpasta de root tiver pelo menos uma imagem."""
    if not os.path.isdir(root):
        return False
    for current, _, files in os.walk(root):
        if current != root and any(f.lower().endswith(IMAGE_EXTENSIONS) for f in files):
            return True
    return False


def skip_if_exists(paths, force):
    """Motivo para pular, se todas as saídas existirem (e não for --force)."""
    if force:
        return None
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        return None
    shown = paths[0] + (f" e mais {len(paths) - 1}" if len(paths) > 1 else "")
    return f"já existe ({shown})"


def figure_steps():
    return [
        Step("data_overview", "src/data_overview.py"),
        Step("segmentation_report", "src/segmentation_report.py"),
        Step("fft_analysis", "src/fft_analysis.py"),
    ]


def all_steps(force):
    return [
        Step("download_dataset", "src/download_dataset.py",
             skip_reason=f"já existe ({DATA_DIR}/ tem imagens)" if has_images(DATA_DIR) else None),
        Step("preprocessing", "src/preprocessing.py",
             skip_reason=None if force or not has_images(PROCESSED_DIR)
             else f"já existe ({PROCESSED_DIR}/ tem imagens)"),
        Step("split_dataset", "src/split_dataset.py",
             skip_reason=skip_if_exists([SPLIT_MANIFEST], force)),
        Step("data_overview", "src/data_overview.py",
             skip_reason=skip_if_exists(DATA_OUTPUTS, force)),
        Step("segmentation_report", "src/segmentation_report.py",
             skip_reason=skip_if_exists(SEGMENTATION_OUTPUTS, force)),
        Step("segmentation_comparacao", "src/segmentation_report.py", ["--comparacao"],
             skip_reason=skip_if_exists([COMPARISON_OUTPUT], force)),
        Step("fft_analysis", "src/fft_analysis.py",
             skip_reason=skip_if_exists(FFT_OUTPUTS, force)),
        Step("histogram_analysis", "src/histogram_analysis.py",
             skip_reason=skip_if_exists([HISTOGRAM_OUTPUT], force)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="refaz todas as etapas (o download continua só com data/ vazia)")
    parser.add_argument("--so-figuras", action="store_true",
                        help="regera só data_overview, segmentation_report e fft_analysis")
    args = parser.parse_args()
    go_to_project_root()

    if args.so_figuras:
        if not has_images(PROCESSED_DIR) or not os.path.exists(SPLIT_MANIFEST):
            print(f"--so-figuras precisa de {PROCESSED_DIR}/ e {SPLIT_MANIFEST}. "
                  "Rode antes: python src/run_pi.py")
            sys.exit(1)
        sys.exit(run_steps("Bloco 1: Processamento de Imagens (só figuras)", figure_steps()))

    sys.exit(run_steps("Bloco 1: Processamento de Imagens", all_steps(args.force)))


if __name__ == "__main__":
    main()

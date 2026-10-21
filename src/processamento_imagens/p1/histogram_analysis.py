"""
Script de documentação da P1: gera um gráfico comparando o histograma de
intensidade de uma imagem original com o de sua versão normalizada.

Material da P1, mantido só como registro. A normalização de intensidade
min-max mostrada aqui NÃO faz parte do pipeline final: o preprocessing.py
não normaliza intensidade (as etapas são resize, blur, BGR->HSV e
segmentação). Não é usado por nenhum outro script.

Uso (na raiz do repositório, com src/ no PYTHONPATH: no Linux "export PYTHONPATH=src",
no PowerShell '$env:PYTHONPATH = "src"'):
    python -m processamento_imagens.p1.histogram_analysis                      -> imagem aleatória de data/
    python -m processamento_imagens.p1.histogram_analysis caminho/da/imagem.jpg -> imagem específica

Gera:
    reports/pi/p1_histograma/histograma_exemplo.png
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

from common.config import DATA_DIR, P1_REPORTS_DIR
from common.utils import pick_random_image

OUTPUT_PATH = os.path.join(P1_REPORTS_DIR, "histograma_exemplo.png")


def compute_histogram(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return hist.flatten()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else pick_random_image(DATA_DIR)[0]
    print(f"Imagem usada: {img_path}")

    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalização de intensidade (min-max stretching para 0-255)
    img_normalized = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)

    hist_original = compute_histogram(img_gray)
    hist_normalized = compute_histogram(img_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(hist_original, color="black")
    axes[0].set_title("Histograma - Imagem Original")
    axes[0].set_xlabel("Intensidade do pixel")
    axes[0].set_ylabel("Número de pixels")

    axes[1].plot(hist_normalized, color="blue")
    axes[1].set_title("Histograma - Imagem Normalizada")
    axes[1].set_xlabel("Intensidade do pixel")
    axes[1].set_ylabel("Número de pixels")

    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Gráfico salvo em: {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
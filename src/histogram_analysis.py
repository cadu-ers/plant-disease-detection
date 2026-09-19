"""
Script de documentação: gera um gráfico comparando o histograma de
intensidade de uma imagem original com o de sua versão normalizada.

Não faz parte do pipeline principal — é só um material de apoio pra
documentação/apresentação, mostrando o efeito da normalização de
intensidade numa imagem de exemplo.

Uso:
    python src/histogram_analysis.py                      -> imagem aleatória de data/
    python src/histogram_analysis.py caminho/da/imagem.jpg -> imagem específica
"""

import sys
import random
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_PATH = "reports/histograma_exemplo.png"


def pick_random_image():
    classes = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))]
    class_name = random.choice(classes)
    class_dir = os.path.join(DATA_DIR, class_name)
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_name = random.choice(images)
    return os.path.join(class_dir, img_name)


def compute_histogram(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return hist.flatten()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else pick_random_image()
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
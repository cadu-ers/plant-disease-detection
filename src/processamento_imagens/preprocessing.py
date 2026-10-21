"""
Pipeline de pré-processamento: segmenta as folhas do dataset e salva as
imagens segmentadas, que alimentam o treino da CNN.

Etapas (na ordem em que são executadas):
    1. Leitura e resize
    2. Suavização / remoção de ruído (Gaussian Blur)
    3. Conversão de espaço de cor (BGR -> HSV; o OpenCV lê as imagens em BGR)
    4. Segmentação da folha (máscara verde + lesão, contorno externo preenchido)

A extração de features clássicas fica em src/processamento_imagens/features.py.

Uso (na raiz do repositório, com src/ no PYTHONPATH: no Linux "export PYTHONPATH=src",
no PowerShell '$env:PYTHONPATH = "src"'):
    python -m processamento_imagens.preprocessing
    (requer o dataset em data/<NomeDaClasse>/*.jpg)

Gera:
    data_processed/<NomeDaClasse>/*.jpg   -> imagens segmentadas, prontas pro treino da CNN
"""

import os

import cv2
import numpy as np

from common.config import (
    DATA_DIR, PROCESSED_DIR, SEG_IMG_SIZE,
    LOWER_GREEN, UPPER_GREEN, LOWER_LESION, UPPER_LESION, MIN_LEAF_AREA_RATIO,
)
from common.utils import list_classes, list_images


def load_and_resize(path, size=SEG_IMG_SIZE):
    """Etapa 1: leitura e redimensionamento."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    return img


def denoise(img):
    """Etapa 2: suavização / remoção de ruído."""
    return cv2.GaussianBlur(img, (5, 5), 0)


def to_hsv(img):
    """Etapa 3: conversão de espaço de cor BGR -> HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def segment_leaf(img_bgr, img_hsv, lower_lesion=None):
    """Etapa 4: segmentação da folha (máscara verde + lesão, contorno externo preenchido).

    lower_lesion: limite inferior [H, S, V] da máscara de lesão; se None,
    usa config.LOWER_LESION.
    """
    if lower_lesion is None:
        lower_lesion = LOWER_LESION

    # a) Máscara de verde unida à máscara de tons de lesão (marrom/amarelo)
    green_mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)
    lesion_mask = cv2.inRange(img_hsv, lower_lesion, UPPER_LESION)
    mask = cv2.bitwise_or(green_mask, lesion_mask)

    # b) Limpeza morfológica (abertura remove ruído pequeno, fechamento fecha buracos)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # c) Mantém só o maior contorno externo (a folha) e preenche o interior,
    #    para que tudo dentro da folha (lesões incluídas) entre na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_LEAF_AREA_RATIO * mask.shape[0] * mask.shape[1]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # d) Fallback: se o maior contorno for pequeno demais, usa a máscara do passo b
        if cv2.contourArea(largest) >= min_area:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
            mask = filled

    # e) Aplica a máscara na imagem
    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return segmented, mask


def segment_image(path, lower_lesion=None):
    """Executa as etapas 1–4 em uma imagem e retorna (original, segmented, mask).

    `original` é a imagem redimensionada, antes da suavização.
    lower_lesion é repassado para segment_leaf (None = config.LOWER_LESION).
    Retorna None se a imagem não puder ser lida.
    """
    original = load_and_resize(path)
    if original is None:
        return None
    img = denoise(original)
    img_hsv = to_hsv(img)
    segmented, mask = segment_leaf(img, img_hsv, lower_lesion)
    return original, segmented, mask


def process_dataset():
    total = 0
    for class_name in list_classes(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        out_class_dir = os.path.join(PROCESSED_DIR, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        images = list_images(class_dir)
        n = len(images)
        saved = 0
        for i, fname in enumerate(images, start=1):
            result = segment_image(os.path.join(class_dir, fname))
            if result is not None:
                _, segmented, _ = result
                cv2.imwrite(os.path.join(out_class_dir, fname), segmented)
                saved += 1
            print(f"\r{class_name}: {i}/{n}", end="", flush=True)

        print(f"\r{class_name}: {saved}/{n} imagens segmentadas")
        total += saved

    print(f"\nConcluído. {total} imagens processadas.")
    print(f"Imagens segmentadas em: {PROCESSED_DIR}/")


if __name__ == "__main__":
    process_dataset()

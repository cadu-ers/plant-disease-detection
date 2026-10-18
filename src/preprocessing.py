"""
Pipeline de pré-processamento de imagens para detecção de doenças em folhas.

Etapas:
    1. Leitura e resize
    2. Conversão de espaço de cor (RGB -> HSV)
    3. Suavização / remoção de ruído (Gaussian Blur)
    4. Segmentação da folha (máscara de cor em HSV)
    5. Extração de features (histograma de cor + textura GLCM)

Uso:
    python src/preprocessing.py

Requer que o dataset esteja em: data/<NomeDaClasse>/*.jpg
Gera:
    data_processed/<NomeDaClasse>/*.jpg   -> imagens segmentadas, prontas pro treino da CNN
    features.csv                          -> features clássicas (cor + textura) por imagem, opcional
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# ---------- Configurações ----------
DATA_DIR = "data"
OUTPUT_DIR = "data_processed"
IMG_SIZE = (224, 224)

# Faixa de verde em HSV, usada para separar a folha do fundo.
# Ajuste esses valores se o fundo do seu dataset não for removido corretamente.
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])

# Faixa de tons de lesão (manchas marrons/amarelas) em HSV.
# As lesões das doenças têm matiz abaixo do verde (H ~8–35) e ficavam fora da
# máscara de verde, virando "buracos" pretos.
# S >= 40 segura o fundo bege/cinza do PlantVillage (baixa saturação, mas matiz
# na faixa do marrom); com S menor o fundo passa e o contorno preenchido o engole.
# V >= 20 recupera as lesões escuras na borda da folha, que falhavam com V >= 30.
LOWER_LESION = np.array([8, 40, 20])
UPPER_LESION = np.array([35, 255, 255])

# Área mínima (fração da imagem) para aceitar o maior contorno como a folha.
MIN_LEAF_AREA_RATIO = 0.05


def load_and_resize(path, size=IMG_SIZE):
    """Etapa 1: leitura e redimensionamento."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    return img


def denoise(img):
    """Etapa 3: suavização / remoção de ruído."""
    return cv2.GaussianBlur(img, (5, 5), 0)


def to_hsv(img):
    """Etapa 2: conversão de espaço de cor RGB -> HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def segment_leaf(img_bgr, img_hsv):
    """Etapa 4: segmentação da folha (máscara verde + lesão, contorno externo preenchido)."""
    # a) Máscara de verde unida à máscara de tons de lesão (marrom/amarelo)
    green_mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)
    lesion_mask = cv2.inRange(img_hsv, LOWER_LESION, UPPER_LESION)
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


def segment_image(path):
    """Executa as etapas 1–4 em uma imagem e retorna (original, segmented, mask).

    `original` é a imagem redimensionada, antes da suavização.
    Retorna None se a imagem não puder ser lida.
    """
    original = load_and_resize(path)
    if original is None:
        return None
    img = denoise(original)
    img_hsv = to_hsv(img)
    segmented, mask = segment_leaf(img, img_hsv)
    return original, segmented, mask


def extract_color_histogram(img_hsv, bins=8):
    """Etapa 5a: histograma de cor (H, S, V) normalizado."""
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [bins, bins, bins],
                         [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_texture_features(img_bgr):
    """Etapa 5b: features de textura via GLCM (matriz de co-ocorrência)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                         symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }


def process_dataset():
    records = []
    classes = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))]

    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        out_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"Processando classe '{class_name}' ({len(images)} imagens)...")

        for fname in images:
            path = os.path.join(class_dir, fname)

            img = load_and_resize(path)
            if img is None:
                continue

            img = denoise(img)
            img_hsv = to_hsv(img)
            segmented, mask = segment_leaf(img, img_hsv)

            # Salva a imagem segmentada (entrada da CNN)
            out_path = os.path.join(out_class_dir, fname)
            cv2.imwrite(out_path, segmented)

            # Extrai features clássicas (opcional, útil pra análise/relatório)
            color_hist = extract_color_histogram(img_hsv)
            texture = extract_texture_features(segmented)

            record = {"filename": fname, "class": class_name}
            record.update(texture)
            for i, val in enumerate(color_hist):
                record[f"hist_{i}"] = val
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv("features.csv", index=False)
    print(f"\nConcluído. {len(df)} imagens processadas.")
    print(f"Imagens segmentadas em: {OUTPUT_DIR}/")
    print(f"Features salvas em: features.csv")


if __name__ == "__main__":
    process_dataset()

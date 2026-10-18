"""
Extração de features clássicas (histograma de cor + textura GLCM) das imagens.
Este módulo será usado pelo classificador híbrido (features clássicas + CNN).

TODO (problemas conhecidos, a corrigir no próximo checkpoint):
    1. O histograma de cor usa a imagem inteira, com fundo; deveria usar só
       os pixels da folha (máscara da segmentação).
    2. O GLCM é calculado com o fundo preto incluído (distorce contraste,
       homogeneidade etc.) e com 256 níveis de cinza (lento).

Uso:
    from features import extract_color_histogram, extract_texture_features

Gera:
    nada (módulo de funções; quem chama decide onde salvar as features)
"""

import cv2
from skimage.feature import graycomatrix, graycoprops


def extract_color_histogram(img_hsv, bins=8):
    """Histograma de cor (H, S, V) normalizado."""
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [bins, bins, bins],
                         [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_texture_features(img_bgr):
    """Features de textura via GLCM (matriz de co-ocorrência)."""
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

"""
Extração de features clássicas ("handcrafted") calculadas SÓ NA FOLHA, usando a
máscara da segmentação. Usadas pelo classificador híbrido (features clássicas +
embeddings da CNN -> SVM).

Grupos de features (147 no total):
    a. Cor (82): histogramas H/S/V e momentos de cor em HSV e Lab.
    b. Proporção de lesão (1): fração da folha na faixa de tons de lesão.
    c. Textura GLCM (36): 6 propriedades x 3 distâncias x (média, desvio entre ângulos).
    d. Textura LBP (28): histogramas LBP uniformes (P=8,R=1) e (P=16,R=2).

Uso:
    from features import extract_handcrafted, feature_names
    feats = extract_handcrafted(original_bgr_224, segmented, mask)

Gera:
    nada (módulo de funções; extract_features.py salva as features em disco)
"""

import warnings

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from config import LOWER_LESION, UPPER_LESION

# ---------- Parâmetros ----------
H_BINS, S_BINS, V_BINS = 32, 16, 16
GLCM_GRAY_LEVELS = 32                            # folha em 1..32, fundo em 0
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 graus
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
LBP_CONFIGS = [(8, 1), (16, 2)]                  # (P vizinhos, raio R)

MOMENT_CHANNELS = ["hsv_h", "hsv_s", "hsv_v", "lab_l", "lab_a", "lab_b"]


def feature_names():
    """Nomes das features na ordem em que extract_handcrafted as devolve."""
    names = []
    names += [f"h_hist_{i:02d}" for i in range(H_BINS)]
    names += [f"s_hist_{i:02d}" for i in range(S_BINS)]
    names += [f"v_hist_{i:02d}" for i in range(V_BINS)]
    names += [f"{ch}_{stat}" for ch in MOMENT_CHANNELS for stat in ("mean", "std", "skew")]
    names += ["lesion_ratio"]
    names += [f"glcm_{prop}_d{d}_{stat}"
              for prop in GLCM_PROPS for d in GLCM_DISTANCES for stat in ("mean", "std")]
    names += [f"lbp_p{p}r{r}_{i:02d}" for p, r in LBP_CONFIGS for i in range(p + 2)]
    return names


def _normalized_hist(channel, leaf, bins, value_range):
    hist, _ = np.histogram(channel[leaf], bins=bins, range=value_range)
    return hist / max(hist.sum(), 1)


def color_features(original_bgr, leaf):
    """a. Cor: distribuição de matiz/saturação/brilho da folha (amarelamento,
    manchas, palidez) e momentos de cor (média, desvio, assimetria) em HSV e Lab."""
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)

    feats = []
    feats += list(_normalized_hist(hsv[..., 0], leaf, H_BINS, (0, 180)))
    feats += list(_normalized_hist(hsv[..., 1], leaf, S_BINS, (0, 256)))
    feats += list(_normalized_hist(hsv[..., 2], leaf, V_BINS, (0, 256)))

    for img in (hsv, lab):
        pixels = img[leaf].astype(np.float64)        # (n_pixels_folha, 3)
        for c in range(3):
            values = pixels[:, c]
            # skew de um canal constante é NaN -> 0
            feats += [values.mean(), values.std(), np.nan_to_num(skew(values))]
    return feats


def lesion_features(segmented_bgr, leaf):
    """b. Proporção de lesão: fração dos pixels da folha nos tons de lesão
    (marrom/amarelo), ou seja, quanto da folha está manchada."""
    # Mesma imagem (suavizada) e mesma faixa usadas na segmentação
    hsv = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2HSV)
    lesion = cv2.inRange(hsv, LOWER_LESION, UPPER_LESION) > 0
    return [float((lesion & leaf).sum() / leaf.sum())]


def glcm_features(original_bgr, leaf):
    """c. Textura GLCM: como os tons de cinza vizinhos se relacionam na folha
    (rugosidade, uniformidade, manchas granuladas), invariante à rotação."""
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    # Quantiza a folha em 1..32 e deixa o fundo em 0
    quantized = (gray.astype(np.int32) * GLCM_GRAY_LEVELS // 256 + 1).astype(np.uint8)
    quantized[~leaf] = 0

    glcm = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=GLCM_GRAY_LEVELS + 1, symmetric=True, normed=False)
    # Descarta a linha/coluna 0 (pares que envolvem o fundo) e renormaliza
    glcm = glcm[1:, 1:, :, :].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, totals, out=np.zeros_like(glcm), where=totals > 0)

    feats = []
    for prop in GLCM_PROPS:
        values = np.nan_to_num(graycoprops(glcm, prop))  # (distâncias, ângulos)
        for i in range(len(GLCM_DISTANCES)):
            feats += [values[i].mean(), values[i].std()]
    return feats


def lbp_features(original_bgr, leaf):
    """d. Textura LBP: padrões locais de micro-textura (bordas, pontos, manchas
    finas) na folha, com histogramas uniformes em duas escalas."""
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    feats = []
    for p, r in LBP_CONFIGS:
        lbp = local_binary_pattern(gray, p, r, method="uniform").astype(np.int64)
        # Erode a folha pelo raio do LBP, para não contar padrões da borda folha/fundo
        kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        inner = cv2.erode(leaf.astype(np.uint8), kernel) > 0
        region = inner if inner.any() else leaf
        hist = np.bincount(lbp[region], minlength=p + 2)[:p + 2]
        feats += list(hist / max(hist.sum(), 1))
    return feats


def extract_handcrafted(original_bgr_224, segmented, mask):
    """Extrai todas as features clássicas da folha (pixels com mask > 0).

    original_bgr_224: imagem redimensionada, sem blur (cor e textura finas).
    segmented: imagem suavizada e segmentada (usada na proporção de lesão,
               a mesma imagem de onde a máscara saiu).
    mask: máscara da segmentação (uint8, 0 = fundo).
    Retorna um dict {nome_da_feature: valor} na ordem de feature_names().
    Folha vazia (mask toda zero): todas as features = 0, com um aviso.
    """
    names = feature_names()
    leaf = mask > 0
    if not leaf.any():
        warnings.warn("Máscara vazia: nenhuma folha segmentada; features = 0.")
        return dict.fromkeys(names, 0.0)

    values = (color_features(original_bgr_224, leaf)
              + lesion_features(segmented, leaf)
              + glcm_features(original_bgr_224, leaf)
              + lbp_features(original_bgr_224, leaf))
    assert len(values) == len(names)
    return {name: float(v) for name, v in zip(names, values)}

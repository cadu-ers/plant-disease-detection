"""
Medidas visuais da folha para o SISTEMA ESPECIALISTA.

Fica separado de features.py de propósito: as 147 features clássicas do SVM não
mudam. Aqui as medidas são proporções de pixels por FAIXA DE COR dentro da
máscara da segmentação, e NÃO uma segmentação de lesão validada (não existem
máscaras de referência para conferir). As faixas vêm de common/config.py.

Medidas (calculadas sobre os pixels da máscara da folha):
    frac_lesao_total         fração da folha fora do "verde saudável" (H 36-90):
                             inclui marrom, amarelo e pixels pálidos ou escuros
    frac_marrom              fração da folha na faixa marrom (H 8-24)
    frac_amarelo             fração da folha na faixa amarela (H 25-35); folhas
                             verde-claras saudáveis também podem pontuar aqui
    n_lesoes                 componentes conexos marrons com ao menos
                             MIN_LESION_AREA_PX pixels
    n_lesoes_pequenas        dessas, as com área até SMALL_LESION_MAX_RATIO da folha
    dist_borda_media_lesoes  distância média dos pixels de lesão marrom à borda da
                             folha, normalizada em [0, 1] (0 = na borda, 1 = no
                             ponto mais interno); None quando não há lesão marrom

A medição do halo amarelo foi ADIADA de propósito: seria uma heurística sem
referência para validar.

Limitações conhecidas:
    - A 224x224, lesões muito pequenas (septoriose, mancha bacteriana) podem
      desaparecer na suavização ou ficar abaixo de MIN_LESION_AREA_PX.
    - Sombras e regiões escuras entram em frac_lesao_total.
    - O blur faz a transição verde -> marrom passar pela faixa amarela, então cada
      lesão tem um pequeno anel amarelo ao redor.
    - segment_image não informa se usou o fallback da máscara de cor; isso não é
      detectado aqui.
    - Não houve verificação com imagens reais do dataset; os testes usam folhas
      sintéticas.

Uso:
    from processamento_imagens.preprocessing import segment_image
    from processamento_imagens.medidas_lesao import avaliar_mascara, medir_lesoes
    original, segmented, mask = segment_image(caminho)
    if avaliar_mascara(mask).confiavel:
        medidas = medir_lesoes(segmented, mask)

Gera:
    nada (módulo de funções; o pipeline em aplicacao/ usa estes resultados)
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from common.config import (
    LOWER_BROWN, LOWER_HEALTHY_GREEN, LOWER_YELLOW, MIN_LESION_AREA_PX,
    SMALL_LESION_MAX_RATIO, SUSPECT_LEAF_FRACTION, UPPER_BROWN, UPPER_HEALTHY_GREEN,
    UPPER_YELLOW,
)


class MascaraInvalida(ValueError):
    """A máscara não permite medir a folha (vazia ou suspeita)."""


@dataclass(frozen=True)
class QualidadeMascara:
    leaf_fraction: float          # fração da imagem ocupada pela folha
    vazia: bool
    suspeita: bool
    motivo: Optional[str]         # texto para o usuário; None se a máscara é confiável

    @property
    def confiavel(self):
        return not (self.vazia or self.suspeita)


@dataclass(frozen=True)
class MedidasLesao:
    frac_lesao_total: float
    frac_marrom: float
    frac_amarelo: float
    n_lesoes: int
    n_lesoes_pequenas: int
    dist_borda_media_lesoes: Optional[float]   # None quando n_lesoes == 0
    leaf_fraction: float


def avaliar_mascara(mask):
    """Diz se a máscara da folha (uint8, 0 = fundo) é confiável para medir.

    É um indicador indireto: a máscara é vazia, ou a folha ocupa uma fração da
    imagem fora da faixa usual (possível falha de segmentação ou vazamento do fundo).
    Não é uma medida de qualidade da segmentação.
    """
    if mask is None or mask.size == 0 or not (mask > 0).any():
        return QualidadeMascara(0.0, True, False, "máscara vazia: nenhuma folha segmentada")
    fraction = float((mask > 0).mean())
    low, high = SUSPECT_LEAF_FRACTION
    if fraction < low:
        return QualidadeMascara(
            fraction, False, True,
            f"a folha ocupa só {fraction:.0%} da imagem (suspeito: menos de {low:.0%})")
    if fraction > high:
        return QualidadeMascara(
            fraction, False, True,
            f"a folha ocupa {fraction:.0%} da imagem (suspeito: mais de {high:.0%}); "
            "possível vazamento do fundo")
    return QualidadeMascara(fraction, False, False, None)


def _in_range(hsv, lower, upper):
    return cv2.inRange(hsv, lower, upper) > 0


def medir_lesoes(segmented_bgr, mask):
    """Calcula as medidas de lesão da folha.

    segmented_bgr: imagem suavizada e segmentada devolvida por segment_image (a mesma
                   de onde saiu a máscara); mask: máscara da folha (uint8, 0 = fundo).
    Levanta MascaraInvalida se a máscara não for confiável e ValueError se os
    tamanhos da imagem e da máscara forem diferentes. Não altera os argumentos.
    """
    if segmented_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("imagem segmentada e máscara têm tamanhos diferentes: "
                         f"{segmented_bgr.shape[:2]} x {mask.shape[:2]}")
    quality = avaliar_mascara(mask)
    if not quality.confiavel:
        raise MascaraInvalida(quality.motivo)

    leaf = (mask > 0).astype(np.uint8)
    in_leaf = leaf > 0
    n_leaf = int(in_leaf.sum())

    hsv = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2HSV)
    healthy_green = _in_range(hsv, LOWER_HEALTHY_GREEN, UPPER_HEALTHY_GREEN) & in_leaf
    brown = _in_range(hsv, LOWER_BROWN, UPPER_BROWN) & in_leaf
    yellow = _in_range(hsv, LOWER_YELLOW, UPPER_YELLOW) & in_leaf

    # Lesões marrons: componentes conexos com o tamanho mínimo
    _, labels, stats, _ = cv2.connectedComponentsWithStats(brown.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]                     # o rótulo 0 é o fundo
    kept = np.flatnonzero(areas >= MIN_LESION_AREA_PX) + 1  # rótulos das lesões mantidas
    small_limit = SMALL_LESION_MAX_RATIO * n_leaf
    n_small = int((areas[kept - 1] <= small_limit).sum()) if len(kept) else 0

    # Distância à borda da folha, relativa ao ponto mais interno (1 = centro)
    dist_border = None
    if len(kept):
        dist = cv2.distanceTransform(leaf, cv2.DIST_L2, 5)
        lesion_pixels = np.isin(labels, kept)
        dist_border = float((dist[lesion_pixels] / dist.max()).mean())

    return MedidasLesao(
        frac_lesao_total=float((in_leaf & ~healthy_green).sum() / n_leaf),
        frac_marrom=float(brown.sum() / n_leaf),
        frac_amarelo=float(yellow.sum() / n_leaf),
        n_lesoes=int(len(kept)),
        n_lesoes_pequenas=n_small,
        dist_borda_media_lesoes=dist_border,
        leaf_fraction=quality.leaf_fraction,
    )

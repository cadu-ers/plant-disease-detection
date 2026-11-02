"""Testes das medidas de lesão (processamento_imagens/medidas_lesao.py).

Usam folhas SINTÉTICAS desenhadas com OpenCV e passam pela segmentação real do
projeto (segment_image): não precisam de dataset, de modelo nem de TensorFlow.
Os testes verificam o comportamento do código, não a qualidade da segmentação em
fotos reais, que não foi avaliada.
"""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from common import config
from processamento_imagens.medidas_lesao import (
    MascaraInvalida, avaliar_mascara, medir_lesoes,
)
from processamento_imagens.preprocessing import segment_image

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cores em BGR (OpenCV)
VERDE, MARROM, AMARELO = (40, 160, 60), (30, 70, 120), (0, 220, 230)
FUNDO_NEUTRO, FUNDO_QUENTE = (150, 150, 150), (140, 150, 160)


def desenhar_folha(manchas=(), amarelos=(), fundo=FUNDO_NEUTRO):
    """Elipse verde sobre um fundo; manchas e amarelos são listas de (x, y, raio)."""
    img = np.full((224, 224, 3), fundo, np.uint8)
    cv2.ellipse(img, (112, 112), (85, 70), 0, 0, 360, VERDE, -1)
    for x, y, r in manchas:
        cv2.circle(img, (x, y), r, MARROM, -1)
    for x, y, r in amarelos:
        cv2.circle(img, (x, y), r, AMARELO, -1)
    return img


def medir(tmp_path, img):
    """Grava a imagem, segmenta com o código real do projeto e mede."""
    caminho = str(tmp_path / "folha.png")
    assert cv2.imwrite(caminho, img)
    resultado = segment_image(caminho)
    assert resultado is not None
    _, segmented, mask = resultado
    return medir_lesoes(segmented, mask)


# ---------------------------------------------------------------- folhas sintéticas
def test_folha_verde_nao_tem_lesao(tmp_path):
    m = medir(tmp_path, desenhar_folha())
    assert m.frac_lesao_total < 0.005
    assert m.frac_marrom == 0.0 and m.n_lesoes == 0 and m.n_lesoes_pequenas == 0
    assert m.dist_borda_media_lesoes is None          # sem lesão: não há distância a medir


@pytest.mark.parametrize("fundo", [FUNDO_NEUTRO, FUNDO_QUENTE])
def test_borda_folha_fundo_nao_vira_lesao(tmp_path, fundo):
    # A transição verde -> fundo pelo blur não pode ser contada como lesão
    m = medir(tmp_path, desenhar_folha(fundo=fundo))
    assert m.frac_lesao_total < 0.005 and m.n_lesoes == 0


def test_tres_manchas_pequenas(tmp_path):
    m = medir(tmp_path, desenhar_folha(manchas=[(95, 105, 6), (112, 112, 6), (130, 120, 6)]))
    assert m.n_lesoes == 3 and m.n_lesoes_pequenas == 3
    assert 0.005 < m.frac_marrom < 0.03
    assert m.frac_lesao_total >= m.frac_marrom


def test_mancha_grande_nao_e_pequena(tmp_path):
    m = medir(tmp_path, desenhar_folha(manchas=[(112, 112, 30)]))
    assert m.n_lesoes == 1 and m.n_lesoes_pequenas == 0
    assert m.frac_marrom > 0.08


def test_mistura_de_tamanhos(tmp_path):
    m = medir(tmp_path, desenhar_folha(manchas=[(112, 112, 28), (60, 100, 4), (160, 120, 4)]))
    assert m.n_lesoes == 3 and m.n_lesoes_pequenas == 2


def test_muitas_manchas_pequenas(tmp_path):
    manchas = [(70 + i * 12, 100 + (i % 3) * 10, 3) for i in range(8)]
    m = medir(tmp_path, desenhar_folha(manchas=manchas))
    assert m.n_lesoes >= 6                             # pode fundir/perder alguma na suavização
    assert m.n_lesoes_pequenas == m.n_lesoes


def test_filtro_de_area_minima_da_lesao(tmp_path):
    # Raio 2 deixa, após a suavização, um componente marrom de ~1 pixel: abaixo de
    # MIN_LESION_AREA_PX, é ruído e não conta como lesão
    ruido = medir(tmp_path, desenhar_folha(manchas=[(112, 112, 2)]))
    assert ruido.n_lesoes == 0
    # Raio 3 deixa ~13 pixels: acima do mínimo, conta como uma lesão pequena
    pequena = medir(tmp_path, desenhar_folha(manchas=[(112, 112, 3)]))
    assert pequena.n_lesoes == 1 and pequena.n_lesoes_pequenas == 1


def test_distancia_a_borda_separa_borda_de_centro(tmp_path):
    na_borda = medir(tmp_path, desenhar_folha(manchas=[(38, 112, 8)]))
    no_centro = medir(tmp_path, desenhar_folha(manchas=[(112, 112, 8)]))
    assert na_borda.dist_borda_media_lesoes < 0.35
    assert no_centro.dist_borda_media_lesoes > 0.80
    assert na_borda.dist_borda_media_lesoes < no_centro.dist_borda_media_lesoes


def test_amarelo_nao_e_contado_como_marrom(tmp_path):
    m = medir(tmp_path, desenhar_folha(amarelos=[(112, 112, 25)]))
    assert m.frac_amarelo > 0.05
    assert m.frac_marrom < 0.005 and m.n_lesoes == 0 and m.dist_borda_media_lesoes is None
    assert m.frac_lesao_total >= m.frac_amarelo


def test_invariantes_das_fracoes(tmp_path):
    img = desenhar_folha(manchas=[(95, 105, 9), (130, 120, 6)], amarelos=[(150, 90, 12)])
    m = medir(tmp_path, img)
    for valor in (m.frac_lesao_total, m.frac_marrom, m.frac_amarelo, m.leaf_fraction):
        assert 0.0 <= valor <= 1.0
    # marrom e amarelo são faixas de matiz disjuntas, ambas dentro da lesão total
    assert m.frac_marrom + m.frac_amarelo <= m.frac_lesao_total + 1e-9
    assert isinstance(m.n_lesoes, int) and isinstance(m.frac_marrom, float)
    assert 0.0 <= m.dist_borda_media_lesoes <= 1.0


def test_medir_nao_altera_os_argumentos(tmp_path):
    caminho = str(tmp_path / "f.png")
    cv2.imwrite(caminho, desenhar_folha(manchas=[(112, 112, 10)]))
    _, segmented, mask = segment_image(caminho)
    antes_img, antes_mask = segmented.copy(), mask.copy()
    medir_lesoes(segmented, mask)
    assert np.array_equal(segmented, antes_img) and np.array_equal(mask, antes_mask)


# ----------------------------------------------------------- máscaras vazias e suspeitas
def _imagem_e_mascara(mask):
    return np.zeros(mask.shape + (3,), np.uint8), mask


def test_mascara_vazia_e_recusada():
    img, mask = _imagem_e_mascara(np.zeros((224, 224), np.uint8))
    q = avaliar_mascara(mask)
    assert q.vazia and not q.confiavel and "vazia" in q.motivo
    with pytest.raises(MascaraInvalida):
        medir_lesoes(img, mask)


def test_avaliar_aceita_none_e_array_vazio():
    assert avaliar_mascara(None).vazia
    assert avaliar_mascara(np.zeros((0, 0), np.uint8)).vazia


def test_folha_pequena_demais_e_suspeita():
    mask = np.zeros((224, 224), np.uint8)
    mask[:50, :50] = 255                                # ~5% da imagem
    img, mask = _imagem_e_mascara(mask)
    q = avaliar_mascara(mask)
    assert q.suspeita and not q.vazia and not q.confiavel and "só" in q.motivo
    with pytest.raises(MascaraInvalida):
        medir_lesoes(img, mask)


def test_folha_ocupando_quase_tudo_e_suspeita():
    img, mask = _imagem_e_mascara(np.full((224, 224), 255, np.uint8))
    q = avaliar_mascara(mask)
    assert q.suspeita and "vazamento" in q.motivo
    with pytest.raises(MascaraInvalida):
        medir_lesoes(img, mask)


def test_fracao_dentro_da_faixa_e_confiavel():
    mask = np.zeros((224, 224), np.uint8)
    mask[:, :112] = 255                                 # 50%
    q = avaliar_mascara(mask)
    assert q.confiavel and q.motivo is None and q.leaf_fraction == pytest.approx(0.5)


def test_tamanhos_diferentes_levantam_erro_que_nao_e_mascara_invalida():
    img = np.zeros((224, 224, 3), np.uint8)
    mask = np.full((100, 100), 255, np.uint8)
    with pytest.raises(ValueError, match="tamanhos diferentes") as erro:
        medir_lesoes(img, mask)
    assert not isinstance(erro.value, MascaraInvalida)


# ------------------------------------------------------------------- configuração
def test_faixas_de_matiz_particionam_sem_sobreposicao():
    # marrom | amarelo | verde saudável devem ser consecutivos e coerentes com a segmentação
    assert config.LOWER_BROWN[0] == config.LOWER_LESION[0]
    assert config.UPPER_BROWN[0] + 1 == config.LOWER_YELLOW[0] == config.LOWER_GREEN[0]
    assert config.UPPER_YELLOW[0] + 1 == config.LOWER_HEALTHY_GREEN[0]
    assert config.UPPER_YELLOW[0] == config.UPPER_LESION[0]
    assert config.UPPER_HEALTHY_GREEN[0] == config.UPPER_GREEN[0]


# ------------------------------------------------------------------ sem TensorFlow
def test_modulo_nao_importa_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); "
              "import processamento_imagens.medidas_lesao; "
              "sys.exit(1 if 'tensorflow' in sys.modules else 0)")
    resultado = subprocess.run([sys.executable, "-I", "-c", codigo, SRC_DIR])
    assert resultado.returncode == 0

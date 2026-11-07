"""Testes da inferência da CNN (ia/inferencia.py).

Os testes sem TensorFlow usam preditores falsos e arquivos em tmp_path. Os que usam o
modelo real só LEEM models/plant_disease_model.keras (nada é gravado fora de tmp_path) e
são pulados se o TensorFlow ou o modelo não existirem. Eles verificam a mecânica
(formato, ordem, pré-processamento), não a qualidade das previsões.
"""

import importlib.util
import inspect
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from aplicacao.contrato import carregar_class_names, montar_fatos
from common.config import MODEL_PATH
from ia import inferencia
from ia.expert_system import carregar_kb, diagnosticar
from ia.inferencia import (
    AVISO_PESOS_LOCAIS, ModeloIndisponivel, PreditorCNN, carregar_preditor, preparar_entrada,
)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = list(carregar_class_names())


def folha_sintetica():
    img = np.zeros((224, 224, 3), np.uint8)
    cv2.ellipse(img, (112, 112), (85, 70), 0, 0, 360, (40, 160, 60), -1)
    cv2.circle(img, (100, 100), 12, (30, 70, 120), -1)
    return img


# ---------------------------------------------------------------- sem TensorFlow
def test_importar_o_modulo_nao_carrega_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import ia.inferencia; "
              "assert 'tensorflow' not in sys.modules, 'importou tensorflow'")
    resultado = subprocess.run([sys.executable, "-c", codigo, SRC_DIR],
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr


def test_modelo_ausente_levanta_modelo_indisponivel(tmp_path):
    nomes = tmp_path / "class_names.json"
    nomes.write_text(json.dumps(CLASSES), encoding="utf-8")
    with pytest.raises(ModeloIndisponivel, match="modelo da CNN não encontrado"):
        carregar_preditor(tmp_path / "nao_existe.keras", nomes)


def test_class_names_invalido_levanta_modelo_indisponivel(tmp_path):
    modelo = tmp_path / "modelo.keras"
    modelo.write_bytes(b"x")
    with pytest.raises(ModeloIndisponivel, match="lista de classes"):
        carregar_preditor(modelo, tmp_path / "nao_existe.json")
    ruins = {"json quebrado": "{nao e json", "12 nomes": json.dumps(CLASSES[:12]),
             "repetidos": json.dumps(["a"] * 13), "nao e lista": json.dumps({"a": 1}),
             "nao e texto": json.dumps(list(range(13)))}
    for nome, conteudo in ruins.items():
        arquivo = tmp_path / "ruim.json"
        arquivo.write_text(conteudo, encoding="utf-8")
        with pytest.raises(ModeloIndisponivel):
            carregar_preditor(modelo, arquivo)


def test_nenhuma_funcao_aceita_caminho_de_imagem_nome_ou_classe_real():
    permitidos = {
        carregar_preditor: {"model_path", "class_names_path", "verbose"},   # arquivos do modelo
        preparar_entrada: {"segmentada_bgr", "extensao"},
        PreditorCNN.__call__: {"self", "segmentada_bgr", "extensao"},
    }
    for funcao, nomes in permitidos.items():
        assert set(inspect.signature(funcao).parameters) == nomes, funcao


@pytest.mark.parametrize("ruim", [None, "imagem.jpg", np.zeros((224, 224), np.uint8),
                                  np.zeros((224, 224, 4), np.uint8),
                                  np.zeros((224, 224, 3), np.float32),
                                  np.zeros((0, 0, 3), np.uint8)])
def test_imagem_invalida_e_rejeitada(ruim):
    with pytest.raises(ValueError):
        preparar_entrada(ruim)


@pytest.mark.parametrize("extensao", [".bmp", "jpg", "", None])
def test_extensao_nao_suportada_e_rejeitada(extensao):
    with pytest.raises(ValueError, match="extensão"):
        preparar_entrada(folha_sintetica(), extensao)


def test_preditor_falso_alimenta_o_contrato_e_o_motor():
    def preditor_falso(segmentada_bgr, extensao=".jpg"):
        p = np.zeros(13, np.float32)
        p[CLASSES.index("Tomato_Late_blight")] = 1.0
        return p

    probs = preditor_falso(folha_sintetica())
    fatos = montar_fatos(probabilidades=probs, class_names=CLASSES)
    resultado = diagnosticar(fatos.fatos, carregar_kb())
    assert resultado.cultura == "tomate" and resultado.usou_cnn


def test_aviso_dos_pesos_cita_as_duas_limitacoes():
    assert "70/15/15" in AVISO_PESOS_LOCAIS and "split de teste" in AVISO_PESOS_LOCAIS
    assert "segmentação antiga" in AVISO_PESOS_LOCAIS
    assert inferencia.__doc__ and "LIMITAÇÃO DOS PESOS LOCAIS" in inferencia.__doc__


# ------------------------------------------------------------ com TensorFlow e o modelo
# Cada teste abaixo pede o TensorFlow; sem ele só estes são pulados (os de cima rodam).
requer_tensorflow = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None,
                                       reason="TensorFlow não está instalado")


@pytest.fixture(scope="module")
def preditor():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("models/plant_disease_model.keras não existe")
    return carregar_preditor()


@requer_tensorflow
def test_arquivo_de_modelo_corrompido_levanta_modelo_indisponivel(tmp_path):
    nomes = tmp_path / "class_names.json"
    nomes.write_text(json.dumps(CLASSES), encoding="utf-8")
    corrompido = tmp_path / "modelo.keras"
    corrompido.write_bytes(b"isto nao e um modelo keras")
    with pytest.raises(ModeloIndisponivel, match="não foi possível carregar"):
        carregar_preditor(corrompido, nomes)


@requer_tensorflow
def test_preditor_real_devolve_13_probabilidades_na_ordem_do_json(preditor):
    assert list(preditor.class_names) == CLASSES
    p = preditor(folha_sintetica())
    assert p.shape == (13,) and p.dtype == np.float32
    assert np.isfinite(p).all() and (p >= 0).all() and abs(float(p.sum()) - 1.0) < 1e-3


@requer_tensorflow
def test_preditor_real_e_deterministico(preditor):
    img = folha_sintetica()
    assert np.array_equal(preditor(img), preditor(img))


@requer_tensorflow
def test_preditor_real_nao_altera_a_imagem(preditor):
    img = folha_sintetica()
    copia = img.copy()
    preditor(img)
    assert np.array_equal(img, copia)


@requer_tensorflow
def test_pre_processamento_igual_ao_do_treino(tmp_path):
    # Treino: preprocessing.py grava com cv2.imwrite em data_processed/ e a CNN lê o arquivo
    # com tf.io.read_file + decode_and_resize (extract_features.py).
    import tensorflow as tf
    from ia.hibrido.extract_features import decode_and_resize
    img = folha_sintetica()
    arquivo = str(tmp_path / "folha.jpg")
    assert cv2.imwrite(arquivo, img)
    esperado = decode_and_resize(tf.io.read_file(arquivo))
    obtido = preparar_entrada(img)
    assert tuple(obtido.shape) == (1, 160, 160, 3)
    assert np.array_equal(obtido.numpy()[0], esperado.numpy())
    assert float(tf.reduce_max(obtido)) > 1.0                      # float 0-255, não 0-1


@requer_tensorflow
def test_saida_real_passa_no_contrato(preditor):
    fatos = montar_fatos(probabilidades=preditor(folha_sintetica()),
                         class_names=list(preditor.class_names))
    assert fatos.cultura_origem == "a_inferir_pela_cnn" and len(fatos.fatos["cnn_prior"]) == 13

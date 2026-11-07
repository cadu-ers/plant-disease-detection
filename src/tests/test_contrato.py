"""Testes do contrato (aplicacao/contrato.py).

Usam a KB e o class_names.json reais (só leitura), sem TensorFlow e sem imagens.
"""

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from aplicacao import contrato
from aplicacao.contrato import (
    ContratoInvalido, carregar_class_names, montar_cnn_prior, montar_fatos, montar_medidas,
)
from ia.expert_system import carregar_kb, diagnosticar
from processamento_imagens.medidas_lesao import MedidasLesao, QualidadeMascara

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


@pytest.fixture(scope="module")
def classes():
    return carregar_class_names()


def vetor(classes, **pesos):
    """Vetor de 13 probabilidades com a massa nas classes indicadas (por nome)."""
    p = np.zeros(len(classes), np.float32)
    for nome, valor in pesos.items():
        p[classes.index(nome)] = valor
    return p


def medidas(n_lesoes=3, dist=0.4, **extra):
    base = dict(frac_lesao_total=0.3, frac_marrom=0.2, frac_amarelo=0.05, n_lesoes=n_lesoes,
                n_lesoes_pequenas=2, dist_borda_media_lesoes=dist, leaf_fraction=0.5)
    base.update(extra)
    return MedidasLesao(**base)


CONFIAVEL = QualidadeMascara(0.5, False, False, None, 0.0)
SUSPEITA = QualidadeMascara(0.5, False, True, "possível fundo vazado", 0.4)
VAZIA = QualidadeMascara(0.0, True, False, "máscara vazia: nenhuma folha segmentada", 0.0)


# ------------------------------------------------------------------- class_names
def test_class_names_reais_tem_as_13_classes(classes):
    assert len(classes) == 13 and classes[0] == "Potato___Early_blight"


def test_class_names_invalido_e_rejeitado():
    with pytest.raises(ContratoInvalido):
        montar_cnn_prior(np.full(13, 1 / 13), ["a"] * 13)                  # repetidos
    with pytest.raises(ContratoInvalido):
        montar_cnn_prior(np.full(12, 1 / 12), ["Tomato_healthy"] * 12)     # tamanho
    with pytest.raises(ContratoInvalido):
        montar_cnn_prior(np.full(13, 1 / 13), [f"classe_{i}" for i in range(13)])


# ---------------------------------------------------------------------- cnn_prior
def test_prior_segue_a_ordem_de_class_names(classes):
    prior = montar_cnn_prior(vetor(classes, Tomato_Late_blight=0.7, Tomato_Early_blight=0.3),
                             classes)
    assert list(prior) == list(classes)
    assert prior["Tomato_Late_blight"] == pytest.approx(0.7)
    assert prior["Potato___healthy"] == 0.0


def test_permutar_classes_e_vetor_juntos_nao_muda_o_resultado(classes, kb):
    p = vetor(classes, Tomato_Late_blight=0.7, Tomato_Early_blight=0.3)
    ordem = np.random.default_rng(0).permutation(13)
    p2, classes2 = p[ordem], [classes[i] for i in ordem]
    assert montar_cnn_prior(p2, classes2) == montar_cnn_prior(p, classes)
    r1 = diagnosticar({"cnn_prior": montar_cnn_prior(p, classes)}, kb)
    r2 = diagnosticar({"cnn_prior": montar_cnn_prior(p2, classes2)}, kb)
    assert [(h.id, h.cf_final) for h in r1.hipoteses] == [(h.id, h.cf_final) for h in r2.hipoteses]


def test_permutar_so_o_vetor_troca_o_diagnostico(classes, kb):
    p = vetor(classes, Tomato_Late_blight=1.0)
    trocado = np.roll(p, 1)                                  # classes na ordem errada
    assert montar_cnn_prior(trocado, classes) != montar_cnn_prior(p, classes)
    melhor = diagnosticar({"cnn_prior": montar_cnn_prior(trocado, classes)}, kb).melhor.id
    assert melhor != "tomate_late_blight"


@pytest.mark.parametrize("ruim", [
    np.full(12, 1 / 12), np.full(14, 1 / 14),                 # tamanho
    [float("nan")] + [1 / 12] * 12, [float("inf")] + [0.0] * 12,
    [-0.1, 1.1] + [0.0] * 11,                                 # negativo
    [1.5] + [0.0] * 12,                                       # > 1
    np.full(13, 0.5 / 13), np.full(13, 1.5 / 13),             # soma longe de 1
    [True] + [0.0] * 12, ["0.5"] * 13, np.full((13, 1), 1 / 13), 0.5, None,
])
def test_probabilidades_invalidas_sao_rejeitadas(classes, ruim):
    with pytest.raises(ContratoInvalido):
        montar_cnn_prior(ruim, classes)


def test_soma_dentro_da_tolerancia_e_aceita(classes):
    p = np.full(13, 1 / 13, np.float32)
    assert montar_cnn_prior(p, classes)
    assert montar_cnn_prior(p * 1.005, classes)               # soma 1,005
    with pytest.raises(ContratoInvalido):
        montar_cnn_prior(p * 1.02, classes)                   # soma 1,02


# ------------------------------------------------------------------------ medidas
@pytest.mark.parametrize("qualidade", [SUSPEITA, VAZIA, None])
def test_mascara_ruim_omite_todas_as_medidas(qualidade, kb):
    f = montar_fatos(medidas=medidas(), qualidade=qualidade)
    assert "medidas" not in f.fatos and "sintomas" not in f.fatos
    assert not f.sem_manchas_marrons
    assert any("desconhecidas" in a for a in f.avisos)
    sintomas = diagnosticar(f.fatos, kb).sintomas
    assert not {"lesoes_grandes", "lesoes_marrom_escuro", "lesoes_na_borda"} & set(sintomas)


def test_motivo_da_mascara_suspeita_aparece_no_aviso():
    f = montar_fatos(medidas=medidas(), qualidade=SUSPEITA)
    assert any("fundo vazado" in a for a in f.avisos)


def test_mascara_confiavel_envia_so_as_medidas_da_kb(kb):
    f = montar_fatos(medidas=medidas(), qualidade=CONFIAVEL)
    assert set(f.fatos["medidas"]) == {"frac_marrom", "frac_lesao_total", "frac_amarelo",
                                       "n_lesoes_pequenas", "dist_borda_media_lesoes"}
    assert not f.sem_manchas_marrons
    assert "frac_lesoes_com_halo" not in f.fatos["medidas"]
    medidas_kb = {s.medida for s in kb.sintomas.values() if s.medida}
    assert set(f.fatos["medidas"]) <= medidas_kb
    assert "lesoes_na_borda" in diagnosticar(f.fatos, kb).sintomas       # medida presente


def test_n_lesoes_zero_omite_distancia_e_nao_cria_lesoes_na_borda(kb):
    f = montar_fatos(medidas=medidas(n_lesoes=0, dist=None, frac_marrom=0.0,
                                     n_lesoes_pequenas=0), qualidade=CONFIAVEL)
    assert f.sem_manchas_marrons is True
    assert any("nenhuma mancha marrom detectada" in a and "224x224" in a for a in f.avisos)
    assert "dist_borda_media_lesoes" not in f.fatos["medidas"]
    assert f.fatos["medidas"]["frac_marrom"] == 0.0                      # medida, não palpite
    assert "sintomas" not in f.fatos
    assert "lesoes_na_borda" not in diagnosticar(f.fatos, kb).sintomas   # segue desconhecido
    assert "sem_manchas_marrons" not in f.fatos                          # o motor não o vê


def test_medidas_incoerentes_sao_rejeitadas():
    with pytest.raises(ContratoInvalido):
        montar_medidas(medidas(n_lesoes=3, dist=None), CONFIAVEL)        # lesão sem distância
    with pytest.raises(ContratoInvalido):
        montar_medidas(medidas(n_lesoes=0, dist=0.3), CONFIAVEL)         # distância sem lesão
    with pytest.raises(ContratoInvalido):
        montar_medidas(None, CONFIAVEL)
    with pytest.raises(ContratoInvalido):
        montar_medidas(medidas(frac_marrom=float("nan")), CONFIAVEL)


# ----------------------------------------------------------------------- sintomas
def test_sintomas_nao_informados_ficam_desconhecidos(kb):
    f = montar_fatos(medidas=medidas(), qualidade=CONFIAVEL)
    sintomas = diagnosticar(f.fatos, kb).sintomas
    assert not {s.id for s in kb.sintomas.values() if s.origem == "usuario"} & set(sintomas)


def test_sintomas_do_usuario_sao_validados(kb):
    f = montar_fatos(sintomas_usuario={"aneis_concentricos": 1.0, "teia_fina": 0.0}, kb=kb)
    assert f.fatos["sintomas"] == {"aneis_concentricos": 1.0, "teia_fina": 0.0}
    for ruim in ({"lesoes_grandes": 1.0}, {"inventado": 1.0}, {"teia_fina": 1.5},
                 {"teia_fina": float("nan")}, {"teia_fina": True}, {"teia_fina": None}):
        with pytest.raises(ContratoInvalido):
            montar_fatos(sintomas_usuario=ruim, kb=kb)


# ------------------------------------------------------------------------ cultura
def test_cultura_origem(classes, kb):
    p = vetor(classes, Tomato_Late_blight=1.0)
    assert montar_fatos(probabilidades=p, class_names=classes).cultura_origem == "a_inferir_pela_cnn"
    assert montar_fatos().cultura_origem == "desconhecida"
    informada = montar_fatos(cultura="Tomato")
    assert informada.cultura_origem == "informada" and informada.fatos["cultura"] == "tomate"
    sem_cultura = montar_fatos(probabilidades=p, class_names=classes)
    assert "cultura" not in sem_cultura.fatos
    assert diagnosticar(sem_cultura.fatos, kb).cultura == "tomate"       # inferida pelo motor
    with pytest.raises(ContratoInvalido):
        montar_fatos(cultura="milho")


def test_sem_cnn_traz_aviso_e_nao_inventa_probabilidades():
    f = montar_fatos(medidas=medidas(), qualidade=CONFIAVEL)
    assert "cnn_prior" not in f.fatos and any("sem CNN" in a for a in f.avisos)


# --------------------------------------------------------------------- vazamento
# Lista fechada dos parâmetros de cada função do contrato. Nenhum deles pode carregar
# caminho de imagem, nome de arquivo ou classe real: se alguém acrescentar um parâmetro
# novo, este teste falha e obriga a revisar se ele abre uma porta para vazamento.
# (O "caminho" de carregar_class_names é o do models/class_names.json, não de uma imagem.)
PARAMETROS_PERMITIDOS = {
    montar_fatos: {"probabilidades", "class_names", "medidas", "qualidade", "cultura",
                   "sintomas_usuario", "kb"},
    montar_cnn_prior: {"probabilidades", "class_names"},
    montar_medidas: {"medidas", "qualidade"},
    carregar_class_names: {"caminho"},
}


@pytest.mark.parametrize("funcao", list(PARAMETROS_PERMITIDOS), ids=lambda f: f.__name__)
def test_nenhuma_funcao_aceita_caminho_nome_ou_classe_real(funcao):
    assert set(inspect.signature(funcao).parameters) == PARAMETROS_PERMITIDOS[funcao]


def test_fatos_so_tem_chaves_e_valores_permitidos(classes):
    f = montar_fatos(probabilidades=vetor(classes, Tomato_Late_blight=1.0), class_names=classes,
                     medidas=medidas(), qualidade=CONFIAVEL, cultura="tomate",
                     sintomas_usuario={"teia_fina": 0.0})
    assert set(f.fatos) <= {"cultura", "medidas", "sintomas", "cnn_prior"}
    assert set(f.fatos["cnn_prior"]) == set(classes)
    for grupo in ("medidas", "sintomas", "cnn_prior"):
        assert all(isinstance(v, float) for v in f.fatos[grupo].values())
    json.dumps(f.fatos)                                                  # serializável


def test_isca_de_classe_real_nao_aparece_nos_fatos(classes):
    # A classe real (isca) existe só no nome da pasta/arquivo, que o contrato nunca recebe.
    isca = "Potato___Late_blight/ISCA_CLASSE_REAL.JPG"
    f = montar_fatos(probabilidades=vetor(classes, Tomato_healthy=1.0), class_names=classes,
                     medidas=medidas(), qualidade=CONFIAVEL)
    assert "ISCA" not in json.dumps(f.fatos) and "batata" not in json.dumps(f.fatos)
    assert isca not in repr(f)


# ------------------------------------------------------------------- integração
def test_fatos_completos_passam_no_motor(classes, kb):
    f = montar_fatos(probabilidades=vetor(classes, Tomato_Late_blight=0.8, Tomato_Early_blight=0.2),
                     class_names=classes, medidas=medidas(), qualidade=CONFIAVEL,
                     sintomas_usuario={"aspecto_encharcado": 1.0, "umidade_alta": 1.0})
    r = diagnosticar(f.fatos, kb)
    assert r.cultura == "tomate" and r.usou_cnn and r.melhor is not None


def test_modulo_nao_importa_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import aplicacao.contrato; "
              "assert 'tensorflow' not in sys.modules, 'importou tensorflow'")
    resultado = subprocess.run([sys.executable, "-c", codigo, SRC_DIR],
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr

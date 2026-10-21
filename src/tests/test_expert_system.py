"""Testes do sistema especialista. Usam fatos montados à mão: não precisam de
imagem, de dataset nem do modelo treinado."""

import copy

import pytest
import yaml

from ia.expert_system import (carregar_kb, chave_canonica, diagnosticar,
                              diagnostico_interativo, explicar, proxima_pergunta)
from ia.expert_system.classes import CHAVES_CANONICAS, MAPEAMENTO_DATASET
from ia.expert_system.engine import combinar, combinar_cf
from ia.expert_system.fuzzy import trapezio
from ia.expert_system.kb import KB_PADRAO, kb_de_dict


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


def topo(res):
    return res.melhor.id


# ------------------------------------------------------------------ classes
def test_mapeamento_cobre_as_13_classes_do_dataset(kb):
    assert len(MAPEAMENTO_DATASET) == 13 and len(CHAVES_CANONICAS) == 13
    for nome, chave in MAPEAMENTO_DATASET.items():
        assert chave_canonica(nome) == chave
        assert chave in kb.doencas


def test_chave_canonica_tolera_caixa_e_separadores():
    assert chave_canonica("tomato__target_spot") == "tomate_target_spot"
    assert chave_canonica("tomate_saudavel") == "tomate_saudavel"
    with pytest.raises(KeyError):
        chave_canonica("Pepper__bell___healthy")


# ---------------------------------------------------------------------- kb
def test_toda_doenca_tem_ao_menos_uma_regra_positiva(kb):
    for d in kb.doencas:
        assert any(r.cf > 0 for r in kb.regras_de(d)), d


def test_kb_rejeita_sintoma_inexistente():
    with open(KB_PADRAO, encoding="utf-8") as f:
        dados = yaml.safe_load(f)
    ruim = copy.deepcopy(dados)
    ruim["regras"][0]["se"].append("sintoma_que_nao_existe")
    with pytest.raises(ValueError, match="sintoma inexistente"):
        kb_de_dict(ruim)


def test_kb_rejeita_cf_fora_do_intervalo_e_id_duplicado():
    with open(KB_PADRAO, encoding="utf-8") as f:
        dados = yaml.safe_load(f)
    ruim = copy.deepcopy(dados)
    ruim["regras"][0]["entao"]["cf"] = 1.5
    with pytest.raises(ValueError, match="cf"):
        kb_de_dict(ruim)
    dup = copy.deepcopy(dados)
    dup["regras"][1]["id"] = dup["regras"][0]["id"]
    with pytest.raises(ValueError, match="duplicado"):
        kb_de_dict(dup)


# ------------------------------------------------------------- álgebra de CF
def test_combinar_cf_propriedades():
    assert combinar_cf(0.5, 0.5) == pytest.approx(0.75)
    assert combinar_cf(0.3, 0.6) == pytest.approx(combinar_cf(0.6, 0.3))
    assert combinar_cf(-0.5, -0.5) == pytest.approx(-0.75)
    assert combinar_cf(0.8, -0.3) == pytest.approx(0.5 / 0.7)
    assert -1 <= combinar_cf(0.99, -0.99) <= 1


def test_combinar_nao_depende_da_ordem():
    cfs = [0.4, -0.3, 0.6, -0.2, 0.1]
    assert combinar(cfs) == pytest.approx(combinar(list(reversed(cfs))))
    assert combinar([]) == 0.0


def test_trapezio():
    assert trapezio(0.0, 0.1, 0.2, 1, 1) == 0.0
    assert trapezio(0.15, 0.1, 0.2, 1, 1) == pytest.approx(0.5)
    assert trapezio(0.5, 0.1, 0.2, 1, 1) == 1.0
    assert trapezio(0.0, 0.0, 0.0, 0.25, 0.5) == 1.0   # sintoma "na borda"
    assert trapezio(0.4, 0.0, 0.0, 0.25, 0.5) == pytest.approx(0.4)


# --------------------------------------------------------------- diagnósticos
def test_requeima_do_tomate(kb):
    res = diagnosticar({"cultura": "tomate", "sintomas": {
        "aspecto_encharcado": 1.0, "lesoes_grandes": 0.9, "micelio_branco_inferior": 1.0}}, kb)
    assert topo(res) == "tomate_late_blight"
    assert res.conclusivo


def test_filtro_de_cultura(kb):
    sintomas = {"aspecto_encharcado": 1.0, "lesoes_grandes": 0.9, "micelio_branco_inferior": 1.0}
    res = diagnosticar({"cultura": "batata", "sintomas": sintomas}, kb)
    assert topo(res) == "batata_late_blight"
    assert all(h.id.startswith("batata_") for h in res.hipoteses)


def test_cultura_inferida_pelo_prior_da_cnn(kb):
    res = diagnosticar({"cnn_prior": {"Potato___Late_blight": 0.9}}, kb)
    assert res.cultura == "batata"


def test_pinta_preta_com_aneis(kb):
    res = diagnosticar({"cultura": "tomate", "sintomas": {
        "lesoes_marrom_escuro": 0.9, "aneis_concentricos": 1.0}}, kb)
    assert topo(res) == "tomate_early_blight"
    assert res.hipotese("tomate_target_spot").cf_final >= kb.config["limiar_exibicao"]  # ambiguidade


def test_mofo_da_folha(kb):
    res = diagnosticar({"cultura": "tomate", "sintomas": {
        "mofo_face_inferior": 1.0, "manchas_amarelo_palidas_superior": 0.8}}, kb)
    assert topo(res) == "tomate_leaf_mold" and res.conclusivo


def test_septoriose_e_acaro_e_virus(kb):
    sept = diagnosticar({"cultura": "tomate", "sintomas": {
        "lesoes_pequenas_numerosas": 1.0, "centro_cinza_pontos_pretos": 1.0}}, kb)
    assert topo(sept) == "tomate_septoria"
    acaro = diagnosticar({"cultura": "tomate", "sintomas": {"teia_fina": 1.0}}, kb)
    assert topo(acaro) == "tomate_spider_mites"
    curl = diagnosticar({"cultura": "tomate", "sintomas": {
        "folha_enrolada_cima": 1.0, "amarelecimento_difuso": 0.8}}, kb)
    assert topo(curl) == "tomate_yellow_leaf_curl"
    mosaico = diagnosticar({"cultura": "tomate", "sintomas": {
        "mosaico_claro_escuro": 1.0, "folha_deformada": 0.9}}, kb)
    assert topo(mosaico) == "tomate_mosaic"


def test_folha_saudavel(kb):
    ausentes = {s: 0.0 for s in ("lesoes_marrom_escuro", "amarelecimento_difuso",
                                 "lesoes_pequenas_numerosas", "lesoes_grandes", "halo_amarelo")}
    res = diagnosticar({"cultura": "tomate", "sintomas": ausentes}, kb)
    assert topo(res) == "tomate_saudavel"
    doente = diagnosticar({"cultura": "tomate", "sintomas": {"lesoes_marrom_escuro": 0.9}}, kb)
    assert doente.hipotese("tomate_saudavel").cf_final < 0


def test_desconhecido_nao_e_ausente(kb):
    res = diagnosticar({"cultura": "tomate", "sintomas": {"lesoes_marrom_escuro": 0.9}}, kb)
    assert "aneis_concentricos" in res.hipotese("tomate_early_blight").pendentes
    assert res.hipotese("tomate_saudavel").cf_regras < 0  # saudável não dispara sem os demais sintomas
    with pytest.raises(ValueError, match="Sintoma desconhecido"):
        diagnosticar({"sintomas": {"sintoma_com_typo": 1.0}}, kb)


def test_medidas_das_mascaras_sao_fuzzificadas(kb):
    res = diagnosticar({"cultura": "tomate", "medidas": {
        "frac_marrom": 0.10, "frac_lesao_total": 0.10, "frac_lesoes_com_halo": 0.7,
        "frac_amarelo": 0.0, "n_lesoes_pequenas": 1}}, kb)
    assert res.sintomas["lesoes_marrom_escuro"] == 1.0
    assert res.sintomas["amarelecimento_difuso"] == 0.0
    assert res.hipotese("tomate_early_blight").cf_regras > 0   # R03


def test_sintomas_explicitos_sobrescrevem_medidas(kb):
    res = diagnosticar({"medidas": {"frac_marrom": 0.5}, "sintomas": {"lesoes_marrom_escuro": 0.0}}, kb)
    assert res.sintomas["lesoes_marrom_escuro"] == 0.0


# ---------------------------------------------------------------------- CNN
def test_prior_da_cnn_e_ablacao(kb):
    fatos = {"cultura": "tomate", "cnn_prior": {"Tomato_Early_blight": 0.8}}
    com = diagnosticar(fatos, kb)
    assert topo(com) == "tomate_early_blight"
    assert com.melhor.cf_final == pytest.approx(kb.config["peso_cnn"] * 0.8)
    sem = diagnosticar(fatos, kb, usar_cnn=False)
    assert sem.melhor.cf_final == 0.0 and not sem.usou_cnn


def test_conflito_entre_regras_e_cnn(kb):
    res = diagnosticar({"cultura": "tomate",
                        "sintomas": {"mofo_face_inferior": 1.0},
                        "cnn_prior": {"Tomato_Early_blight": 0.6}}, kb)
    assert res.conflito == ("tomate_leaf_mold", "tomate_early_blight")
    assert "divergem" in explicar(res, kb)


# --------------------------------------------------------------- explicação
def test_explicacao_traz_regras_por_que_nao_e_recomendacoes(kb):
    res = diagnosticar({"cultura": "tomate", "sintomas": {
        "lesoes_marrom_escuro": 0.9, "aneis_concentricos": 1.0}}, kb)
    texto = explicar(res, kb)
    assert "R01" in texto and "anéis concêntricos" in texto
    assert "Por que não" in texto and "Mancha-alvo" in texto
    assert "Recomendações gerais" in texto and "agrônomo" in texto


def test_explicacao_de_resultado_vazio_e_inconclusivo(kb):
    texto = explicar(diagnosticar({"cultura": "tomate"}, kb), kb)
    assert "INCONCLUSIVO" in texto


# --------------------------------------------------------------- interativo
def test_proxima_pergunta_separa_as_lideres(kb):
    res = diagnosticar({"cultura": "tomate", "sintomas": {
        "lesoes_pequenas_numerosas": 1.0, "lesoes_marrom_escuro": 0.9}}, kb)
    assert proxima_pergunta(res, kb).id == "centro_cinza_pontos_pretos"


def test_modo_interativo_chega_ao_diagnostico_sem_repetir_perguntas(kb):
    verdade = {"centro_cinza_pontos_pretos": 1.0, "halo_amarelo": 0.0, "aspecto_encharcado": 0.0}
    perguntados = []

    def oraculo(sintoma):
        perguntados.append(sintoma.id)
        return verdade.get(sintoma.id)

    res, hist = diagnostico_interativo(
        {"cultura": "tomate", "sintomas": {"lesoes_pequenas_numerosas": 1.0,
                                           "lesoes_marrom_escuro": 0.9}},
        kb, oraculo, max_perguntas=4)
    assert topo(res) == "tomate_septoria" and res.conclusivo
    assert len(perguntados) == len(set(perguntados)) <= 4
    assert [s for s, _ in hist] == perguntados


def test_modo_interativo_respeita_nao_sei_e_limite(kb):
    res, hist = diagnostico_interativo({"cultura": "tomate"}, kb, lambda s: None, max_perguntas=3)
    assert len(hist) <= 3 and not res.conclusivo

"""Testes do guardrail do laudo (ia/generativa/guardrail.py): funções puras, com um resumo
montado à mão no formato de laudo.resumir_diagnostico. Sem rede, sem imagem, sem modelo."""

import copy
import json

import pytest

from ia.expert_system import carregar_kb
from ia.generativa.guardrail import (
    SCHEMA_LAUDO, VIOLACAO_JSON_INVALIDO, acoes_irreversiveis_sem_condicao, montar_texto,
    regras_chamadas_de_fracas, registrar, sem_acentos, validar_laudo,
)
from ia.generativa.prompts import TITULOS_LAUDO

MOSAICO = "Vírus do mosaico do tomateiro (ToMV)"


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


@pytest.fixture
def resumo():
    """Caso do primeiro laudo real: mosaico, inconclusivo, com perguntas pendentes."""
    return {
        "situacao": "inconclusivo",
        "hipotese_principal": {"nome": MOSAICO, "agente": "Tobamovirus", "cf": 0.49,
                               "urgencia": "alta"},
        "outras_hipoteses": [],
        "pode_ser_confundida_com": ["Vírus do enrolamento amarelo da folha do tomateiro (TYLCV)"],
        "rede_neural_top3": [{"classe": MOSAICO, "probabilidade_relativa": 0.99}],
        "conflito_regras_x_rede_neural": None,
        "perguntas_pendentes": ["A folha tem um mosqueado (mosaico) de verde claro e escuro?",
                                "As folhas estão deformadas, estreitas ou menores que o normal?"],
        "recomendacoes": ["Não há cura: remover e destruir plantas infectadas.",
                          "Higienizar mãos e ferramentas e usar sementes certificadas.",
                          "Procurar orientação técnica sobre variedades resistentes."],
    }


@pytest.fixture
def valido():
    return {
        "o_que_encontrou": "O sistema não tem certeza. A hipótese mais forte é o mosaico do "
                           "tomateiro, mas faltam sinais para confirmar.",
        "por_que": "A análise da foto aponta mosaico, mas os sinais medidos na folha ainda "
                   "não confirmam.",
        "perguntas": ["A folha tem um mosqueado de verde claro e escuro?"],
        "o_que_fazer": "Se o diagnóstico for confirmado por um técnico, remova e destrua as "
                       "plantas infectadas. Lave as mãos e as ferramentas. Procure um agrônomo.",
        "hipotese_citada": MOSAICO,
        "situacao_declarada": "inconclusivo",
        "recomendacoes_usadas": [0, 1],
        "perguntas_usadas": [0],
    }


def com(dados, **troca):
    novo = copy.deepcopy(dados)
    novo.update(troca)
    return novo


# --------------------------------------------------------------------------- aprovado
def test_resposta_valida_nao_tem_violacoes(valido, resumo, kb):
    assert validar_laudo(valido, resumo, kb) == []


def test_schema_pede_todos_os_campos():
    assert set(SCHEMA_LAUDO["required"]) == set(SCHEMA_LAUDO["properties"]) == {
        "o_que_encontrou", "por_que", "o_que_fazer", "perguntas", "hipotese_citada",
        "situacao_declarada", "recomendacoes_usadas", "perguntas_usadas"}
    json.dumps(SCHEMA_LAUDO)                                   # vai para a API como JSON


# ----------------------------------------------------------------- formato do JSON
@pytest.mark.parametrize("dados", [None, [], "texto", 3])
def test_json_que_nao_e_objeto_e_violacao(dados, resumo, kb):
    assert validar_laudo(dados, resumo, kb) == [VIOLACAO_JSON_INVALIDO]


def test_campo_ausente_ou_com_tipo_errado_e_violacao(valido, resumo, kb):
    sem = copy.deepcopy(valido)
    del sem["por_que"]
    assert validar_laudo(sem, resumo, kb) == ['campo "por_que" ausente ou vazio']
    assert validar_laudo(com(valido, perguntas="uma só"), resumo, kb) == [
        'campo "perguntas" deve ser uma lista de textos']
    assert validar_laudo(com(valido, recomendacoes_usadas=[True]), resumo, kb) == [
        'campo "recomendacoes_usadas" deve ser uma lista de números inteiros']
    sem_hip = copy.deepcopy(valido)
    del sem_hip["hipotese_citada"]
    assert validar_laudo(sem_hip, resumo, kb) == ['campo "hipotese_citada" ausente']


# -------------------------------------------------------------- coerência com o motor
def test_hipotese_citada_tem_de_ser_a_principal(valido, resumo, kb):
    v = validar_laudo(com(valido, hipotese_citada="Mosaico"), resumo, kb)
    assert v == ["hipotese_citada não é a hipótese principal do sistema especialista"]
    assert "Mosaico" not in v[0].replace("hipótese", "")       # não repete o texto do LLM


def test_sem_hipotese_principal_a_citada_tem_de_ser_null(valido, resumo, kb):
    resumo = com(resumo, hipotese_principal=None, situacao="sem_hipotese", recomendacoes=[])
    dados = com(valido, situacao_declarada="sem_hipotese", recomendacoes_usadas=[],
                o_que_fazer="Procure um agrônomo.")
    assert validar_laudo(dados, resumo, kb) == [
        "hipotese_citada deveria ser null (não há hipótese principal)"]
    assert validar_laudo(com(dados, hipotese_citada=None), resumo, kb) == []


def test_situacao_declarada_tem_de_ser_a_do_motor(valido, resumo, kb):
    assert validar_laudo(com(valido, situacao_declarada="conclusivo"), resumo, kb) == [
        'situacao_declarada deveria ser "inconclusivo"']


def test_indices_inexistentes_sao_violacao(valido, resumo, kb):
    v = validar_laudo(com(valido, recomendacoes_usadas=[0, 3, -1], perguntas_usadas=[2]),
                      resumo, kb)
    assert v == ["recomendacoes_usadas tem índice(s) inexistente(s): -1, 3",
                 "perguntas_usadas tem índice(s) inexistente(s): 2"]


def test_doenca_fora_dos_fatos_e_violacao(valido, resumo, kb):
    dados = com(valido, por_que="Pode ser também a septoriose do tomateiro.")
    assert validar_laudo(dados, resumo, kb) == [
        "cita doença fora do diagnóstico: Septoriose do tomateiro (Septoria leaf spot)"]
    # a confundível que está no resumo pode ser citada
    assert validar_laudo(com(valido, por_que="Pode ser confundido com o TYLCV."), resumo, kb) == []


# ------------------------------------------------------------------ "%", dose e produto
@pytest.mark.parametrize("trecho", ["cerca de 30% da folha", "100 %"])
def test_porcentagem_e_violacao(trecho, valido, resumo, kb):
    v = validar_laudo(com(valido, o_que_encontrou=f"A folha tem {trecho}."), resumo, kb)
    assert v == ['usa "%" (força da evidência não é porcentagem)']


@pytest.mark.parametrize("trecho, descricao", [
    ("aplique 20 ml por bomba", "quantidade com unidade"),
    ("use 1,5 kg", "quantidade com unidade"),
    ("10 L de calda", "quantidade com unidade"),
    ("3g por planta", "quantidade com unidade"),
    ("2 litros/ha", "quantidade por hectare"),
    ("500 por hectare", "quantidade por hectare"),
    ("na dose certa", '"dose"'),
    ("um fungicida", "nome de classe de produto"),
    ("inseticidas", "nome de classe de produto"),
    ("Bactericida", "nome de classe de produto"),
])
def test_dose_ou_produto_e_violacao(trecho, descricao, valido, resumo, kb):
    dados = com(valido, o_que_fazer=f"Se o diagnóstico for confirmado, {trecho}.")
    assert f"indica dose ou produto ({descricao})" in validar_laudo(dados, resumo, kb)


def test_numeros_sem_unidade_nao_sao_dose(valido, resumo, kb):
    dados = com(valido, o_que_encontrou="A folha tem 2 regiões marrons e 1 lesão grande.")
    assert validar_laudo(dados, resumo, kb) == []


# ------------------------------------------------------------------- ação irreversível
VIOLACAO_IRREVERSIVEL = ('"o_que_fazer" manda destruir, arrancar, eliminar, remover, retirar '
                         "ou queimar plantas sem a condição de confirmação (o caso não é "
                         "conclusivo)")


@pytest.mark.parametrize("frase", [
    "Remova e destrua as plantas infectadas.",
    "Arranque a planta doente.",
    "É preciso eliminar as plantas.",
    "As plantas devem ser destruídas.",
    "Se o diagnóstico for confirmado, lave as ferramentas. Destrua as plantas.",  # outra frase
    "Remova as plantas.",
    "Retire a planta da lavoura.",
    "Queime as plantas doentes.",
    "As plantas infectadas devem ser removidas.",
    "Remover plantas muito afetadas.",                       # texto da própria KB
])
def test_acao_irreversivel_sem_condicao_em_inconclusivo_e_violacao(frase, valido, resumo, kb):
    assert validar_laudo(com(valido, o_que_fazer=frase), resumo, kb) == [VIOLACAO_IRREVERSIVEL]


@pytest.mark.parametrize("frase", [
    "Se o diagnóstico for confirmado por um técnico, destrua as plantas infectadas.",
    "Se for confirmado, arranque a planta.",
    "Caso o diagnóstico seja confirmado, elimine as plantas.",
    "Após a confirmação, destrua as plantas.",
    "Depois de confirmado por um agrônomo, elimine a planta.",
    "Se o diagnóstico for confirmado, remova as plantas.",
    "Caso seja confirmado, retire e queime as plantas.",
    "Elimine os restos culturais e remova as folhas afetadas.",   # não é ação sobre a planta
    "Remova as folhas afetadas da planta.",                       # pedaço da planta
    "Retire as partes doentes das plantas e queime os restos da planta.",
])
def test_acao_irreversivel_com_condicao_ou_sem_planta_nao_e_violacao(frase, valido, resumo, kb):
    assert validar_laudo(com(valido, o_que_fazer=frase), resumo, kb) == []


def test_em_caso_conclusivo_a_acao_irreversivel_e_permitida(valido, resumo, kb):
    resumo = com(resumo, situacao="conclusivo")
    dados = com(valido, situacao_declarada="conclusivo", perguntas=[], perguntas_usadas=[],
                o_que_fazer="Remova e destrua as plantas infectadas. Procure um agrônomo.")
    assert validar_laudo(dados, resumo, kb) == []
    assert acoes_irreversiveis_sem_condicao(dados["o_que_fazer"]) == 1


# ------------------------------------------------------------ nenhuma regra disparou
def sem_regras(resumo, regras=0):
    return com(resumo, origem_da_evidencia={
        "forca_pela_rede_neural": 0.49, "maximo_pela_rede_neural": 0.5,
        "forca_pelas_regras": 0.0 if regras == 0 else 0.4, "regras_que_dispararam": regras})


POR_QUE_OK = ("A análise da foto aponta mosaico, mas nenhuma regra confirmou: os sinais "
              "medidos na folha não mostram os sintomas típicos.")
VIOLACAO_FRACA = ('chama a força das regras de "fraca", "baixa" ou "pequena", mas nenhuma '
                  "regra disparou")
VIOLACAO_NAO_DIZ = '"por_que" não diz que nenhuma regra confirmou a hipótese'


@pytest.mark.parametrize("por_que", [
    POR_QUE_OK,
    "A foto aponta mosaico. Nenhuma das regras de fitopatologia confirmou a hipótese.",
    "A foto aponta mosaico, mas não houve regra que confirmasse.",
    "A foto aponta mosaico, mas as regras não confirmaram.",
])
def test_sem_regras_dizer_que_nenhuma_confirmou_e_aprovado(por_que, valido, resumo, kb):
    assert validar_laudo(com(valido, por_que=por_que), sem_regras(resumo), kb) == []


@pytest.mark.parametrize("trecho", [
    "A evidência pelas regras é fraca.",
    "As regras deram uma força baixa.",
    "A força das regras ficou pequena.",
    "Pelas regras, o apoio é fraco.",
])
def test_sem_regras_chamar_a_forca_das_regras_de_fraca_e_violacao(trecho, valido, resumo, kb):
    dados = com(valido, por_que=POR_QUE_OK + " " + trecho)
    assert validar_laudo(dados, sem_regras(resumo), kb) == [VIOLACAO_FRACA]


def test_sem_regras_e_sem_dizer_que_nenhuma_confirmou_e_violacao(valido, resumo, kb):
    dados = com(valido, por_que="A análise da foto aponta mosaico, mas a evidência das regras "
                                "é baixa.")
    assert validar_laudo(dados, sem_regras(resumo), kb) == [VIOLACAO_FRACA, VIOLACAO_NAO_DIZ]
    dados = com(valido, por_que="A análise da foto aponta mosaico.")
    assert validar_laudo(dados, sem_regras(resumo), kb) == [VIOLACAO_NAO_DIZ]


def test_com_regras_que_dispararam_fraca_e_permitido(valido, resumo, kb):
    dados = com(valido, por_que="A foto aponta mosaico e as regras dão uma evidência fraca.")
    assert validar_laudo(dados, sem_regras(resumo, regras=1), kb) == []


def test_manchas_pequenas_sem_falar_de_regras_nao_e_violacao(valido, resumo, kb):
    dados = com(valido, por_que=POR_QUE_OK + " A folha tem manchas pequenas e fracas.")
    assert validar_laudo(dados, sem_regras(resumo), kb) == []


# ------------------------------------------------------------------------------ acentos
def test_sem_acentos_so_para_comparar():
    assert sem_acentos("Não há força no Vírus; agrônomo, ação") == \
        "Nao ha forca no Virus; agronomo, acao"


def test_texto_com_acentos_nao_tem_violacao(valido, resumo, kb):
    dados = com(valido, por_que="A análise da foto aponta o vírus, mas a força das regras "
                                "não é suficiente. Você deve falar com um agrônomo.")
    assert validar_laudo(dados, resumo, kb) == []


def test_texto_sem_acentos_e_violacao(valido, resumo, kb):
    # o primeiro laudo v3 real veio assim do Gemini
    dados = com(valido, o_que_encontrou="O sistema nao tem certeza sobre o diagnostico "
                                        "porque a forca das evidencias ficou abaixo do minimo.")
    assert validar_laudo(dados, resumo, kb) == [
        'texto sem acentos ("nao", "diagnostico", "forca"); escreva em português com acentos']


def test_palavra_com_virus_dentro_nao_e_texto_sem_acento(valido, resumo, kb):
    dados = com(valido, o_que_encontrou="O agente é um Tobamovirus (vírus).")
    assert validar_laudo(dados, resumo, kb) == []


def test_checagens_pegam_tambem_texto_sem_acento(valido, resumo, kb):
    # doença fora dos fatos escrita sem acento: o nome devolvido é o da base, com acento
    fora = validar_laudo(com(valido, por_que="Pode ser Acaro-rajado."), resumo, kb)
    assert fora == ["cita doença fora do diagnóstico: Ácaro-rajado (Two-spotted spider mite)"]
    # condição sem acento continua valendo como condição
    assert acoes_irreversiveis_sem_condicao("Apos a confirmacao, destrua as plantas.") == 0
    assert acoes_irreversiveis_sem_condicao("Depois da confirmacao, queime a planta.") == 0
    # "nao" sem acento também conta como "nenhuma regra confirmou"
    assert regras_chamadas_de_fracas(["As regras nao confirmaram, a forca e fraca."]) == []


def test_validar_nao_altera_os_textos_do_llm(valido, resumo, kb):
    original = copy.deepcopy(valido)
    validar_laudo(valido, resumo, kb)
    assert valido == original                      # o texto mostrado mantém os acentos


# ----------------------------------------------------------------------------- perguntas
def test_inconclusivo_com_pendencias_exige_perguntas(valido, resumo, kb):
    v = validar_laudo(com(valido, perguntas=[], perguntas_usadas=[]), resumo, kb)
    assert v == ['"perguntas" está vazia, mas há perguntas pendentes']
    assert validar_laudo(com(valido, perguntas=[], perguntas_usadas=[]),
                         com(resumo, perguntas_pendentes=[]), kb) == []


# -------------------------------------------------------------------------- texto final
def test_montar_texto_usa_os_titulos_e_uma_pergunta_por_linha(valido):
    texto = montar_texto(com(valido, perguntas=["- Primeira?", "Segunda?", "  "]))
    blocos = texto.split("\n\n")
    assert [b.splitlines()[0] for b in blocos] == list(TITULOS_LAUDO)
    assert blocos[2] == "Perguntas para você:\n- Primeira?\n- Segunda?"


def test_montar_texto_sem_perguntas_omite_o_bloco(valido):
    texto = montar_texto(com(valido, perguntas=[]))
    assert "Perguntas para você:" not in texto and texto.endswith("Procure um agrônomo.")


# ----------------------------------------------------------------------------- registro
def test_registrar_acrescenta_uma_linha_json_por_laudo(tmp_path):
    log = tmp_path / "sub" / "guardrail_log.jsonl"
    for status in ("aprovado", "reprovado"):
        assert registrar(log, modelo="m-teste", versao_prompt=3, situacao="inconclusivo",
                         status=status, tentativas=1, violacoes_por_tentativa=[[]],
                         do_cache_por_tentativa=[status == "reprovado"])
    linhas = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert [l["status_guardrail"] for l in linhas] == ["aprovado", "reprovado"]
    assert [l["do_cache"] for l in linhas] == [[False], [True]]
    assert set(linhas[0]) == {"data_hora", "modelo", "versao_prompt", "situacao",
                              "status_guardrail", "tentativas", "violacoes", "do_cache"}


def test_registrar_nunca_levanta(tmp_path):
    pasta = tmp_path / "ocupado"
    pasta.mkdir()                                       # um diretório no lugar do arquivo
    assert registrar(pasta, modelo="m", versao_prompt=3, situacao="x", status="aprovado",
                     tentativas=1, violacoes_por_tentativa=[[]]) is False
    assert registrar(None, modelo="m", versao_prompt=3, situacao="x", status="aprovado",
                     tentativas=1, violacoes_por_tentativa=[[]]) is False

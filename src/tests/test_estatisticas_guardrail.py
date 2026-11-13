"""Testes das estatísticas do guardrail (ia/generativa/estatisticas_guardrail.py), com um
log falso em tmp_path. Não usam a API nem o log real."""

import json

import pytest

from ia.generativa.estatisticas_guardrail import (
    formatar, ler_log, origem, principal, resumir, tipo_da_violacao,
)

ACENTO = {"violacao": 'texto sem acentos ("nao", "diagnostico"); escreva em português com '
                      "acentos", "nivel": "qualidade"}
FORA = {"violacao": "cita doença fora do diagnóstico: Septoriose do tomateiro (Septoria leaf "
                    "spot)", "nivel": "seguranca"}
DOSE = {"violacao": "indica dose ou produto (quantidade com unidade)", "nivel": "seguranca"}


def registro(status, violacoes, do_cache, versao=4):
    return {"data_hora": "2026-11-12T10:00:00", "modelo": "m-teste", "versao_prompt": versao,
            "situacao": "inconclusivo", "status_guardrail": status,
            "tentativas": len(violacoes), "violacoes": violacoes, "do_cache": do_cache}


@pytest.fixture
def log(tmp_path):
    linhas = [
        registro("aprovado", [[]], [False]),
        registro("aprovado", [[]], [False]),
        registro("corrigido", [[ACENTO], []], [False, False]),
        registro("corrigido", [[FORA, ACENTO], []], [False, False]),
        registro("reprovado", [[DOSE], [DOSE, FORA]], [False, False]),
        registro("aprovado_com_aviso", [[ACENTO], [ACENTO]], [False, False]),
        registro("corrigido", [[ACENTO], []], [True, True]),          # repetição do cache
        registro("corrigido", [[ACENTO], []], [True, True]),
        registro("corrigido", [[FORA], []], [True, False]),           # misto
    ]
    caminho = tmp_path / "guardrail_log.jsonl"
    texto = "\n".join(json.dumps(l, ensure_ascii=False) for l in linhas)
    caminho.write_text(texto + "\n{quebrada\n\n[1, 2]\n", encoding="utf-8")
    return caminho


def test_ler_log_ignora_linhas_invalidas(log):
    registros, ignoradas = ler_log(log)
    assert len(registros) == 9 and ignoradas == 2      # "{quebrada" e "[1, 2]"; vazia não conta


@pytest.mark.parametrize("do_cache, esperado", [
    ([False], "api"), ([False, False], "api"), ([True], "cache"), ([True, True], "cache"),
    ([True, False], "misto"), ([False, True], "misto"), ([None], "api"), (None, "api"),
])
def test_origem(do_cache, esperado):
    assert origem({"do_cache": do_cache}) == esperado


@pytest.mark.parametrize("texto, tipo", [
    (ACENTO["violacao"], "texto sem acentos"),
    (FORA["violacao"], "cita doença fora do diagnóstico"),
    (DOSE["violacao"], "indica dose ou produto"),
    ('usa "%" (força da evidência não é porcentagem)', 'usa "%"'),
    ("recomendacoes_usadas tem índice(s) inexistente(s): 3", "recomendacoes_usadas tem índice(s) inexistente(s)"),
    ("a resposta não é um JSON válido no formato pedido", "a resposta não é um JSON válido no formato pedido"),
])
def test_tipo_da_violacao(texto, tipo):
    assert tipo_da_violacao(texto) == tipo


def test_resumir_separa_status_origem_e_nivel(log):
    resumo = resumir(ler_log(log)[0])
    api, cache, misto = resumo["api"], resumo["cache"], resumo["misto"]
    assert (api["total"], cache["total"], misto["total"]) == (6, 2, 1)
    assert api["status"] == {"aprovado": 2, "corrigido": 2, "reprovado": 1,
                             "aprovado_com_aviso": 1}
    assert cache["status"] == {"corrigido": 2} and misto["status"] == {"corrigido": 1}
    # todas as tentativas contam
    assert api["violacoes"]["qualidade"] == {"texto sem acentos": 4}
    assert api["violacoes"]["seguranca"] == {"indica dose ou produto": 2,
                                             "cita doença fora do diagnóstico": 2}
    assert cache["violacoes"]["qualidade"] == {"texto sem acentos": 2}


def test_violacao_antiga_em_texto_puro_conta_como_seguranca():
    resumo = resumir([registro("corrigido", [["texto sem acentos (x)"], []], [False])])
    assert resumo["api"]["violacoes"]["seguranca"] == {"texto sem acentos": 1}
    assert not resumo["api"]["violacoes"]["qualidade"]


def test_mostra_no_maximo_5_violacoes_por_nivel_mais_frequentes_primeiro():
    regs = [registro("reprovado", [[{"violacao": f"tipo {c}", "nivel": "seguranca"}] * n],
                     [False]) for c, n in zip("abcdefg", (1, 7, 3, 3, 5, 2, 6))]
    texto = formatar(resumir(regs))
    linhas = [l.split() for l in texto.splitlines() if "tipo " in l]
    # 5 mais frequentes, na ordem; empate (c e d, 3 cada) em ordem alfabética
    assert [(int(l[0]), l[-1]) for l in linhas] == [(7, "b"), (6, "g"), (5, "e"), (3, "c"),
                                                    (3, "d")]


def test_formatar_mostra_as_secoes_e_as_porcentagens(log):
    texto = formatar(resumir(ler_log(log)[0]), ignoradas=2)
    assert "Laudos no log: 9 (sem cache: 6, do cache: 2, misto: 1)" in texto
    assert "Linhas ignoradas (inválidas): 2" in texto
    assert texto.index("Chamadas reais ao LLM (sem cache)") < texto.index("Do cache (repetições")
    assert "Misto (parte das tentativas do cache)" in texto
    secao_api = texto[texto.index("Chamadas reais"):texto.index("Do cache")]
    assert "Total de laudos: 6" in secao_api
    for rotulo, n, pct in (("Aprovados na 1a tentativa:", 2, "33%"),
                           ("Corrigidos na 2a tentativa:", 2, "33%"),
                           ("Aprovados com aviso:", 1, "17%"),
                           ("Reprovados (texto do sistema especialista):", 1, "17%")):
        linha = next(l for l in secao_api.splitlines() if rotulo in l)
        assert linha.split()[-2:] == [str(n), f"({pct})"], linha
    assert "4  texto sem acentos" in secao_api
    texto.encode("cp850")                                   # cabe no console do Windows


def test_sem_cache_nem_misto_as_secoes_sao_omitidas():
    texto = formatar(resumir([registro("aprovado", [[]], [False])]))
    assert "Do cache" not in texto and "Misto" not in texto
    assert "Violações de segurança mais frequentes: nenhuma" in " ".join(texto.split())


def test_principal_imprime_o_resumo(log):
    saida = []
    assert principal(["--log", str(log)], saida=saida.append) == 0
    assert "Estatísticas do guardrail do laudo" in saida[0] and "sem cache: 6" in saida[0]


def test_log_inexistente_ou_vazio(tmp_path):
    saida = []
    assert principal(["--log", str(tmp_path / "nao_existe.jsonl")], saida=saida.append) == 0
    assert "Log não encontrado" in saida[0]
    vazio = tmp_path / "vazio.jsonl"
    vazio.write_text("\n", encoding="utf-8")
    assert principal(["--log", str(vazio)], saida=saida.append) == 0
    assert "O log está vazio" in saida[1]


def test_opcao_desconhecida_sai_com_2(capsys):
    assert principal(["--versao", "4"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_nao_altera_o_log(log):
    antes = log.read_bytes()
    principal(["--log", str(log)], saida=lambda *_: None)
    assert log.read_bytes() == antes

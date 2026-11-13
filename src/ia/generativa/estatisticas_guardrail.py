"""Estatísticas do guardrail do laudo, a partir de rotina_demo/guardrail_log.jsonl.

Uso (na raiz do repositório, com src/ no PYTHONPATH):
    python -m ia.generativa.estatisticas_guardrail
    python -m ia.generativa.estatisticas_guardrail --log outro_log.jsonl

Só LÊ o log gravado por guardrail.registrar (uma linha JSON por laudo); não muda nada.
Imprime, para cada origem das respostas:
    - total de laudos e quantos foram aprovados na 1a tentativa, corrigidos na 2a,
      aprovados com aviso e reprovados (caíram no texto do sistema especialista);
    - as 5 violações mais frequentes por nível (segurança e qualidade), somando todas as
      tentativas.

Origem (campo "do_cache", uma entrada por tentativa):
    "api"    nenhuma tentativa veio do cache: chamadas reais ao LLM (o resumo principal);
    "cache"  todas vieram do cache: repetições de uma entrada já vista;
    "misto"  parte de cada (ex.: 1a do cache e correção chamada de verdade).
As repetições do cache não são novas respostas do LLM: contá-las junto infla os números.

Violações são agrupadas pelo TIPO, o texto antes do primeiro ": " ou " (": por exemplo,
'texto sem acentos ("nao", ...)' conta como "texto sem acentos". Violação gravada como
texto puro (sem nível) conta como segurança, como no guardrail. Linhas que não são JSON
válido, ou sem "status_guardrail", são ignoradas e contadas.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from common.config import GUARDRAIL_LOG
from ia.generativa.cliente import RAIZ_REPOSITORIO
from ia.generativa.guardrail import QUALIDADE, SEGURANCA

TOP = 5
ORIGENS = ("api", "cache", "misto")
TITULOS_ORIGEM = {"api": "Chamadas reais ao LLM (sem cache)",
                  "cache": "Do cache (repetições, não são respostas novas)",
                  "misto": "Misto (parte das tentativas do cache)"}
STATUS = (("aprovado", "Aprovados na 1a tentativa"),
          ("corrigido", "Corrigidos na 2a tentativa"),
          ("aprovado_com_aviso", "Aprovados com aviso"),
          ("reprovado", "Reprovados (texto do sistema especialista)"))
NOMES_NIVEL = {SEGURANCA: "segurança", QUALIDADE: "qualidade"}


# ---------------------------------------------------------------------------- leitura
def _caminho(caminho):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else RAIZ_REPOSITORIO / caminho


def ler_log(caminho):
    """(registros, linhas_ignoradas). Linha vazia não conta como ignorada."""
    registros, ignoradas = [], 0
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            try:
                registro = json.loads(linha)
            except ValueError:
                ignoradas += 1
                continue
            if not isinstance(registro, dict) or "status_guardrail" not in registro:
                ignoradas += 1
                continue
            registros.append(registro)
    return registros, ignoradas


# ----------------------------------------------------------------------------- resumo
def origem(registro):
    """"api", "cache" ou "misto", pelo do_cache de cada tentativa."""
    do_cache = registro.get("do_cache") or []
    if not isinstance(do_cache, list):
        do_cache = [do_cache]
    vieram = [v is True for v in do_cache]
    if vieram and all(vieram):
        return "cache"
    if any(vieram):
        return "misto"
    return "api"


def tipo_da_violacao(texto):
    """'cita doença fora do diagnóstico: X' -> 'cita doença fora do diagnóstico'."""
    texto = str(texto).strip()
    cortes = [i for i in (texto.find(": "), texto.find(" (")) if i > 0]
    return texto[:min(cortes)] if cortes else texto


def _violacoes(registro):
    """(nível, tipo) de cada violação de todas as tentativas do registro."""
    for tentativa in registro.get("violacoes") or []:
        for v in tentativa or []:
            if isinstance(v, dict):
                nivel = v.get("nivel") if v.get("nivel") in NOMES_NIVEL else SEGURANCA
                yield nivel, tipo_da_violacao(v.get("violacao", ""))
            else:                                     # formato antigo: texto puro
                yield SEGURANCA, tipo_da_violacao(v)


def resumir(registros):
    """{origem: {"total", "status": Counter, "violacoes": {nível: Counter}}}."""
    resumo = {o: {"total": 0, "status": Counter(),
                  "violacoes": {SEGURANCA: Counter(), QUALIDADE: Counter()}} for o in ORIGENS}
    for registro in registros:
        secao = resumo[origem(registro)]
        secao["total"] += 1
        secao["status"][registro.get("status_guardrail")] += 1
        for nivel, tipo in _violacoes(registro):
            secao["violacoes"][nivel][tipo] += 1
    return resumo


# ---------------------------------------------------------------------------- formato
def _pct(parte, total):
    return f"{100 * parte / total:.0f}%" if total else "-"


def _secao(titulo, secao):
    total = secao["total"]
    linhas = [titulo, f"  Total de laudos: {total}"]
    if not total:
        return linhas
    conhecidos = {s for s, _ in STATUS}
    for status, rotulo in STATUS:
        n = secao["status"][status]
        linhas.append(f"  {rotulo + ':':<46}{n:>5}  ({_pct(n, total)})")
    outros = sum(n for s, n in secao["status"].items() if s not in conhecidos)
    if outros:
        linhas.append(f"  {'Outro status:':<46}{outros:>5}  ({_pct(outros, total)})")
    for nivel in (SEGURANCA, QUALIDADE):
        contagem = secao["violacoes"][nivel]
        linhas.append(f"  Violações de {NOMES_NIVEL[nivel]} mais frequentes:")
        if not contagem:
            linhas.append("    nenhuma")
        # mais frequentes primeiro; empate em ordem alfabética (saída estável)
        for tipo, n in sorted(contagem.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]:
            linhas.append(f"    {n:>4}  {tipo}")
    return linhas


def formatar(resumo, ignoradas=0, caminho=None):
    linhas = ["Estatísticas do guardrail do laudo"]
    if caminho is not None:
        linhas.append(f"Log: {caminho}")
    total = sum(resumo[o]["total"] for o in ORIGENS)
    linhas.append(f"Laudos no log: {total} (sem cache: {resumo['api']['total']}, do cache: "
                  f"{resumo['cache']['total']}, misto: {resumo['misto']['total']})")
    if ignoradas:
        linhas.append(f"Linhas ignoradas (inválidas): {ignoradas}")
    for o in ORIGENS:
        if o != "api" and not resumo[o]["total"]:
            continue                                  # seção vazia de cache/misto: omitida
        linhas += [""] + _secao(TITULOS_ORIGEM[o], resumo[o])
    return "\n".join(linhas)


# ----------------------------------------------------------------------------- comando
def principal(argv=None, saida=print):
    parser = argparse.ArgumentParser(
        prog="python -m ia.generativa.estatisticas_guardrail",
        description="Resumo do log do guardrail do laudo.")
    parser.add_argument("--log", default=GUARDRAIL_LOG,
                        help="caminho do log (padrão: rotina_demo/guardrail_log.jsonl)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as erro:                        # opção desconhecida / --help
        return erro.code if isinstance(erro.code, int) else 2
    caminho = _caminho(args.log)
    if not caminho.is_file():
        saida(f"Log não encontrado: {caminho}. Rode a CLI com o laudo ligado para gerá-lo.")
        return 0
    registros, ignoradas = ler_log(caminho)
    if not registros:
        saida(f"O log está vazio: {caminho}.")
        return 0
    saida(formatar(resumir(registros), ignoradas, caminho))
    return 0


if __name__ == "__main__":
    for fluxo in (sys.stdout, sys.stderr):           # console antigo: nunca quebrar
        try:
            fluxo.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(principal())

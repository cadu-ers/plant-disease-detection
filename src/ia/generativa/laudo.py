"""Laudo em linguagem simples, gerado pelo LLM a partir do diagnóstico do sistema especialista.

    laudo = gerar_laudo(diagnostico, kb, cliente)      # diagnostico: ResultadoDiagnostico
    laudo.texto             # texto para o agricultor
    laudo.fonte             # "llm" ou "motor" (explicação do sistema especialista)
    laudo.status_guardrail  # "aprovado" | "corrigido" | "aprovado_com_aviso" | "reprovado"
                            # (None: o LLM não respondeu)

O que vai para o LLM (resumir_diagnostico): SÓ fatos que o motor já produziu (hipótese,
CF, se é conclusivo, regras que dispararam, top 3 da CNN, de onde veio a evidência (rede
neural x regras), medidas, pendências e as recomendações da base de conhecimento). Não vão
a imagem, o caminho, o nome do arquivo nem a classe real: o ResultadoDiagnostico não tem
nada disso.

Guardrail (ia/generativa/guardrail.py): o LLM responde em JSON (SCHEMA_LAUDO) e o sistema
especialista confere a resposta (validar_laudo). O laço é:
    1a geração -> validar -> sem violações: "aprovado";
    com violações (de segurança OU de qualidade) -> gera de novo UMA vez, dizendo ao LLM o
    que ele violou
        -> sem violações: "corrigido" (violacoes = as da 1a tentativa);
        -> só violações de QUALIDADE (ex.: sem acentos): "aprovado_com_aviso", fonte "llm"
           (violacoes = as que restaram);
        -> alguma de SEGURANÇA: "reprovado", fonte "motor" (violacoes = as da 2a).
    Se o LLM cair na hora da correção: a 1a resposta sai com aviso se só tinha violação de
    qualidade; senão, "reprovado".
Cada laudo em que o guardrail rodou vira uma linha em log (guardrail.registrar), se `log`
for um caminho. criar_gerador_de_laudo (a CLI) liga o log em GUARDRAIL_LOG.

gerar_laudo nunca levanta exceção por causa do LLM: sem chave, sem rede ou resposta
reprovada, devolve a explicação do sistema especialista com o motivo em `aviso`.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from common.config import GUARDRAIL_LOG
from ia.expert_system import chave_canonica
from ia.generativa.cliente import (RAIZ_REPOSITORIO, ClienteLLM, LLMIndisponivel,
                                   RespostaInvalida)
from ia.generativa.guardrail import (  # noqa: F401 (reexportadas: os testes e a CLI usam daqui)
    CAMPOS_TEXTO, SCHEMA_LAUDO, VIOLACAO_JSON_INVALIDO, doencas_fora_dos_fatos, montar_texto,
    Violacao, nomes_no_resumo, registrar, so_qualidade, validar_laudo,
)
from ia.generativa.prompts import (MODELO_CORRECAO_LAUDO, MODELO_PROMPT_LAUDO, SISTEMA_LAUDO,
                                   VERSAO_PROMPT_LAUDO)

MAX_PERGUNTAS = 3

# Só ASCII e letras do português (COM acento) chegam ao console (ver cli/formatacao.py).
_TROCAS = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-",
           "—": "-", "…": "...", "•": "-", " ": " ", "→": "->"}


@dataclass(frozen=True)
class Laudo:
    texto: str
    fonte: str                        # "llm" | "motor"
    modelo: Optional[str] = None      # modelo do LLM que respondeu (None se fonte == "motor")
    aviso: Optional[str] = None       # por que o LLM não foi usado, quando fonte == "motor"
    doencas_fora_dos_fatos: tuple = ()  # nomes citados pelo LLM que não estavam no resumo
    # corrigido: as da 1a tentativa; aprovado_com_aviso: as de qualidade que restaram;
    # reprovado: as da última. Cada uma é uma guardrail.Violacao (str com .nivel e .curto).
    violacoes: tuple = ()
    tentativas: int = 0               # respostas do LLM avaliadas pelo guardrail (0, 1 ou 2)
    # "aprovado" | "corrigido" | "aprovado_com_aviso" | "reprovado" | None (não rodou)
    status_guardrail: Optional[str] = None


# ------------------------------------------------------------------------- resumo (entrada)
def _nome(kb, hid):
    try:
        return kb.doencas[hid].nome
    except KeyError:
        return str(hid)


def _motivos_inconclusivo(resultado, kb):
    cfg, hips = kb.config, resultado.hipoteses
    if not hips:
        return ["nenhuma hipótese foi encontrada"]
    melhor = hips[0]
    motivos = []
    if melhor.cf_final < cfg["limiar_conclusivo"]:
        motivos.append(f"a força das evidências ({melhor.cf_final:.2f}) ficou abaixo do mínimo "
                       f"de {cfg['limiar_conclusivo']:.2f} para concluir")
    if len(hips) > 1 and melhor.cf_final - hips[1].cf_final < cfg["margem_minima"]:
        motivos.append(f"a segunda hipótese ({_nome(kb, hips[1].id)}) ficou muito próxima")
    return motivos


def _pergunta(texto):
    """Pendência do pipeline ('Pergunta sugerida: ...' ou 'id: pergunta') -> só a pergunta."""
    prefixo = "Pergunta sugerida: "
    if texto.startswith(prefixo):
        return texto[len(prefixo):]
    return texto.split(": ", 1)[-1]


def _origem_da_evidencia(resultado, hipotese_id, cfg):
    """De onde veio a força da hipótese principal: rede neural x regras (None sem hipótese).

    A força pela rede neural é peso_cnn * probabilidade, então nunca passa de peso_cnn
    (vai no resumo como "maximo_pela_rede_neural", para o LLM não chamar 0,5 de "moderada"
    quando a rede deu quase 100%). Regras que dispararam: só as A FAVOR (CF > 0)."""
    hip = resultado.hipotese(hipotese_id) if hipotese_id is not None else None
    if hip is None:
        return None
    return {
        "forca_pela_rede_neural": round(hip.cf_cnn, 2),
        "maximo_pela_rede_neural": round(cfg["peso_cnn"], 2),
        "forca_pelas_regras": round(hip.cf_regras, 2),
        "regras_que_dispararam": sum(1 for e in hip.evidencias
                                     if e.fonte == "regra" and e.cf > 0),
    }


def resumir_diagnostico(diag, kb):
    """Dicionário só com os fatos do diagnóstico, no formato que o prompt espera."""
    if getattr(diag, "status", None) != "ok":
        raise ValueError("o laudo exige um diagnóstico concluído (status 'ok')")
    r, c, cfg = diag.resultado, diag.conclusao, kb.config

    if c.hipotese_id is None:
        situacao = "sem_hipotese"
    else:
        situacao = "conclusivo" if c.conclusivo else "inconclusivo"

    principal = None
    recomendacoes, confundivel = [], []
    if c.hipotese_id is not None:
        d = kb.doencas[c.hipotese_id]
        principal = {"nome": d.nome, "agente": d.agente, "cf": round(c.cf, 2),
                     "urgencia": d.urgencia}
        recomendacoes = list(d.recomendacoes)
        confundivel = [_nome(kb, x) for x in d.confundivel_com]

    outras = [{"nome": _nome(kb, h.id), "cf": round(h.cf_final, 2)}
              for h in r.hipoteses[1:3] if h.cf_final >= cfg["limiar_exibicao"]]

    rede = [{"classe": legivel, "probabilidade_relativa": round(p, 2)}
            for _, legivel, p in diag.top3_cnn]
    concorda = None
    if diag.top3_cnn and c.hipotese_id is not None:
        try:
            concorda = chave_canonica(diag.top3_cnn[0][0]) == c.hipotese_id
        except KeyError:
            concorda = None

    evidencias = [r_.porque for r_ in diag.regras_disparadas
                  if c.hipotese_id is not None and c.hipotese_id in r_.hipoteses]

    medidas = None
    if diag.medidas is not None:
        m = diag.medidas
        medidas = {"area_fora_do_verde_saudavel_pct": round(100 * m.frac_lesao_total),
                   "area_marrom_pct": round(100 * m.frac_marrom),
                   "area_amarela_pct": round(100 * m.frac_amarelo),
                   "regioes_marrons": m.n_lesoes}

    conflito = None
    if r.conflito:
        conflito = {"regras_apontam": _nome(kb, r.conflito[0]),
                    "rede_neural_aponta": _nome(kb, r.conflito[1])}

    return {
        "situacao": situacao,
        "motivo_da_incerteza": _motivos_inconclusivo(r, kb) if situacao != "conclusivo" else [],
        "hipotese_principal": principal,
        "outras_hipoteses": outras,
        "pode_ser_confundida_com": confundivel,
        "rede_neural_top3": rede,
        "rede_neural_concorda_com_a_hipotese": concorda,
        "conflito_regras_x_rede_neural": conflito,
        "origem_da_evidencia": _origem_da_evidencia(r, c.hipotese_id, cfg),
        "evidencias_das_regras": evidencias,
        "medidas_da_folha": medidas,
        "mascara_da_folha_confiavel": bool(diag.qualidade.confiavel),
        "perguntas_pendentes": [_pergunta(p) for p in diag.pendencias][:MAX_PERGUNTAS],
        "recomendacoes": recomendacoes,
    }


# ----------------------------------------------------------------------- limpeza (saída)
def limpar_texto(texto):
    """Tira markdown simples e troca caracteres que o console do Windows não mostra.

    MANTÉM os acentos. Primeiro junta letra + acento combinante numa letra só (NFC): sem
    isso, um "não" escrito como "na" + til combinante perderia o til no filtro abaixo."""
    texto = unicodedata.normalize("NFC", texto)
    for de, para in _TROCAS.items():
        texto = texto.replace(de, para)
    texto = re.sub(r"\*\*(.+?)\*\*", r"\1", texto)
    texto = re.sub(r"^\s*#+\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"^(\s*)\*\s+", r"\1- ", texto, flags=re.MULTILINE)
    texto = "".join(ch for ch in texto if ch in "\n\t" or (32 <= ord(ch) < 256 and not 127 <= ord(ch) < 160))
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


# --------------------------------------------------------------------------- geração
def montar_prompt(resumo, violacoes_anteriores=()):
    """Prompt do laudo; na 2a tentativa, acrescenta a lista do que a 1a violou."""
    prompt = MODELO_PROMPT_LAUDO.format(
        resultado_json=json.dumps(resumo, ensure_ascii=False, indent=2))
    if violacoes_anteriores:
        prompt += MODELO_CORRECAO_LAUDO.format(
            violacoes="\n".join(f"- {v}" for v in violacoes_anteriores))
    return prompt


def _tentar(cliente, resumo, kb, temperatura, violacoes_anteriores=()):
    """Uma geração + validação -> (dados ou None, violações). LLMIndisponivel sobe."""
    try:
        dados = cliente.gerar_json(montar_prompt(resumo, violacoes_anteriores),
                                   schema=SCHEMA_LAUDO, sistema=SISTEMA_LAUDO,
                                   temperatura=temperatura)
    except RespostaInvalida:
        # a mensagem original repete a resposta do LLM: não vai para a tela nem para o log
        return None, [VIOLACAO_JSON_INVALIDO]
    violacoes = validar_laudo(dados, resumo, kb)
    if not violacoes:                       # ex.: um bloco só com emojis
        violacoes = [Violacao(f'campo "{c}" ficou vazio depois da limpeza')
                     for c in CAMPOS_TEXTO if not limpar_texto(dados[c])]
    return dados, violacoes


def _caminho_log(log):
    if log is None:
        return None
    caminho = Path(log)
    return caminho if caminho.is_absolute() else RAIZ_REPOSITORIO / caminho


def gerar_laudo(diag, kb, cliente=None, temperatura=0.3, log=None):
    """Laudo do LLM controlado pelo guardrail; sem LLM ou reprovado, a explicação do motor.

    `log`: caminho do registro do guardrail (JSON Lines) ou None para não registrar. Nunca
    levanta por causa do LLM nem do log."""
    resumo = resumir_diagnostico(diag, kb)
    cliente = cliente if cliente is not None else ClienteLLM()
    por_tentativa, do_cache = [], []

    def anotar(violacoes):
        por_tentativa.append(violacoes)
        do_cache.append(getattr(cliente, "veio_do_cache", None))

    def motor(aviso, status=None, violacoes=()):
        return Laudo(texto=diag.explicacao, fonte="motor", aviso=aviso,
                     violacoes=tuple(violacoes), tentativas=len(por_tentativa),
                     status_guardrail=status)

    def fechar(laudo):
        if laudo.status_guardrail is not None:
            registrar(_caminho_log(log), modelo=cliente.modelo_usado,
                      versao_prompt=VERSAO_PROMPT_LAUDO, situacao=resumo["situacao"],
                      status=laudo.status_guardrail, tentativas=laudo.tentativas,
                      violacoes_por_tentativa=por_tentativa, do_cache_por_tentativa=do_cache)
        return laudo

    try:
        dados, violacoes = _tentar(cliente, resumo, kb, temperatura)
    except LLMIndisponivel as erro:
        return motor(str(erro))
    anotar(violacoes)

    def do_llm(dados, status, violacoes):
        texto = limpar_texto(montar_texto(dados, limpar_texto))
        return fechar(Laudo(texto=texto, fonte="llm", modelo=cliente.modelo_usado,
                            violacoes=tuple(violacoes), tentativas=len(por_tentativa),
                            status_guardrail=status))

    if not violacoes:
        return do_llm(dados, "aprovado", ())

    dados_1a, violacoes_1a = dados, violacoes
    try:
        dados, violacoes = _tentar(cliente, resumo, kb, temperatura, violacoes_1a)
    except LLMIndisponivel as erro:
        if so_qualidade(violacoes_1a):           # a 1a era segura: sai com aviso
            return do_llm(dados_1a, "aprovado_com_aviso", violacoes_1a)
        return fechar(motor(
            f"o guardrail reprovou a 1a resposta e a correção não pôde ser pedida ({erro})",
            "reprovado", por_tentativa[0]))
    anotar(violacoes)
    if not violacoes:
        return do_llm(dados, "corrigido", por_tentativa[0])
    if so_qualidade(violacoes):
        return do_llm(dados, "aprovado_com_aviso", violacoes)
    return fechar(motor("o guardrail reprovou as duas respostas do LLM: "
                        + "; ".join(violacoes), "reprovado", violacoes))


def criar_gerador_de_laudo(cliente=None, log=GUARDRAIL_LOG, usar_cache=True):
    """(diagnostico, kb) -> Laudo, com UM cliente do LLM para a sessão inteira (para a CLI).

    Diferente de gerar_laudo, aqui o log do guardrail vem LIGADO (GUARDRAIL_LOG).
    usar_cache=False (--sem-cache na CLI): o cliente criado aqui não lê nem grava o cache.
    Só vale quando `cliente` não é passado."""
    cliente = cliente if cliente is not None else ClienteLLM(usar_cache=usar_cache)

    def gerar(diag, kb):
        return gerar_laudo(diag, kb, cliente, log=log)

    return gerar

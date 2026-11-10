"""Conversa: a resposta LIVRE do usuário vira sintomas, e o sistema especialista recalcula.

    sintomas = sintomas_para_perguntar(diag, kb)              # tuple de Sintoma (origem "usuario")
    interp = interpretar_resposta(texto, sintomas, cliente)   # Interpretacao
    novo = reavaliar(diag, segmentada, mascara, None, interp.sintomas, preditor, kb)
    conversar = criar_conversa(cliente)                       # para a CLI (log ligado)

Quem decide é sempre o SISTEMA ESPECIALISTA. O LLM só traduz o que o usuário escreveu em
graus de sintoma (0 a 1). Ele nunca escolhe a doença, e não recebe a hipótese, a rede
neural nem nome de doença: só os sintomas perguntados (id, descrição, pergunta) e a
resposta do usuário (prompts.TAREFA_CONVERSA).

Guardrail da conversa (validar_interpretacao, função pura): cada sintoma devolvido pelo LLM
só é aceito se
    a) o id foi perguntado;
    b) o valor é um número em [0, 1] (null = o usuário não falou disso: omitido, sem violação);
    c) o "trecho" que justifica o valor existe LITERALMENTE na resposta do usuário
       (comparação sem acento, sem diferença de maiúsculas e ignorando espaços extras).
       Sem isso, o LLM teria afirmado algo que o usuário não disse.
O resto vai para `descartados`, com o motivo. Resposta vazia não chama o LLM. Sem LLM (sem
chave, sem rede, JSON inválido), devolve uma Interpretacao sem sintomas e com `aviso`:
nunca levanta exceção.

Log (registrar_conversa): uma linha JSON por interpretação, se `log` for um caminho (a CLI
liga em common.config.CONVERSA_LOG; nos testes fica desligado). Grava só data/hora,
modelo, versão do prompt, ids perguntados, sintomas aceitos, descartados com motivo e
do_cache. NUNCA o texto do usuário nem os trechos (que são pedaços dele), caminho, classe
real ou seed. Falha ao gravar não quebra a conversa.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from aplicacao.pipeline import diagnosticar_amostra
from common.config import CONVERSA_LOG
from ia.expert_system import proxima_pergunta
from ia.generativa.cliente import RAIZ_REPOSITORIO, ClienteLLM, LLMIndisponivel, RespostaInvalida
from ia.generativa.guardrail import sem_acentos
from ia.generativa.prompts import (MODELO_PROMPT_CONVERSA, SISTEMA_CONVERSA,
                                   VERSAO_PROMPT_CONVERSA)

MAX_SINTOMAS = 4

# Motivos de descarte: textos fixos (nunca repetem texto do LLM nem do usuário).
MOTIVO_NAO_PERGUNTADO = "o sintoma não foi perguntado"
MOTIVO_FORMATO = "resposta do modelo fora do formato pedido"
MOTIVO_VALOR = "valor fora de [0, 1] ou não numérico"
MOTIVO_SEM_TRECHO = "o valor veio sem o trecho da resposta que o justifica"
MOTIVO_TRECHO = "não encontrei isso na sua resposta"
ID_DESCONHECIDO = "(id desconhecido)"     # id inventado pelo LLM: não vai para tela nem log

_ESPACOS = re.compile(r"\s+")
_PONTAS = " \t\n\"'“”‘’.,;:!?()[]"


@dataclass(frozen=True)
class Interpretacao:
    sintomas: dict = field(default_factory=dict)   # id -> grau em [0, 1] (só os aceitos)
    trechos: dict = field(default_factory=dict)    # id -> trecho da resposta do usuário
    descartados: tuple = ()                        # ((id, motivo), ...)
    perguntados: tuple = ()                        # ids enviados ao LLM, na ordem
    modelo: Optional[str] = None
    veio_do_cache: Optional[bool] = None
    aviso: Optional[str] = None                    # por que o LLM não foi usado


# ------------------------------------------------------------------------ o que perguntar
def sintomas_para_perguntar(diag, kb, maximo=MAX_SINTOMAS):
    """Sintomas (origem "usuario") que vale perguntar, o mais útil primeiro.

    1o: o de proxima_pergunta (o que mais separa as hipóteses líderes); depois os pendentes
    da hipótese principal, na ordem da base de conhecimento. Sem repetir."""
    if getattr(diag, "status", None) != "ok" or diag.resultado is None:
        return ()
    if diag.conclusao is None or diag.conclusao.hipotese_id is None or not diag.resultado.hipoteses:
        return ()
    ids = []
    sugerido = proxima_pergunta(diag.resultado, kb)
    if sugerido is not None:
        ids.append(sugerido.id)
    pendentes = diag.resultado.melhor.pendentes
    ids += [sid for sid in kb.sintomas if sid in pendentes]           # ordem estável (KB)
    saida = []
    for sid in ids:
        s = kb.sintomas.get(sid)
        if s is not None and s.origem == "usuario" and s not in saida:
            saida.append(s)
    return tuple(saida[:maximo])


# --------------------------------------------------------------------------------- prompt
def schema_conversa(sintomas):
    """JSON Schema: um campo por sintoma perguntado, cada um {valor, trecho}."""
    item = {
        "type": "object",
        "properties": {
            "valor": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1},
                                {"type": "null"}]},
            "trecho": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["valor", "trecho"],
    }
    ids = [s.id for s in sintomas]
    return {"type": "object", "properties": {sid: item for sid in ids}, "required": ids}


def montar_prompt_conversa(texto_usuario, sintomas):
    """Prompt da conversa: SÓ os sintomas (id, descrição, pergunta) e a resposta do usuário."""
    lista = [{"id": s.id, "descricao": s.descricao, "pergunta": s.pergunta} for s in sintomas]
    return MODELO_PROMPT_CONVERSA.format(
        sintomas_json=json.dumps(lista, ensure_ascii=False, indent=2),
        resposta=texto_usuario.strip())


# ------------------------------------------------------------------------------ guardrail
def _normalizar(texto):
    """Para comparar trechos: sem acento, minúsculas e espaços simples."""
    return _ESPACOS.sub(" ", sem_acentos(texto).lower()).strip()


def trecho_esta_na_resposta(trecho, texto_usuario):
    """O trecho aparece literalmente na resposta? (sem acento, sem caixa, sem espaços extras;
    aspas e pontuação nas pontas do trecho não contam)."""
    if not isinstance(trecho, str):
        return False
    alvo = _normalizar(trecho).strip(_PONTAS)
    return bool(alvo) and alvo in _normalizar(texto_usuario)


def _numero(valor):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    valor = float(valor)
    return valor if math.isfinite(valor) and 0.0 <= valor <= 1.0 else None


def validar_interpretacao(dados, ids_perguntados, texto_usuario):
    """(sintomas aceitos, trechos, descartados) a partir do JSON do LLM. Função pura."""
    if not isinstance(dados, dict):
        return {}, {}, ((ID_DESCONHECIDO, MOTIVO_FORMATO),)
    perguntados = list(ids_perguntados)
    aceitos, trechos, descartados = {}, {}, []
    for sid, item in dados.items():
        if sid not in perguntados:
            nome = sid if isinstance(sid, str) and re.fullmatch(r"[a-z_]{1,40}", sid) else ID_DESCONHECIDO
            descartados.append((nome, MOTIVO_NAO_PERGUNTADO))
            continue
        if not isinstance(item, dict):
            descartados.append((sid, MOTIVO_FORMATO))
            continue
        valor_bruto = item.get("valor")
        if valor_bruto is None:
            continue                                   # o usuário não falou disso: omitido
        valor = _numero(valor_bruto)
        if valor is None:
            descartados.append((sid, MOTIVO_VALOR))
            continue
        trecho = item.get("trecho")
        if not isinstance(trecho, str) or not trecho.strip():
            descartados.append((sid, MOTIVO_SEM_TRECHO))
            continue
        if not trecho_esta_na_resposta(trecho, texto_usuario):
            descartados.append((sid, MOTIVO_TRECHO))
            continue
        aceitos[sid] = valor
        trechos[sid] = trecho.strip().strip(_PONTAS)
    # na ordem em que foram perguntados
    ordem = {sid: i for i, sid in enumerate(perguntados)}
    aceitos = dict(sorted(aceitos.items(), key=lambda kv: ordem[kv[0]]))
    trechos = {sid: trechos[sid] for sid in aceitos}
    return aceitos, trechos, tuple(descartados)


# ---------------------------------------------------------------------------- interpretar
def _caminho(log):
    if log is None:
        return None
    caminho = Path(log)
    return caminho if caminho.is_absolute() else RAIZ_REPOSITORIO / caminho


def registrar_conversa(caminho, interp, agora=None):
    """Acrescenta uma linha JSON ao log da conversa. Nunca levanta; nunca grava o texto do
    usuário nem os trechos."""
    caminho = _caminho(caminho)
    if caminho is None:
        return False
    linha = {
        "data_hora": (agora or datetime.now()).isoformat(timespec="seconds"),
        "modelo": interp.modelo,
        "versao_prompt": VERSAO_PROMPT_CONVERSA,
        "perguntados": list(interp.perguntados),
        "aceitos": dict(interp.sintomas),
        "descartados": [{"sintoma": sid, "motivo": motivo} for sid, motivo in interp.descartados],
        "do_cache": interp.veio_do_cache,
    }
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError):
        return False


def interpretar_resposta(texto_usuario, sintomas, cliente=None, log=None, temperatura=0.0):
    """Resposta livre do usuário -> Interpretacao (graus validados). Nunca levanta por causa
    do LLM. Resposta vazia (ou nada a perguntar) não chama o LLM."""
    sintomas = tuple(sintomas)
    perguntados = tuple(s.id for s in sintomas)
    if not isinstance(texto_usuario, str) or not texto_usuario.strip() or not sintomas:
        return Interpretacao(perguntados=perguntados)
    cliente = cliente if cliente is not None else ClienteLLM()
    try:
        dados = cliente.gerar_json(montar_prompt_conversa(texto_usuario, sintomas),
                                   schema=schema_conversa(sintomas), sistema=SISTEMA_CONVERSA,
                                   temperatura=temperatura)
    except LLMIndisponivel as erro:
        return Interpretacao(perguntados=perguntados, aviso=str(erro))
    except RespostaInvalida:
        # a mensagem original repete a resposta do LLM: não vai para a tela
        interp = Interpretacao(perguntados=perguntados, modelo=cliente.modelo_usado,
                               veio_do_cache=getattr(cliente, "veio_do_cache", None),
                               descartados=((ID_DESCONHECIDO, MOTIVO_FORMATO),),
                               aviso="o modelo não devolveu um JSON válido")
        registrar_conversa(log, interp)
        return interp
    aceitos, trechos, descartados = validar_interpretacao(dados, perguntados, texto_usuario)
    interp = Interpretacao(sintomas=aceitos, trechos=trechos, descartados=descartados,
                           perguntados=perguntados, modelo=cliente.modelo_usado,
                           veio_do_cache=getattr(cliente, "veio_do_cache", None))
    registrar_conversa(log, interp)
    return interp


def criar_conversa(cliente=None, log=CONVERSA_LOG, usar_cache=True):
    """(texto_usuario, sintomas) -> Interpretacao, com UM cliente do LLM para a sessão (CLI).

    Diferente de interpretar_resposta, aqui o log vem LIGADO (CONVERSA_LOG). usar_cache=False
    (--sem-cache) só vale quando `cliente` não é passado."""
    cliente = cliente if cliente is not None else ClienteLLM(usar_cache=usar_cache)

    def conversar(texto_usuario, sintomas):
        return interpretar_resposta(texto_usuario, sintomas, cliente, log=log)

    return conversar


# ------------------------------------------------------------------------------ reavaliar
def reavaliar(diag_anterior, segmentada, mascara, divisao=None, sintomas=None, preditor=None,
              kb=None):
    """Novo ResultadoDiagnostico com os sintomas informados pelo usuário.

    Reaproveita o preditor já carregado (a CNN dá o mesmo resultado para a mesma imagem) e
    a divisão do diagnóstico anterior. Quem recalcula é o sistema especialista."""
    if divisao is None:
        divisao = getattr(diag_anterior, "divisao", None)
    return diagnosticar_amostra(segmentada, mascara, divisao, preditor=preditor, kb=kb,
                                sintomas_usuario=dict(sintomas or {}))

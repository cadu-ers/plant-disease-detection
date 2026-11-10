"""Guardrail do laudo: o sistema especialista controla o que o LLM pode dizer.

O LLM responde em JSON (SCHEMA_LAUDO), e não em texto livre. Cada bloco do laudo vem num
campo, e o LLM também DECLARA o que usou: a hipótese citada, a situação, os índices das
recomendações e das perguntas. validar_laudo confere tudo isso contra o resumo que o
sistema especialista produziu (laudo.resumir_diagnostico) e devolve a lista de violações.

    violacoes = validar_laudo(dados, resumo, kb)     # [] = aprovado
    texto = montar_texto(dados)                      # blocos com os títulos de TITULOS_LAUDO

O laço (gerar, validar, corrigir uma vez, ou voltar para o texto do motor) fica em
laudo.gerar_laudo. Aqui só há funções puras e o registro em disco (registrar).

Níveis (Violacao.nivel): só "texto sem acentos" é de QUALIDADE; todo o resto é de
SEGURANÇA (inclusive o formato do JSON, que impede montar o texto). As duas pedem a 2a
tentativa; se na 2a só restar violação de qualidade, o laudo sai "aprovado_com_aviso".

Checagens (cada violação é um texto curto, em português, para a tela e para o log):
    - formato: todos os campos do schema, com os tipos certos e os textos não vazios;
    - hipotese_citada é o nome exato da hipótese principal (null quando não há);
    - situacao_declarada é a situação do motor;
    - os índices de recomendacoes_usadas e perguntas_usadas existem no resumo;
    - nenhuma doença da base fora dos fatos (doencas_fora_dos_fatos);
    - nenhum "%" e nenhum padrão de dose ou produto (ml, g, kg, L, /ha, dose, fungicida...);
    - o texto tem acentos (palavras como "nao" ou "diagnostico" sem acento são violação);
    - se nenhuma regra disparou a favor da hipótese (origem_da_evidencia): o texto diz que
      nenhuma regra confirmou e não chama a força das regras de fraca, baixa ou pequena
      (força zero não é força fraca);
    - se não é conclusivo: ação irreversível (destruir, arrancar, eliminar, remover,
      retirar ou queimar plantas) só na mesma frase de uma condição ("se o diagnóstico for
      confirmado...", "após a confirmação"...);
    - se não é conclusivo e há perguntas pendentes: "perguntas" não pode vir vazia.

Acentos: as comparações usam sem_acentos (os dois lados), para pegar também o texto que o
LLM escreveu sem acento. O texto MOSTRADO nunca passa por sem_acentos: montar_texto usa os
campos como o LLM escreveu (laudo.py só aplica limpar_texto).

Vazamento: as mensagens de violação NUNCA repetem texto livre do LLM (só nomes de campos,
índices e nomes de doença da base). O log (registrar) grava só data/hora, modelo, versão
do prompt, situação, status, tentativas, violações e se cada resposta veio do cache: nunca
caminho, classe real ou seed.
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from ia.generativa.prompts import TITULOS_LAUDO

SITUACOES = ("conclusivo", "inconclusivo", "sem_hipotese")
CAMPOS_TEXTO = ("o_que_encontrou", "por_que", "o_que_fazer")
STATUS = ("aprovado", "corrigido", "aprovado_com_aviso", "reprovado")

# Níveis de violação. As duas pedem a 2a tentativa; na 2a, só "seguranca" reprova.
SEGURANCA = "seguranca"   # o texto pode enganar ou pôr a lavoura em risco
QUALIDADE = "qualidade"   # o texto está certo, mas mal escrito (ex.: sem acentos)
NIVEIS = (SEGURANCA, QUALIDADE)


class Violacao(str):
    """Texto curto da violação (é um str: compara, junta e vai para o JSON como texto) com
    o `nivel` ("seguranca" ou "qualidade") e um `curto` para a linha da CLI."""

    def __new__(cls, texto, nivel=SEGURANCA, curto=None):
        if nivel not in NIVEIS:
            raise ValueError(f"nível de violação inválido: {nivel!r}")
        obj = super().__new__(cls, texto)
        obj.nivel = nivel
        obj.curto = curto or str(texto)
        return obj


def nivel_de(violacao):
    """Nível de uma violação; texto comum (sem nível) conta como segurança, por precaução."""
    return getattr(violacao, "nivel", SEGURANCA)


def so_qualidade(violacoes):
    """True se há violações e TODAS são de qualidade (o laudo pode sair com aviso)."""
    return bool(violacoes) and all(nivel_de(v) == QUALIDADE for v in violacoes)

SCHEMA_LAUDO = {
    "type": "object",
    "properties": {
        "o_que_encontrou": {"type": "string"},
        "por_que": {"type": "string"},
        "perguntas": {"type": "array", "items": {"type": "string"}},
        "o_que_fazer": {"type": "string"},
        "hipotese_citada": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "situacao_declarada": {"type": "string", "enum": list(SITUACOES)},
        "recomendacoes_usadas": {"type": "array", "items": {"type": "integer"}},
        "perguntas_usadas": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["o_que_encontrou", "por_que", "perguntas", "o_que_fazer", "hipotese_citada",
                 "situacao_declarada", "recomendacoes_usadas", "perguntas_usadas"],
}

VIOLACAO_JSON_INVALIDO = Violacao("a resposta não é um JSON válido no formato pedido")

# Dose, quantidade ou classe de produto. Número + unidade (2 ml, 1,5 kg, 10 L, 3g), por
# área (/ha, por hectare) e as palavras que indicam produto químico.
_PADROES_DOSE = (
    (re.compile(r"\d+(?:[.,]\d+)?\s*(?:ml|mg|g|kg|l)\b", re.IGNORECASE), "quantidade com unidade"),
    (re.compile(r"/\s*ha\b|\bpor hectare\b", re.IGNORECASE), "quantidade por hectare"),
    (re.compile(r"\bdoses?\b", re.IGNORECASE), '"dose"'),
    (re.compile(r"\b(?:fungicida|inseticida|bactericida|acaricida|herbicida)s?\b",
                re.IGNORECASE), "nome de classe de produto"),
)
# Ação irreversível sobre a PLANTA INTEIRA: verbo (qualquer forma) e "planta(s)" a até 4
# palavras de distância, em qualquer ordem ("destrua as plantas", "as plantas devem ser
# removidas"). Se no meio houver folha, parte, ramo, galho, fruto ou restos, a ação é sobre
# um pedaço da planta e não conta ("remova as folhas afetadas da planta"). "Eliminar restos
# culturais" ou "remover folhas" também não contam (não falam de planta).
# "arranc" + "arranq" (arranque, arranquem: o c vira qu antes de e).
# Na ordem inversa (planta antes do verbo) só vale o particípio da voz passiva ("as plantas
# devem ser removidas"); "das plantas e queime os restos" não é ação sobre a planta.
# Todos os padrões abaixo são aplicados ao texto SEM acentos (sem_acentos): por isso estão
# escritos sem acento ("apos a confirmacao", "nao", "destruid").
_VERBO = r"\b(?:destru|arranc|arranq|elimin|remov|retir|queim)\w*"
_PARTICIPIO = r"\b(?:destruid|arrancad|eliminad|removid|retirad|queimad)[oa]s?\b"
_PLANTA = r"\bplantas?\b"
_ENTRE = (r"(?:\W+(?!(?:folhas?|partes?|ramos?|galhos?|frutos?|restos?)\b)\w+){0,4}?\W+")
_ACAO_IRREVERSIVEL = re.compile(
    f"{_VERBO}{_ENTRE}{_PLANTA[2:]}|{_PLANTA}{_ENTRE}{_PARTICIPIO[2:]}", re.IGNORECASE)
_CONDICAO = re.compile(
    r"\bse\b[^.!?]{0,60}?\bfor(?:em)? confirmad"          # se (o diagnóstico) for confirmado
    r"|\bcaso\b[^.!?]{0,60}?\bseja(?:m)? confirmad"      # caso (o diagnóstico) seja confirmado
    r"|\bapos a confirmacao|\bdepois de confirmad|\bdepois da confirmacao",
    re.IGNORECASE)
_FIM_DE_FRASE = re.compile(r"(?<=[.!?])\s+|\n+")

# Palavras que, em português correto, SEMPRE têm acento (ou ç). Escritas sem acento no
# texto ORIGINAL indicam que o LLM tirou os acentos (já aconteceu no modo JSON do Gemini).
# Aplicado ao texto original, e não ao texto sem acentos.
_SEM_ACENTO = re.compile(
    r"\b(?:nao|sao|estao|entao|voce|voces|tambem|diagnosticos?|tecnicos?|agronomos?|"
    r"agricolas?|analises?|hipoteses?|evidencias?|minimo|necessari[oa]s?|tipic[oa]s?|"
    r"orientacao|confirmacao|forca|maos|virus|sintomatic[oa]s?|possivel|provavel)\b",
    re.IGNORECASE)

# Zero regras a favor: força ZERO não é "fraca". O texto tem de dizer que nenhuma regra
# confirmou, e não pode descrever as regras como fracas, baixas ou pequenas.
_REGRA = re.compile(r"\bregras?\b", re.IGNORECASE)
_FRACO = re.compile(r"\b(?:frac|baix|pequen)[oa]s?\b", re.IGNORECASE)
_NENHUMA_REGRA = re.compile(
    r"\bnenhuma\s+(?:das\s+)?regras?\b"                       # nenhuma (das) regra(s)
    r"|\bnao\b[^.!?]{0,25}?\bregras?\b[^.!?]{0,30}?\bconfirm"  # não houve regra que confirmasse
    r"|\bregras?\b[^.!?]{0,30}?\bnao\b[^.!?]{0,15}?\bconfirm", # as regras não confirmaram
    re.IGNORECASE)


# ------------------------------------------------------------- doenças fora dos fatos
def _apelidos(nome):
    """'Pinta-preta do tomateiro (Early blight)' -> {'pinta-preta do tomateiro', 'early blight', ...}."""
    nome = nome.lower().strip()
    apelidos = {nome}
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", nome)
    if m:
        apelidos.update(p.strip() for p in m.groups() if len(p.strip()) >= 4)
    return apelidos


def nomes_no_resumo(resumo):
    """Nomes de doença que aparecem no resumo enviado ao LLM."""
    nomes = []
    if resumo["hipotese_principal"]:
        nomes.append(resumo["hipotese_principal"]["nome"])
    nomes += [h["nome"] for h in resumo["outras_hipoteses"]]
    nomes += resumo["pode_ser_confundida_com"]
    nomes += [c["classe"] for c in resumo["rede_neural_top3"]]
    if resumo["conflito_regras_x_rede_neural"]:
        nomes += list(resumo["conflito_regras_x_rede_neural"].values())
    return nomes


def doencas_fora_dos_fatos(texto, resumo, kb):
    """Doenças da KB citadas no texto que não estavam no resumo (possível alucinação)."""
    # comparação sem acento dos dois lados: "Acaro-rajado" também é o "Ácaro-rajado"; o nome
    # devolvido é o da base, com acento
    texto = sem_acentos(texto).lower()
    permitidos = set()
    for nome in nomes_no_resumo(resumo):
        permitidos |= _apelidos(sem_acentos(nome))
    fora = []
    for d in kb.doencas.values():
        proprios = _apelidos(sem_acentos(d.nome)) - permitidos  # apelido compartilhado não conta
        if any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", texto) for a in proprios):
            fora.append(d.nome)
    return tuple(sorted(set(fora)))


# ----------------------------------------------------------------------------- validação
def _violacoes_de_formato(dados):
    """Campos ausentes ou com tipo errado. Se houver, as outras checagens não rodam."""
    if not isinstance(dados, dict):
        return [VIOLACAO_JSON_INVALIDO]
    erros = []
    for campo in CAMPOS_TEXTO:
        if not isinstance(dados.get(campo), str) or not dados[campo].strip():
            erros.append(f'campo "{campo}" ausente ou vazio')
    if not isinstance(dados.get("perguntas"), list) or not all(
            isinstance(p, str) for p in dados["perguntas"]):
        erros.append('campo "perguntas" deve ser uma lista de textos')
    if dados.get("hipotese_citada") is not None and not isinstance(dados["hipotese_citada"], str):
        erros.append('campo "hipotese_citada" deve ser texto ou null')
    if "hipotese_citada" not in dados:
        erros.append('campo "hipotese_citada" ausente')
    if dados.get("situacao_declarada") not in SITUACOES:
        erros.append('campo "situacao_declarada" ausente ou inválido')
    for campo in ("recomendacoes_usadas", "perguntas_usadas"):
        valor = dados.get(campo)
        # bool é subclasse de int em Python: True não vale como índice
        if not isinstance(valor, list) or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in valor):
            erros.append(f'campo "{campo}" deve ser uma lista de números inteiros')
    return erros


def _textos(dados):
    return [dados[c] for c in CAMPOS_TEXTO] + list(dados["perguntas"])


def sem_acentos(texto):
    """'Não há força' -> 'Nao ha forca'. SÓ para comparar: nunca vai para a tela.

    O LLM às vezes escreve sem acento ("nao", "apos a confirmacao", "Acaro-rajado"); as
    checagens comparam os dois lados sem acento para não deixar isso escapar."""
    return "".join(c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c))


def _frases(texto):
    """Frases do texto, já SEM acentos (só para as comparações)."""
    return [f for f in _FIM_DE_FRASE.split(sem_acentos(texto)) if f.strip()]


def palavras_sem_acento(textos):
    """Palavras da lista _SEM_ACENTO escritas sem o acento (ex.: "nao", "diagnostico")."""
    achadas = []
    for t in textos:
        for m in _SEM_ACENTO.finditer(t):
            palavra = m.group(0).lower()
            if palavra not in achadas:
                achadas.append(palavra)
    return achadas


def acoes_irreversiveis_sem_condicao(texto):
    """Quantas frases mandam destruir/arrancar/eliminar/remover/retirar/queimar plantas sem
    uma condição de confirmação na mesma frase."""
    return sum(1 for f in _frases(texto)
               if _ACAO_IRREVERSIVEL.search(f) and not _CONDICAO.search(f))


def regras_chamadas_de_fracas(textos):
    """Frases que falam das regras como fracas/baixas/pequenas (errado quando nenhuma disparou)."""
    return [f for t in textos for f in _frases(t)
            if _REGRA.search(f) and _FRACO.search(f) and not _NENHUMA_REGRA.search(f)]


def validar_laudo(dados, resumo, kb):
    """Lista de violações (Violacao: texto curto + nível) do JSON do LLM contra o resumo do
    motor. Texto sem nível explícito vira violação de SEGURANÇA."""
    return [v if isinstance(v, Violacao) else Violacao(v) for v in _validar(dados, resumo, kb)]


def _validar(dados, resumo, kb):
    erros = _violacoes_de_formato(dados)
    if erros:
        return erros

    principal = resumo["hipotese_principal"]
    esperado = principal["nome"] if principal else None
    if dados["hipotese_citada"] != esperado:
        erros.append("hipotese_citada não é a hipótese principal do sistema especialista"
                     if esperado else "hipotese_citada deveria ser null (não há hipótese principal)")

    situacao = resumo["situacao"]
    if dados["situacao_declarada"] != situacao:
        erros.append(f'situacao_declarada deveria ser "{situacao}"')

    for campo, lista in (("recomendacoes_usadas", resumo["recomendacoes"]),
                         ("perguntas_usadas", resumo["perguntas_pendentes"])):
        invalidos = sorted({i for i in dados[campo] if not 0 <= i < len(lista)})
        if invalidos:
            erros.append(f"{campo} tem índice(s) inexistente(s): "
                         + ", ".join(str(i) for i in invalidos))

    textos = _textos(dados)
    juntos = "\n".join(textos)
    fora = doencas_fora_dos_fatos(juntos, resumo, kb)
    if fora:
        erros.append("cita doença fora do diagnóstico: " + "; ".join(fora))
    if "%" in juntos:
        erros.append('usa "%" (força da evidência não é porcentagem)')
    for padrao, descricao in _PADROES_DOSE:
        if padrao.search(sem_acentos(juntos)):
            erros.append(f"indica dose ou produto ({descricao})")
    sem_acento = palavras_sem_acento(textos)
    if sem_acento:
        # as palavras citadas vêm da lista fixa _SEM_ACENTO, não do texto livre do LLM
        erros.append(Violacao("texto sem acentos (" + ", ".join(f'"{p}"' for p in sem_acento[:3])
                              + "); escreva em português com acentos",
                              nivel=QUALIDADE, curto="texto sem acentos"))

    origem = resumo.get("origem_da_evidencia")
    if origem and origem.get("regras_que_dispararam") == 0:
        if regras_chamadas_de_fracas(textos):
            erros.append('chama a força das regras de "fraca", "baixa" ou "pequena", mas '
                         "nenhuma regra disparou")
        if not _NENHUMA_REGRA.search(sem_acentos(dados["por_que"] + "\n"
                                                 + dados["o_que_encontrou"])):
            erros.append('"por_que" não diz que nenhuma regra confirmou a hipótese')

    if situacao != "conclusivo":
        if acoes_irreversiveis_sem_condicao(dados["o_que_fazer"]):
            erros.append('"o_que_fazer" manda destruir, arrancar, eliminar, remover, retirar '
                         "ou queimar plantas sem a condição de confirmação (o caso não é "
                         "conclusivo)")
        if resumo["perguntas_pendentes"] and not [p for p in dados["perguntas"] if p.strip()]:
            erros.append('"perguntas" está vazia, mas há perguntas pendentes')
    return erros


# ------------------------------------------------------------------------- texto final
def montar_texto(dados, limpar=lambda t: t):
    """Texto do laudo a partir dos blocos do JSON (os mesmos títulos do prompt)."""
    encontrou, por_que, perguntas_titulo, fazer = TITULOS_LAUDO
    blocos = [(encontrou, limpar(dados["o_que_encontrou"])),
              (por_que, limpar(dados["por_que"]))]
    perguntas = [limpar(p) for p in dados["perguntas"]]
    perguntas = [p.lstrip("- ").strip() for p in perguntas if p.strip()]
    if perguntas:
        blocos.append((perguntas_titulo, "\n".join(f"- {p}" for p in perguntas)))
    blocos.append((fazer, limpar(dados["o_que_fazer"])))
    return "\n\n".join(f"{titulo}\n{corpo}" for titulo, corpo in blocos)


# ---------------------------------------------------------------------------- registro
def registrar(caminho, *, modelo, versao_prompt, situacao, status, tentativas,
              violacoes_por_tentativa, do_cache_por_tentativa=(), agora=None):
    """Acrescenta uma linha JSON ao log do guardrail. Nunca levanta: o log é conveniência.

    "violacoes" e "do_cache" são listas alinhadas por tentativa (do_cache: a resposta veio
    do cache do cliente, sem chamar a API; None se não se sabe). Cada violação é gravada
    como {"violacao": texto, "nivel": "seguranca" | "qualidade"}. Para a visão crítica,
    conte só as linhas sem cache, ou rode a avaliação com --sem-cache.
    Só estes campos: nada de caminho de imagem, classe real ou seed."""
    if caminho is None:
        return False
    linha = {
        "data_hora": (agora or datetime.now()).isoformat(timespec="seconds"),
        "modelo": modelo,
        "versao_prompt": versao_prompt,
        "situacao": situacao,
        "status_guardrail": status,
        "tentativas": tentativas,
        "violacoes": [[{"violacao": str(v), "nivel": nivel_de(v)} for v in tentativa]
                      for tentativa in violacoes_por_tentativa],
        "do_cache": list(do_cache_por_tentativa),
    }
    try:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError):
        return False

"""Formatação das telas da CLI: funções PURAS (dados -> texto).

Nenhuma função daqui lê ou grava arquivo, imprime, sorteia ou importa o TensorFlow. Elas só
recebem os objetos já prontos do pipeline e devolvem texto.

Vazamento: formatar_sorteio e formatar_diagnostico NÃO recebem a ImagemSorteada nem a
Comparacao (o ResultadoDiagnostico não tem caminho nem classe real). Só formatar_revelacao,
chamada depois do Enter, recebe o caminho, a classe real e a seed.

Texto: só ASCII e letras acentuadas do português (todas existem em cp850 e cp1252, os
códigos do console do Windows). Nada de setas, travessões, aspas curvas, sinais de maior ou
igual nem caracteres de desenhar caixas.

Legendas das medidas (conferidas em processamento_imagens/medidas_lesao.py):
    - as proporções têm como base a ÁREA DA FOLHA segmentada (a máscara), não a imagem toda;
    - "fora do verde saudável" (frac_lesao_total) inclui marrom, amarelo, pálido e sombra;
    - as "regiões marrons" são componentes conexos da faixa de cor marrom com pelo menos
      MIN_LESION_AREA_PX pixels, e não uma segmentação de lesão validada;
    - a distância à borda vai de 0 (na borda da FOLHA) a 1 (ponto mais interno da folha) e é a
      média só dos pixels das regiões marrons; não existe sem região marrom.
"""

import textwrap
from pathlib import Path

from aplicacao.pipeline import AVISO_CF
from common.config import MIN_LESION_AREA_PX, SMALL_LESION_MAX_RATIO
from ia.generativa.prompts import TITULOS_LAUDO

LARGURA = 78

PROMPT_MENU = "Escolha uma opção: "
PROMPT_REVELAR = "Pressione Enter para revelar a imagem e comparar com a classe real... "
PROMPT_OUTRA = "Analisar outra? (s/n): "

MSG_OPCAO_INVALIDA = "Opção inválida. Digite 1 para analisar uma imagem ou 0 para sair."
MSG_RESPOSTA_INVALIDA = "Resposta inválida. Digite s (sim) ou n (não)."
MSG_CARREGANDO_MODELO = "Carregando modelo (pode levar alguns segundos)..."
MSG_MODELO_CARREGADO = "Modelo carregado."
MSG_ANALISANDO = "Analisando..."
MSG_GERANDO_LAUDO = "Gerando o laudo em linguagem simples (IA generativa)..."
MSG_ATE_LOGO = "Até logo."
MSG_ENCERRANDO = "Encerrando."
# Genérica de propósito: o texto original da exceção pode trazer caminho ou nome de arquivo.
MSG_SORTEIO_INVALIDO = ("Não foi possível sortear uma imagem de teste. Confira se o manifest do "
                        "split e a pasta data/ estão completos e tente de novo.")


# ---------------------------------------------------------------------------------- utilitários
def _titulo(texto):
    return f" {texto} ".center(LARGURA, "=")


def separador():
    return "-" * LARGURA


def _item(texto, recuo=4, marcador="- "):
    """Texto com marcador, quebrado em LARGURA colunas com recuo de continuação."""
    return textwrap.fill(texto, width=LARGURA, initial_indent=" " * recuo + marcador,
                         subsequent_indent=" " * (recuo + len(marcador)), break_long_words=False)


def _paragrafo(texto, recuo=4):
    return textwrap.fill(texto, width=LARGURA, initial_indent=" " * recuo,
                         subsequent_indent=" " * recuo, break_long_words=False)


def _quebrar_linhas(texto, recuo=4):
    """Quebra cada linha do texto preservando a indentação que ela já tinha."""
    saida = []
    for linha in texto.splitlines():
        if not linha.strip():
            saida.append("")
            continue
        espacos = len(linha) - len(linha.lstrip())
        base = " " * (recuo + espacos)
        saida.append(textwrap.fill(linha.strip(), width=LARGURA, initial_indent=base,
                                   subsequent_indent=base + "  ", break_long_words=False))
    return "\n".join(saida)


def _linha(rotulo, valor, largura_rotulo=34, recuo=4):
    pontos = "." * max(2, largura_rotulo - len(rotulo))
    linha = f"{' ' * recuo}{rotulo} {pontos} {valor}"
    if len(linha) <= LARGURA:
        return linha
    # valor longo (ex.: caminho): rótulo numa linha e o valor na seguinte, sem quebrar no meio
    return f"{' ' * recuo}{rotulo}:" + "\n" + f"{' ' * (recuo + 4)}{valor}"


def _campo(rotulo, valor, quebrar=True):
    """Campo 'Rótulo: valor'. Se não couber: quebra por palavras, ou (caminhos) valor na linha seguinte."""
    linha = f"{rotulo}: {valor}"
    if len(linha) <= LARGURA:
        return linha
    if quebrar:
        return textwrap.fill(linha, width=LARGURA, subsequent_indent="    ",
                             break_long_words=False)
    return f"{rotulo}:" + "\n" + f"    {valor}"


def _nome(kb, hipotese_id):
    try:
        return kb.doencas[hipotese_id].nome
    except KeyError:
        return str(hipotese_id)


def _caminho_curto(caminho, raiz):
    """Caminho relativo à raiz do repositório, se estiver dentro dela."""
    if caminho is None:
        return ""
    if raiz is not None:
        try:
            return Path(caminho).relative_to(Path(raiz)).as_posix()
        except ValueError:
            pass
    return str(caminho)


# ------------------------------------------------------------------------------------- telas
def formatar_menu():
    return "\n".join([
        "=" * LARGURA,
        " Diagnóstico de doenças em folhas de tomate e batata - demonstração ".center(LARGURA),
        "=" * LARGURA,
        "  1) Analisar imagem aleatória",
        "  0) Sair",
    ])


def formatar_sorteio(divisao):
    """Só a divisão. Sem caminho, nome de arquivo ou classe real."""
    return f"Imagem sorteada do conjunto de teste.\nDivisão: {divisao}"


def formatar_aviso_modelo(motivo):
    return "\n".join([
        f"AVISO: a CNN não está disponível ({motivo}).",
        _paragrafo("Esta sessão continua SEM a CNN: o diagnóstico usa só as medidas da folha e "
                   "as regras do sistema especialista, e por isso é mais limitado. Nenhuma "
                   "probabilidade da CNN é inventada, e resultado inconclusivo não significa "
                   "folha saudável.", recuo=0),
    ])


def formatar_erro_inesperado(erro):
    """Só o tipo do erro: a mensagem poderia conter caminho ou nome de arquivo."""
    return (f"Ocorreu um erro inesperado ({type(erro).__name__}). A análise foi cancelada; "
            "você voltou ao menu.")


def motivo_inconclusivo(resultado, kb):
    """Lista de motivos (texto) pelos quais o resultado não é conclusivo, pelos limiares da KB."""
    cfg = kb.config
    hipoteses = resultado.hipoteses
    if not hipoteses:
        return ["nenhuma hipótese foi encontrada"]
    melhor = hipoteses[0]
    motivos = []
    if melhor.cf_final < cfg["limiar_conclusivo"]:
        motivos.append(f"o CF ({melhor.cf_final:.2f}) está abaixo do mínimo de "
                       f"{cfg['limiar_conclusivo']:.2f} para concluir")
    segunda = hipoteses[1] if len(hipoteses) > 1 else None
    margem = melhor.cf_final - (segunda.cf_final if segunda else 0.0)
    if margem < cfg["margem_minima"]:
        if segunda:
            motivos.append(f"a diferença para a 2a hipótese ({_nome(kb, segunda.id)}, CF "
                           f"{segunda.cf_final:.2f}) é de {margem:.2f}, abaixo da margem mínima "
                           f"de {cfg['margem_minima']:.2f}")
        else:
            motivos.append(f"a margem ({margem:.2f}) está abaixo do mínimo de "
                           f"{cfg['margem_minima']:.2f}")
    return motivos or ["os critérios da base de conhecimento não foram atendidos"]


def _secao_cnn(diag):
    linhas = ["[1] CNN - 3 classes mais prováveis"]
    if not diag.top3_cnn:
        linhas.append(_paragrafo("A CNN não foi usada nesta análise (veja os avisos). O "
                                 "diagnóstico abaixo usa só medidas e regras."))
        return linhas
    linhas.append(_paragrafo("(saída do softmax: pontuação relativa do modelo, não é "
                             "confiança calibrada)"))
    for i, (_, legivel, p) in enumerate(diag.top3_cnn, start=1):
        linhas.append(f"    {i}. {legivel:<58} {p:6.1%}")
    return linhas


def _secao_medidas(diag):
    q, m = diag.qualidade, diag.medidas
    linhas = ["[2] Medidas da folha"]
    if q.confiavel:
        linhas.append(_paragrafo(
            f"Máscara: confiável (a folha ocupa {q.leaf_fraction:.0%} da imagem; "
            f"{q.edge_contact:.0%} das bordas da imagem estão dentro da máscara)."))
    else:
        linhas.append(_paragrafo(f"Máscara: NÃO confiável - {q.motivo}."))
    if m is None:
        linhas.append(_paragrafo("Medidas não calculadas. Os sintomas que dependem delas ficam "
                                 "como sem informação (nunca viram zero)."))
        return linhas
    linhas += [
        _linha("Fora do verde saudável", f"{m.frac_lesao_total:.0%}"),
        _linha("Faixa de cor marrom", f"{m.frac_marrom:.0%}"),
        _linha("Faixa de cor amarela", f"{m.frac_amarelo:.0%}"),
        _linha("Regiões marrons", f"{m.n_lesoes} (pequenas: {m.n_lesoes_pequenas})"),
        _linha("Distância média à borda da folha",
               "não calculada (sem região marrom)" if m.dist_borda_media_lesoes is None
               else f"{m.dist_borda_media_lesoes:.2f}"),
        "    Legenda:",
        _item("As proporções são da área da folha segmentada, não da imagem inteira.", 6),
        _item('"Fora do verde saudável" inclui marrom, amarelo, pixels pálidos e sombras.', 6),
        _item("Regiões marrons: grupos de pixels da faixa de cor marrom com pelo menos "
              f"{MIN_LESION_AREA_PX} pixels (candidatas a lesão; não é uma segmentação de lesão "
              f"validada). Pequenas: até {SMALL_LESION_MAX_RATIO:.1%} da área da folha.", 6),
        _item("Distância à borda: 0 = na borda da folha, 1 = ponto mais interno da folha "
              "(média dos pixels das regiões marrons).", 6),
    ]
    return linhas


def _secao_avisos(diag):
    linhas = ["[3] Avisos"]
    linhas += [_item(a) for a in diag.avisos] or ["    Nenhum aviso."]
    return linhas


def _condicoes_por_regra(resultado):
    mapa = {}
    for h in resultado.hipoteses:
        for e in h.evidencias:
            if e.fonte == "regra":
                mapa.setdefault(e.regra_id, e.condicoes)
    return mapa


def _secao_regras(diag, kb):
    linhas = ["[4] Regras disparadas"]
    if not diag.regras_disparadas:
        linhas.append("    Nenhuma regra disparou com os dados disponíveis.")
        return linhas
    condicoes = _condicoes_por_regra(diag.resultado)
    for r in diag.regras_disparadas:
        nomes = ", ".join(_nome(kb, h) for h in r.hipoteses)
        linhas.append(textwrap.fill(
            f"{r.regra_id} (grau {r.grau:.2f}, CF da regra {r.cf:+.2f}) -> {nomes}",
            width=LARGURA, initial_indent="    ", subsequent_indent="        ",
            break_long_words=False))
        linhas.append(_paragrafo(r.porque, recuo=8))
        evid = ", ".join(f"{'não ' if negado else ''}{nome}={valor:.2f}"
                         for nome, valor, negado in condicoes.get(r.regra_id, ()))
        if evid:
            linhas.append(_paragrafo(f"Evidência: {evid}", recuo=8))
    return linhas


def _secao_pendencias(diag):
    linhas = ["[5] Informações que ajudariam a decidir",
              "    (a CLI não pergunta: você não viu a foto)"]
    if not diag.pendencias:
        linhas.append("    Nenhuma informação adicional sugerida.")
        return linhas
    prefixo = "Pergunta sugerida: "
    for texto in diag.pendencias:
        if texto.startswith(prefixo):
            linhas.append(_item("Mais útil: " + texto[len(prefixo):]))
        else:                                   # formato "id_do_sintoma: pergunta"
            linhas.append(_item(texto.split(": ", 1)[-1]))
    return linhas


def _secao_conclusao(diag, kb):
    c, r = diag.conclusao, diag.resultado
    linhas = ["[6] Conclusão"]
    if c.hipotese_id is None:
        linhas.append(_paragrafo(
            "Nenhuma hipótese atingiu a evidência mínima (CF mínimo de exibição: "
            f"{kb.config['limiar_exibicao']:.2f}). Situação: NÃO CONCLUSIVA. Recomenda-se "
            "avaliação por um agrônomo."))
    else:
        linhas.append(_paragrafo(f"Hipótese principal: {c.nome}  (CF = {c.cf:.2f})"))
        linhas.append("    Situação: " + ("CONCLUSIVA" if c.conclusivo else "NÃO CONCLUSIVA"))
        if not c.conclusivo:
            linhas.append(_paragrafo("Motivo: " + "; ".join(motivo_inconclusivo(r, kb)) + ".",
                                     recuo=4))
        outras = [h for h in r.hipoteses[1:3] if h.cf_final >= kb.config["limiar_exibicao"]]
        for h in outras:
            linhas.append(f"    Outra hipótese: {_nome(kb, h.id)}  (CF = {h.cf_final:.2f})")
    linhas.append(_paragrafo(AVISO_CF))
    return linhas


def formatar_diagnostico(diag, kb):
    """Tela completa do diagnóstico. Não contém caminho, nome de arquivo nem classe real."""
    cabecalho = [_titulo("DIAGNÓSTICO")]
    if diag.status != "ok":
        return "\n".join(cabecalho + [
            _paragrafo("Não foi possível concluir o diagnóstico desta imagem.", recuo=0),
            _paragrafo(str(diag.mensagem_erro), recuo=0),
            *( [""] + _secao_avisos(diag) if diag.avisos else [] ),
        ])
    secoes = [_secao_cnn(diag), _secao_medidas(diag), _secao_avisos(diag),
              _secao_regras(diag, kb), _secao_pendencias(diag), _secao_conclusao(diag, kb),
              ["[7] Explicação do sistema especialista", _quebrar_linhas(diag.explicacao)]]
    linhas = cabecalho
    for secao in secoes:
        linhas += ["", *secao]
    return "\n".join(linhas)


def formatar_revelacao(imagem, diag, comparacao, gravacao, seed, kb, raiz=None):
    """Comparação com a classe real. SÓ pode ser chamada depois do Enter de revelação."""
    linhas = [_titulo("COMPARAÇÃO COM A CLASSE REAL"),
              _campo("Imagem", _caminho_curto(imagem.caminho, raiz), quebrar=False)]

    classe = imagem.classe_real
    canonica = getattr(comparacao, "classe_real_canonica", None)
    legivel = _nome(kb, canonica) if canonica else None
    linhas.append(_campo("Classe real", f"{classe} - {legivel}" if legivel else classe))

    if comparacao is None:
        linhas.append(_paragrafo("Comparação indisponível: o diagnóstico não foi concluído.",
                                 recuo=0))
    else:
        if not diag.top3_cnn:
            cnn = "não utilizada nesta análise"
        else:
            cnn = diag.top3_cnn[0][1]
            if comparacao.cnn_top1_coincide is None:
                cnn += " - classe real sem tradução"
            else:
                cnn += " - " + ("coincide" if comparacao.cnn_top1_coincide else "NÃO coincide")
        linhas.append(_campo("CNN (1a opção)", cnn))

        if comparacao.hipotese_especialista is None:
            esp = "nenhuma hipótese atingiu a evidência mínima"
        else:
            esp = _nome(kb, comparacao.hipotese_especialista)
            if comparacao.especialista_coincide is None:
                esp += " - classe real sem tradução"
            else:
                esp += " - " + ("coincide" if comparacao.especialista_coincide else "NÃO coincide")
            if not comparacao.especialista_conclusivo:
                esp += " (resultado NÃO conclusivo)"
        linhas.append(_campo("Especialista", esp))

    if gravacao is not None and gravacao.ok:
        linhas.append(_campo("Figuras salvas em", _caminho_curto(gravacao.pasta, raiz), quebrar=False))
    else:
        erro = gravacao.erro if gravacao is not None else "gravação não executada"
        linhas.append(_paragrafo(f"As figuras NÃO foram salvas: {erro}", recuo=0))
        if gravacao is not None and gravacao.pasta is not None:
            linhas.append(_campo("Pasta criada", _caminho_curto(gravacao.pasta, raiz), quebrar=False))
    linhas.append(_campo("Seed", f"{seed} (identifica este sorteio)"))
    if diag.top3_cnn:
        linhas.append(_paragrafo(
            "Lembrete: esta comparação é só uma demonstração, não uma avaliação da CNN: os pesos "
            "locais podem ter visto imagens do teste (veja o aviso).", recuo=0))
    return "\n".join(linhas)


def formatar_laudo(laudo):
    """Laudo da IA generativa (ia/generativa/laudo.py). Também é mostrado ANTES do Enter: o
    laudo é gerado só a partir do ResultadoDiagnostico, que não tem caminho nem classe real."""
    linhas = [_titulo("LAUDO EM LINGUAGEM SIMPLES (IA GENERATIVA)")]
    guardrail = formatar_guardrail(laudo)
    if laudo.status_guardrail == "reprovado":
        linhas += [guardrail, "", _quebrar_linhas(laudo.texto, recuo=4)]
        return "\n".join(linhas)
    if laudo.fonte != "llm":
        linhas.append(_paragrafo(
            f"IA generativa indisponível nesta análise ({laudo.aviso}). A explicação do sistema "
            "especialista está na seção [7] acima.", recuo=0))
        return "\n".join(linhas)
    linhas.append(_paragrafo(
        f"Texto gerado pelo Gemini (modelo {laudo.modelo}) só a partir do diagnóstico acima. "
        "É texto de IA e pode conter erros: confira com as seções [1] a [7]. Quem decide o "
        "diagnóstico é o sistema especialista.", recuo=0))
    if guardrail:
        linhas.append(guardrail)
    if laudo.doencas_fora_dos_fatos:
        alerta = ("ATENÇÃO: o texto abaixo cita doença(s) que NÃO estão no diagnóstico ("
                  + "; ".join(laudo.doencas_fora_dos_fatos) + "). Desconsidere essa parte.")
        # sem quebrar nomes com hífen ("Ácaro-rajado") no meio
        linhas += ["", textwrap.fill(alerta, width=LARGURA, break_long_words=False,
                                     break_on_hyphens=False)]
    linhas += ["", _blocos_do_laudo(laudo.texto)]
    return "\n".join(linhas)


def formatar_guardrail(laudo):
    """Linha do guardrail (ia/generativa/guardrail.py), ou "" quando ele não rodou.

    As violações são textos curtos gerados pelo código (nomes de campo, índices e nomes de
    doença da base), nunca texto livre do LLM: podem aparecer antes do Enter."""
    status = getattr(laudo, "status_guardrail", None)
    violacoes = "; ".join(getattr(laudo, "violacoes", ()) or ())
    if status == "aprovado":
        texto = f"Guardrail: aprovado na {laudo.tentativas}a tentativa"
    elif status == "corrigido":
        texto = f"Guardrail: corrigido na {laudo.tentativas}a tentativa (violações: {violacoes})"
    elif status == "aprovado_com_aviso":
        # só sobraram violações de qualidade: mostra a forma curta ("texto sem acentos")
        curtas = []
        for v in getattr(laudo, "violacoes", ()) or ():
            curta = getattr(v, "curto", str(v))
            if curta not in curtas:
                curtas.append(curta)
        texto = f"Guardrail: aprovado com aviso ({'; '.join(curtas)})"
    elif status == "reprovado":
        texto = ("Guardrail: reprovado: mostrando o texto do sistema especialista "
                 f"(violações: {violacoes})")
    else:
        return ""
    # sem quebrar nomes com hífen ("Ácaro-rajado") no meio
    return textwrap.fill(texto, width=LARGURA, subsequent_indent="    ",
                         break_long_words=False, break_on_hyphens=False)


def _titulo_do_laudo(linha):
    """(título canônico, resto da linha) se a linha começa com um título de bloco; senão None.

    Tolera variações comuns do LLM: maiúsculas, "Por que" sem acento e texto na mesma linha
    do título ("Por quê: porque...")."""
    baixa = linha.lower()
    for titulo in TITULOS_LAUDO:
        for variante in {titulo.lower(), titulo.lower().replace("quê", "que")}:
            if baixa.startswith(variante):
                return titulo, linha[len(variante):].strip()
    return None


def _blocos_do_laudo(texto, recuo_titulo=4, recuo_texto=6):
    """Texto do laudo por blocos: título com recuo 4; parágrafos e linhas "- " com recuo 6.

    Linhas seguidas de um mesmo parágrafo são juntadas e quebradas de novo em LARGURA, sem
    recuo pendurado. Linha em branco antes de cada título (exceto o primeiro)."""
    saida, paragrafo = [], []
    base = " " * recuo_texto

    def fechar_paragrafo():
        if paragrafo:
            saida.append(textwrap.fill(" ".join(paragrafo), width=LARGURA, initial_indent=base,
                                       subsequent_indent=base, break_long_words=False))
            paragrafo.clear()

    for bruta in texto.splitlines():
        linha = bruta.strip()
        if not linha:
            fechar_paragrafo()
            continue
        titulo = _titulo_do_laudo(linha)
        if titulo is not None:
            fechar_paragrafo()
            if saida:
                saida.append("")
            saida.append(" " * recuo_titulo + titulo[0])
            if titulo[1]:
                paragrafo.append(titulo[1])
        elif linha.startswith("- "):
            fechar_paragrafo()
            saida.append(textwrap.fill(linha[2:].strip(), width=LARGURA,
                                       initial_indent=base + "- ",
                                       subsequent_indent=base + "  ", break_long_words=False))
        else:
            paragrafo.append(linha)
    fechar_paragrafo()
    return "\n".join(saida)


# --------------------------------------------------------------------------------- conversa
# Só DEPOIS da revelação: o usuário já pode abrir a figura salva e olhar a folha.
PROMPT_RESPOSTA = "Sua resposta: "
PROMPT_CONFIRMA = "Confirma? (s/n): "
MSG_INTERPRETANDO = "Interpretando a sua resposta (IA generativa)..."
MSG_CONVERSA_PULADA = "Conversa pulada: o diagnóstico continua o mesmo."
MSG_NADA_A_RECALCULAR = ("Nenhum sintoma foi entendido com segurança na sua resposta: o "
                         "diagnóstico continua o mesmo.")
MSG_NAO_RECALCULADO = "Não confirmado: o diagnóstico continua o mesmo."
MSG_RECALCULANDO = "Recalculando o diagnóstico com as suas respostas (sistema especialista)..."


def formatar_perguntas_conversa(sintomas, pasta=None, raiz=None):
    """Convite + perguntas numeradas. `pasta`: onde as figuras foram salvas (ou None)."""
    if pasta is not None:
        onde = f"Olhe a folha em {_caminho_curto(Path(pasta) / 'original.png', raiz)}"
    else:
        onde = "Olhe a folha (as figuras não foram salvas nesta análise)"
    linhas = [_titulo("CONVERSA"),
              _paragrafo(f"{onde} e responda em texto livre às perguntas abaixo (Enter vazio "
                         "para pular):", recuo=0)]
    for i, s in enumerate(sintomas, start=1):
        linhas.append(textwrap.fill(f"{i}. {s.pergunta}", width=LARGURA, initial_indent="  ",
                                    subsequent_indent="     ", break_long_words=False))
    return "\n".join(linhas)


def _grau_em_palavras(valor):
    if valor >= 1.0:
        return "sim"
    if valor <= 0.0:
        return "não"
    return f"parcial ({valor:.1f})".replace(".", ",")


def formatar_interpretacao(interp, kb):
    """"Entendi:" com os sintomas aceitos (descrição + sim/não/parcial + trecho) e os
    descartados com o motivo. Os ids desconhecidos (inventados pelo LLM) não aparecem."""
    if interp.aviso:
        return _paragrafo(f"IA generativa indisponível ({interp.aviso}). Nada foi "
                          "recalculado.", recuo=0)
    linhas = ["Entendi:"]
    for sid, valor in interp.sintomas.items():
        descricao = kb.sintomas[sid].descricao if sid in kb.sintomas else sid
        trecho = interp.trechos.get(sid, "")
        linhas.append(_item(f'{descricao}: {_grau_em_palavras(valor)} ("{trecho}")'))
    if not interp.sintomas:
        linhas.append(_item("nenhum sintoma."))
    descartados = [(sid, motivo) for sid, motivo in interp.descartados]
    if descartados:
        linhas.append("Não usei:")
        for sid, motivo in descartados:
            nome = kb.sintomas[sid].descricao if sid in kb.sintomas else "um sintoma não perguntado"
            linhas.append(_item(f"{nome}: {motivo}"))
    return "\n".join(linhas)


def _situacao(conclusao):
    if conclusao is None or conclusao.hipotese_id is None:
        return "nenhuma hipótese", None, "NÃO CONCLUSIVA"
    return conclusao.nome, conclusao.cf, "CONCLUSIVA" if conclusao.conclusivo else "NÃO CONCLUSIVA"


def formatar_antes_depois(antes, depois, kb):
    """Hipótese, CF e situação antes e depois das respostas, e as regras que passaram a
    disparar (quem recalculou foi o sistema especialista)."""
    linhas = [_titulo("ANTES x DEPOIS")]
    for rotulo, diag in (("Antes", antes), ("Depois", depois)):
        nome, cf, situacao = _situacao(diag.conclusao)
        cf_txt = "" if cf is None else f" (CF = {cf:.2f})"
        linhas.append(textwrap.fill(f"{rotulo}: {nome}{cf_txt} - {situacao}", width=LARGURA,
                                    subsequent_indent="        ", break_long_words=False,
                                    break_on_hyphens=False))
    ja = {r.regra_id for r in antes.regras_disparadas}
    novas = [r for r in depois.regras_disparadas if r.regra_id not in ja]
    linhas.append("Regras que passaram a disparar:")
    if not novas:
        linhas.append("    Nenhuma.")
    for r in novas:
        nomes = ", ".join(_nome(kb, h) for h in r.hipoteses)
        linhas.append(textwrap.fill(f"{r.regra_id} (CF da regra {r.cf:+.2f}) -> {nomes}",
                                    width=LARGURA, initial_indent="    ",
                                    subsequent_indent="        ", break_long_words=False))
        linhas.append(_paragrafo(r.porque, recuo=8))
    return "\n".join(linhas)


def formatar_aviso_conversa(motivo):
    """Falha inesperada na conversa (só o tipo do erro)."""
    return _paragrafo(f"Não foi possível concluir a conversa ({motivo}). O diagnóstico acima "
                      "continua válido.", recuo=0)


def formatar_aviso_laudo(motivo):
    """Falha inesperada ao gerar o laudo (só o tipo do erro: a mensagem poderia ter dados)."""
    return _paragrafo(f"Não foi possível gerar o laudo da IA generativa ({motivo}). O "
                      "diagnóstico acima continua válido.", recuo=0)

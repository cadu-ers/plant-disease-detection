"""Explicação do diagnóstico em texto: por que, por que não, conflitos e recomendações."""


def _cond(c):
    nome, valor, negado = c
    return f"{'não ' if negado else ''}{nome}={valor:.2f}"


def _linhas_evidencias(h, sinal):
    linhas = []
    for e in sorted(h.evidencias, key=lambda e: -abs(e.cf)):
        if (e.cf > 0) != (sinal > 0):
            continue
        if e.fonte == "cnn":
            linhas.append(f"  - CNN (p={e.grau:.2f}, CF {e.cf:+.2f}): {e.porque}")
        else:
            linhas.append(f"  - {e.regra_id} (grau {e.grau:.2f}, CF {e.cf:+.2f}): {e.porque}")
            linhas.append(f"      condições: {', '.join(_cond(c) for c in e.condicoes)}")
    return linhas


def _por_que_nao(h, kb):
    d = kb.doencas[h.id]
    linhas = [f"Por que não {d.nome}? (CF {h.cf_final:+.2f})"]
    a_favor = _linhas_evidencias(h, +1)
    if a_favor:
        linhas.append("  Evidências a favor, mas insuficientes:")
        linhas += a_favor
    contra = _linhas_evidencias(h, -1)
    if contra:
        linhas.append("  Evidências contra:")
        linhas += contra
    pendentes = sorted(h.pendentes)
    if pendentes:
        linhas.append("  Poderia ser confirmada se soubéssemos: " + ", ".join(pendentes))
    elif not a_favor:
        fracas = [n for n in h.nao_disparadas if n.limitante and not n.desconhecidos]
        if fracas:
            n = max(fracas, key=lambda n: n.grau)
            linhas.append(f"  Nenhuma regra disparou; a mais próxima foi {n.regra_id}, "
                          f"barrada por {n.limitante[0]} (grau {n.limitante[1]:.2f}).")
        else:
            linhas.append("  Nenhuma regra disparou com os sintomas observados.")
    return linhas


def explicar(resultado, kb, comparar=2):
    cfg = kb.config
    if not resultado.hipoteses or resultado.melhor.cf_final < cfg["limiar_exibicao"]:
        return ("Nenhuma hipótese atingiu evidência mínima. Resultado INCONCLUSIVO: "
                "recomenda-se avaliação por um agrônomo.")

    top = resultado.melhor
    d = kb.doencas[top.id]
    status = "conclusivo" if resultado.conclusivo else "INCONCLUSIVO"
    linhas = [f"Diagnóstico mais provável: {d.nome}",
              f"Confiança (CF): {top.cf_final:.2f} [{status}]"]
    if d.agente:
        linhas.append(f"Agente: {d.agente}")

    linhas.append("")
    linhas.append("Por quê:")
    linhas += _linhas_evidencias(top, +1) or ["  - (sem evidências positivas)"]
    contra = _linhas_evidencias(top, -1)
    if contra:
        linhas.append("Evidências contra:")
        linhas += contra

    if resultado.conflito:
        r, c = resultado.conflito
        linhas += ["", f"ATENÇÃO: regras e CNN divergem. As regras apontam {kb.doencas[r].nome}; "
                       f"a CNN aponta {kb.doencas[c].nome}."]

    rivais = [h for h in resultado.hipoteses[1:1 + comparar]
              if h.cf_final >= cfg["limiar_exibicao"] or h.id in d.confundivel_com]
    for h in rivais:
        linhas += ["", *_por_que_nao(h, kb)]

    confundiveis = [kb.doencas[c].nome for c in d.confundivel_com
                    if (resultado.hipotese(c) is not None and resultado.hipotese(c).cf_final >= cfg["limiar_exibicao"])]
    if confundiveis:
        linhas += ["", "Observação: pode ser confundida com " + ", ".join(confundiveis) + "."]

    if not resultado.conclusivo:
        linhas += ["", "Resultado inconclusivo: responda às perguntas sugeridas ou procure "
                       "avaliação de um agrônomo."]

    if d.recomendacoes:
        linhas += ["", f"Recomendações gerais (urgência: {d.urgencia}):"]
        linhas += [f"  - {r}" for r in d.recomendacoes]
        linhas.append("  - Esta é uma orientação geral e não substitui a avaliação de um agrônomo.")
    return "\n".join(linhas)

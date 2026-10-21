"""Busca heurística sobre as hipóteses e escolha da próxima pergunta.

1. As hipóteses ficam numa fila de prioridade ordenada pelo CF final (melhor
   primeiro, como numa busca best-first).
2. Se o resultado ainda não é conclusivo, o sistema pergunta o sintoma que mais
   ajuda a separar as hipóteses líderes. A heurística: para cada sintoma ainda
   desconhecido, soma sobre os pares de líderes (i, j)
       peso_i * peso_j * |influencia(sintoma, i) - influencia(sintoma, j)|
   em que ``influencia`` resume o quanto a presença do sintoma favorece (+) ou
   contraria (-) a hipótese segundo as regras.
3. Repete até ficar conclusivo, não haver pergunta útil ou estourar o limite.
"""

from .engine import diagnosticar


def influencia(kb, sintoma, hipotese):
    """Quanto a presença do sintoma favorece (+) ou contraria (-) a hipótese, em [-1, 1]."""
    total = 0.0
    for r in kb.regras_de(hipotese):
        if sintoma in r.se:
            total += r.cf
        elif sintoma in r.se_nao:
            total -= r.cf
    return max(-1.0, min(1.0, total))


def pontuar_pergunta(kb, sintoma, resultado, top_k=3):
    lideres = resultado.hipoteses[:top_k]
    pesos = [max(h.cf_final, 0.0) + 0.05 for h in lideres]
    infl = [influencia(kb, sintoma, h.id) for h in lideres]
    pontos = 0.0
    for i in range(len(lideres)):
        for j in range(i + 1, len(lideres)):
            pontos += pesos[i] * pesos[j] * abs(infl[i] - infl[j])
    return pontos


def proxima_pergunta(resultado, kb, ja_perguntados=(), top_k=3):
    """Devolve o Sintoma mais informativo ainda desconhecido, ou None."""
    melhor, melhor_pontos = None, 1e-9
    for s in kb.sintomas.values():
        if s.id in resultado.sintomas or s.id in ja_perguntados:
            continue
        pontos = pontuar_pergunta(kb, s.id, resultado, top_k)
        if pontos > melhor_pontos:
            melhor, melhor_pontos = s, pontos
    return melhor


def diagnostico_interativo(fatos, kb, perguntar, max_perguntas=5, usar_cnn=True):
    """Diagnostica e, enquanto o resultado não for conclusivo, faz perguntas.

    ``perguntar(sintoma)`` deve devolver um grau em [0, 1] ou None ("não sei").
    Devolve (resultado, historico), com historico = [(id_do_sintoma, resposta), ...].
    """
    fatos = {**fatos, "sintomas": dict(fatos.get("sintomas") or {})}
    historico, perguntados = [], set()

    resultado = diagnosticar(fatos, kb, usar_cnn)
    while not resultado.conclusivo and len(historico) < max_perguntas:
        sintoma = proxima_pergunta(resultado, kb, perguntados)
        if sintoma is None:
            break
        perguntados.add(sintoma.id)
        resposta = perguntar(sintoma)
        historico.append((sintoma.id, resposta))
        if resposta is not None:
            fatos["sintomas"][sintoma.id] = resposta
            resultado = diagnosticar(fatos, kb, usar_cnn)
    return resultado, historico

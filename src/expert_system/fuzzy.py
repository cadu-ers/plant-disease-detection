"""Fuzzificação: converte medidas numéricas da imagem em graus de sintoma [0, 1].

Cada sintoma "de imagem" na base de conhecimento declara a ``medida`` de onde
vem e um trapézio ``[a, b, c, d]``: grau 0 abaixo de ``a``, sobe até 1 em ``b``,
fica em 1 até ``c`` e desce a 0 em ``d``.
"""


def trapezio(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzificar(medidas, kb):
    """{medida: valor} -> {sintoma: grau} apenas para sintomas cuja medida foi fornecida."""
    graus = {}
    for s in kb.sintomas.values():
        if s.medida and s.medida in medidas:
            graus[s.id] = trapezio(float(medidas[s.medida]), *s.trapezio)
    return graus

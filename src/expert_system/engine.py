"""Motor de inferência: encadeamento para frente com fatores de certeza (CF).

Contrato de entrada (``fatos``), todos os campos opcionais:

    {
      "cultura": "tomate" | "batata",
      "medidas": {"frac_marrom": 0.12, ...},     # medidas das máscaras (fuzzificadas)
      "sintomas": {"aneis_concentricos": 1.0},   # graus 0..1 informados diretamente
      "cnn_prior": {"Tomato_Early_blight": 0.7}  # probabilidades da CNN (nomes do dataset
                                                 # ou chaves canônicas)
    }

Sintoma ausente do dicionário = desconhecido (diferente de 0 = ausente). Uma regra
que depende de um sintoma desconhecido não dispara, e esse sintoma vira candidato
a pergunta no modo interativo (ver busca.py).
"""

from dataclasses import dataclass, field
from typing import Optional

from .classes import chave_canonica, cultura_de, normalizar_cultura
from .fuzzy import fuzzificar
from .kb import BaseConhecimento

CF_MAX = 0.999


# ---------------------------------------------------------------------------
# Álgebra de fatores de certeza (estilo MYCIN)
# ---------------------------------------------------------------------------
def combinar_cf(a, b):
    """Combina duas evidências sobre a mesma hipótese."""
    a = max(-CF_MAX, min(CF_MAX, a))
    b = max(-CF_MAX, min(CF_MAX, b))
    if a >= 0 and b >= 0:
        return a + b * (1 - a)
    if a <= 0 and b <= 0:
        return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))


def combinar(cfs):
    """Combina uma lista de CFs sem depender da ordem: acumula as positivas e as
    negativas separadamente e só então as confronta."""
    pos = neg = 0.0
    for cf in cfs:
        if cf > 0:
            pos = combinar_cf(pos, cf)
        elif cf < 0:
            neg = combinar_cf(neg, -cf)
    if pos == 0.0 and neg == 0.0:
        return 0.0
    return combinar_cf(pos, -neg)


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------
@dataclass
class Evidencia:
    fonte: str                 # "regra" ou "cnn"
    regra_id: Optional[str]
    cf: float                  # CF efetivo (já multiplicado pelo grau)
    grau: float                # grau da regra, ou probabilidade da CNN
    porque: str
    condicoes: tuple = ()      # ((sintoma, valor, negado), ...)


@dataclass
class RegraNaoDisparada:
    regra_id: str
    grau: float
    limitante: Optional[tuple]  # (sintoma, grau) da condição mais fraca conhecida
    desconhecidos: tuple


@dataclass
class Hipotese:
    id: str
    cf_final: float = 0.0
    cf_regras: float = 0.0
    cf_cnn: float = 0.0
    evidencias: list = field(default_factory=list)
    nao_disparadas: list = field(default_factory=list)
    pendentes: set = field(default_factory=set)  # sintomas que, se respondidos, poderiam disparar regras


@dataclass
class Resultado:
    cultura: Optional[str]
    hipoteses: list            # ordenadas por cf_final decrescente
    conclusivo: bool
    conflito: Optional[tuple]  # (melhor_pelas_regras, melhor_pela_cnn) quando divergem
    sintomas: dict
    usou_cnn: bool

    @property
    def melhor(self):
        return self.hipoteses[0] if self.hipoteses else None

    def hipotese(self, hid):
        return next((h for h in self.hipoteses if h.id == hid), None)


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------
def _preparar(fatos, kb, usar_cnn):
    sintomas = fuzzificar(fatos.get("medidas") or {}, kb)
    for nome, valor in (fatos.get("sintomas") or {}).items():
        if nome not in kb.sintomas:
            raise ValueError(f"Sintoma desconhecido: {nome!r}")
        sintomas[nome] = float(valor)
    sintomas = {k: max(0.0, min(1.0, v)) for k, v in sintomas.items()}

    prior = {}
    if usar_cnn:
        for nome, p in (fatos.get("cnn_prior") or {}).items():
            prior[chave_canonica(nome)] = max(0.0, min(0.99, float(p)))

    cultura = normalizar_cultura(fatos.get("cultura"))
    if cultura is None and prior:
        massa = {}
        for chave, p in prior.items():
            massa[cultura_de(chave)] = massa.get(cultura_de(chave), 0.0) + p
        melhor = max(massa, key=massa.get)
        if massa[melhor] >= 0.5:
            cultura = melhor
    return sintomas, cultura, prior


def _avaliar(regra, sintomas):
    """Devolve (grau, condicoes, desconhecidos, limitante)."""
    graus, conds, desconhecidos = [], [], []
    for nome in regra.se:
        v = sintomas.get(nome)
        if v is None:
            desconhecidos.append(nome)
        else:
            graus.append(v)
            conds.append((nome, v, False))
    for nome in regra.se_nao:
        v = sintomas.get(nome)
        if v is None:
            desconhecidos.append(nome)
        else:
            graus.append(1.0 - v)
            conds.append((nome, 1.0 - v, True))
    limitante = min(((n, g) for n, g, _ in conds), key=lambda x: x[1]) if conds else None
    grau = 0.0 if desconhecidos or not graus else min(graus)
    return grau, tuple(conds), tuple(desconhecidos), limitante


def diagnosticar(fatos, kb: BaseConhecimento, usar_cnn=True):
    cfg = kb.config
    sintomas, cultura, prior = _preparar(fatos, kb, usar_cnn)

    hipoteses = {
        d.id: Hipotese(d.id)
        for d in kb.doencas.values()
        if cultura is None or d.cultura == cultura
    }

    for regra in kb.regras:
        alvos = [h for h in regra.hipoteses if h in hipoteses]
        if not alvos:
            continue
        grau, conds, desconhecidos, limitante = _avaliar(regra, sintomas)
        if not desconhecidos and grau >= cfg["limiar_ativacao"]:
            ev = Evidencia("regra", regra.id, regra.cf * grau, grau, regra.porque, conds)
            for h in alvos:
                hipoteses[h].evidencias.append(ev)
        else:
            nd = RegraNaoDisparada(regra.id, grau, limitante, desconhecidos)
            conhecidas_ok = all(g >= cfg["limiar_ativacao"] for _, g, _ in conds)
            for h in alvos:
                hipoteses[h].nao_disparadas.append(nd)
                if desconhecidos and conhecidas_ok and regra.cf > 0:
                    hipoteses[h].pendentes.update(desconhecidos)

    for hid, p in prior.items():
        if hid in hipoteses and p > 0:
            hipoteses[hid].evidencias.append(Evidencia(
                "cnn", None, cfg["peso_cnn"] * p, p,
                "Probabilidade atribuída pela CNN a esta classe.",
            ))

    for h in hipoteses.values():
        regras_cf = [e.cf for e in h.evidencias if e.fonte == "regra"]
        cnn_cf = [e.cf for e in h.evidencias if e.fonte == "cnn"]
        h.cf_regras = combinar(regras_cf)
        h.cf_cnn = combinar(cnn_cf)
        h.cf_final = combinar(regras_cf + cnn_cf)

    ordenadas = sorted(hipoteses.values(), key=lambda h: (-h.cf_final, h.id))

    conclusivo = False
    if ordenadas:
        margem = ordenadas[0].cf_final - (ordenadas[1].cf_final if len(ordenadas) > 1 else 0.0)
        conclusivo = (ordenadas[0].cf_final >= cfg["limiar_conclusivo"]
                      and margem >= cfg["margem_minima"])

    conflito = None
    if prior and ordenadas:
        top_regras = max(ordenadas, key=lambda h: (h.cf_regras, h.id))
        top_cnn = max(ordenadas, key=lambda h: (h.cf_cnn, h.id))
        if (top_regras.id != top_cnn.id
                and top_regras.cf_regras >= cfg["limiar_ativacao"]
                and top_cnn.cf_cnn >= cfg["limiar_exibicao"]):
            conflito = (top_regras.id, top_cnn.id)

    return Resultado(cultura, ordenadas, conclusivo, conflito, sintomas, bool(prior))

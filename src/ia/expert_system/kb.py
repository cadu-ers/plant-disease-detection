"""Carrega e valida a base de conhecimento (knowledge_base.yaml).

Como o YAML será editado à mão, a validação é estrita: sintoma ou doença
inexistente numa regra, CF fora de [-1, 1] ou id duplicado geram ValueError
com uma mensagem apontando o problema.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .classes import CULTURAS

KB_PADRAO = Path(__file__).with_name("knowledge_base.yaml")

CONFIG_PADRAO = {
    "peso_cnn": 0.5,          # quanto a probabilidade da CNN vale como evidência (CF = peso * p)
    "limiar_ativacao": 0.3,   # grau mínimo de uma regra para disparar
    "limiar_conclusivo": 0.6, # CF mínimo do melhor diagnóstico para ser conclusivo
    "margem_minima": 0.15,    # diferença mínima para a 2ª colocada
    "limiar_exibicao": 0.2,   # abaixo disso a hipótese não é apresentada como provável
}

ORIGENS = ("imagem", "usuario")


@dataclass(frozen=True)
class Sintoma:
    id: str
    descricao: str
    origem: str
    pergunta: str
    medida: str = None
    trapezio: tuple = None


@dataclass(frozen=True)
class Doenca:
    id: str
    nome: str
    cultura: str
    agente: str
    urgencia: str
    confundivel_com: tuple
    recomendacoes: tuple


@dataclass(frozen=True)
class Regra:
    id: str
    se: tuple
    se_nao: tuple
    hipoteses: tuple
    cf: float
    porque: str


@dataclass
class BaseConhecimento:
    config: dict
    sintomas: dict
    doencas: dict
    regras: list = field(default_factory=list)

    def regras_de(self, hipotese):
        return [r for r in self.regras if hipotese in r.hipoteses]


def carregar_kb(caminho=None):
    caminho = Path(caminho) if caminho else KB_PADRAO
    with open(caminho, encoding="utf-8") as f:
        return kb_de_dict(yaml.safe_load(f))


def kb_de_dict(dados):
    config = {**CONFIG_PADRAO, **(dados.get("config") or {})}

    sintomas = {}
    for sid, s in (dados.get("sintomas") or {}).items():
        origem = s.get("origem")
        if origem not in ORIGENS:
            raise ValueError(f"Sintoma {sid}: origem deve ser uma de {ORIGENS}, veio {origem!r}")
        trap = s.get("trapezio")
        if trap is not None:
            trap = tuple(float(x) for x in trap)
            if len(trap) != 4 or list(trap) != sorted(trap):
                raise ValueError(f"Sintoma {sid}: trapezio precisa de 4 valores não decrescentes")
            if not s.get("medida"):
                raise ValueError(f"Sintoma {sid}: trapezio sem 'medida'")
        if not s.get("pergunta"):
            raise ValueError(f"Sintoma {sid}: falta 'pergunta' (usada no modo interativo)")
        sintomas[sid] = Sintoma(sid, s.get("descricao", ""), origem, s["pergunta"],
                                s.get("medida"), trap)

    doencas = {}
    for did, d in (dados.get("doencas") or {}).items():
        if d.get("cultura") not in CULTURAS:
            raise ValueError(f"Doença {did}: cultura deve ser uma de {CULTURAS}")
        doencas[did] = Doenca(
            id=did, nome=d["nome"], cultura=d["cultura"], agente=d.get("agente", ""),
            urgencia=d.get("urgencia", "media"),
            confundivel_com=tuple(d.get("confundivel_com") or ()),
            recomendacoes=tuple(d.get("recomendacoes") or ()),
        )
    for d in doencas.values():
        for outra in d.confundivel_com:
            if outra not in doencas:
                raise ValueError(f"Doença {d.id}: confundivel_com refere doença inexistente {outra!r}")

    regras, vistos = [], set()
    for r in dados.get("regras") or []:
        rid = r.get("id")
        if not rid or rid in vistos:
            raise ValueError(f"Regra com id ausente ou duplicado: {rid!r}")
        vistos.add(rid)
        se, se_nao = tuple(r.get("se") or ()), tuple(r.get("se_nao") or ())
        if not se and not se_nao:
            raise ValueError(f"Regra {rid}: sem condições (se / se_nao)")
        for s in se + se_nao:
            if s not in sintomas:
                raise ValueError(f"Regra {rid}: sintoma inexistente {s!r}")
        entao = r.get("entao") or {}
        hips = tuple(entao.get("hipoteses") or ())
        if not hips:
            raise ValueError(f"Regra {rid}: 'entao.hipoteses' vazio")
        for h in hips:
            if h not in doencas:
                raise ValueError(f"Regra {rid}: doença inexistente {h!r}")
        cf = float(entao.get("cf", 0))
        if cf == 0 or not -1 <= cf <= 1:
            raise ValueError(f"Regra {rid}: cf deve estar em [-1, 1] e ser diferente de 0")
        if not r.get("porque"):
            raise ValueError(f"Regra {rid}: falta o campo 'porque' (usado na explicação)")
        regras.append(Regra(rid, se, se_nao, hips, cf, r["porque"].strip()))

    return BaseConhecimento(config, sintomas, doencas, regras)

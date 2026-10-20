"""Mapeamento entre os nomes de classe do PlantVillage e as chaves canônicas
usadas pelo sistema especialista.

Os nomes das pastas do dataset são inconsistentes (``Tomato_Bacterial_spot``,
``Tomato__Target_Spot``, ``Tomato__Tomato_YellowLeaf__Curl_Virus``...). Toda a
base de conhecimento usa as chaves canônicas (``<cultura>_<doenca>``).
"""

import re

CULTURAS = ("tomate", "batata")

# nome da pasta no dataset -> chave canônica
MAPEAMENTO_DATASET = {
    "Potato___Early_blight": "batata_early_blight",
    "Potato___Late_blight": "batata_late_blight",
    "Potato___healthy": "batata_saudavel",
    "Tomato_Bacterial_spot": "tomate_bacterial_spot",
    "Tomato_Early_blight": "tomate_early_blight",
    "Tomato_Late_blight": "tomate_late_blight",
    "Tomato_Leaf_Mold": "tomate_leaf_mold",
    "Tomato_Septoria_leaf_spot": "tomate_septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "tomate_spider_mites",
    "Tomato__Target_Spot": "tomate_target_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "tomate_yellow_leaf_curl",
    "Tomato__Tomato_mosaic_virus": "tomate_mosaic",
    "Tomato_healthy": "tomate_saudavel",
}

CHAVES_CANONICAS = tuple(sorted(set(MAPEAMENTO_DATASET.values())))


def _normalizar(nome):
    return re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")


_NORMALIZADO = {_normalizar(k): v for k, v in MAPEAMENTO_DATASET.items()}


def chave_canonica(nome):
    """Aceita o nome da pasta do dataset (em qualquer caixa/separador) ou a
    própria chave canônica. Levanta KeyError se não reconhecer."""
    n = _normalizar(nome)
    if n in _NORMALIZADO:
        return _NORMALIZADO[n]
    if n in CHAVES_CANONICAS:
        return n
    raise KeyError(f"Classe desconhecida: {nome!r}")


def cultura_de(chave):
    """'tomate_early_blight' -> 'tomate'."""
    return chave.split("_", 1)[0]


def normalizar_cultura(valor):
    """Aceita 'tomate'/'tomato'/'batata'/'potato'; devolve None se vazio."""
    if valor is None:
        return None
    v = str(valor).strip().lower()
    v = {"tomato": "tomate", "potato": "batata"}.get(v, v)
    if v not in CULTURAS:
        raise ValueError(f"Cultura inválida: {valor!r} (use 'tomate' ou 'batata')")
    return v

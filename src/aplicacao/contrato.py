"""Contrato entre a imagem/CNN e o SISTEMA ESPECIALISTA.

Converte as saídas da CNN (vetor de probabilidades) e de medidas_lesao.py nos fatos
que ia.expert_system.diagnosticar espera:

    {"cultura": "tomate"|"batata", "medidas": {...}, "sintomas": {...}, "cnn_prior": {...}}

O contrato NÃO recebe caminho, nome de arquivo nem classe real: não há parâmetro para
isso. Quem chama (o pipeline) é responsável por nunca derivar a cultura da pasta.

Regras:
    cnn_prior     o vetor segue a ordem de models/class_names.json. É rejeitado se tiver
                  tamanho diferente de 13, valor não numérico, NaN/inf, negativo ou > 1,
                  ou soma a mais de TOLERANCIA_SOMA_CNN de 1 (provisória).
    medidas       vão para os fatos só quando a máscara é confiável. Máscara vazia,
                  suspeita ou sem avaliação: NENHUMA medida é enviada (desconhecido,
                  nunca 0). Só entram as medidas que a KB declara (frac_marrom,
                  frac_lesao_total, frac_amarelo, n_lesoes_pequenas e, havendo lesão
                  marrom, dist_borda_media_lesoes). frac_lesoes_com_halo nunca é enviada
                  (halo adiado).
    n_lesoes == 0 (máscara confiável) dist_borda_media_lesoes é omitida e NÃO se gera
                  o sintoma lesoes_na_borda: fica desconhecido. sem_manchas_marrons=True
                  e um aviso são devolvidos fora de `fatos` (o motor não os vê).
    sintomas      só os que o usuário informou (origem "usuario" na KB). O que ele não
                  informa fica ausente = desconhecido, nunca 0.
    cultura       só se for informada; senão o motor a infere do cnn_prior (cultura_origem
                  registra isso). Nunca vem de pasta, nome de arquivo ou classe real.

Atenção:
    - Uma medida com valor 0 (ex.: frac_marrom = 0) vira sintoma com grau 0: é uma
      MEDIDA, não um palpite. Mas lesões pequenas podem ter sumido na redução para
      224x224 ou ficar abaixo de MIN_LESION_AREA_PX; 0 não prova ausência de lesão.
    - A tolerância da soma das probabilidades é provisória.
"""

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.config import CLASS_NAMES_PATH
from ia.expert_system.classes import MAPEAMENTO_DATASET, normalizar_cultura
from ia.expert_system.kb import carregar_kb

TOLERANCIA_SOMA_CNN = 0.01
N_CLASSES = len(MAPEAMENTO_DATASET)
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]

AVISO_SEM_MANCHAS = ("nenhuma mancha marrom detectada (isso não prova ausência de lesão: "
                     "lesões minúsculas podem ter sumido em 224x224)")
AVISO_SEM_CNN = ("sem CNN: o diagnóstico usa só as medidas da imagem e as regras, com "
                 "menos evidência; o resultado pode ficar inconclusivo")


class ContratoInvalido(ValueError):
    """Os dados recebidos não cumprem o contrato."""


@dataclass(frozen=True)
class FatosEspecialista:
    fatos: dict                 # exatamente o que vai para diagnosticar()
    avisos: tuple               # textos para o usuário
    cultura_origem: str         # "informada" | "a_inferir_pela_cnn" | "desconhecida"
    sem_manchas_marrons: bool   # informativo; NÃO entra em `fatos`


def _numero(valor, nome):
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, numbers.Real):
        raise ContratoInvalido(f"{nome}: esperado um número, veio {valor!r}")
    valor = float(valor)
    if not math.isfinite(valor):
        raise ContratoInvalido(f"{nome}: valor não finito ({valor})")
    return valor


def carregar_class_names(caminho=CLASS_NAMES_PATH):
    """Lê a ordem das classes da CNN e confere que são as 13 do dataset."""
    caminho = Path(caminho)
    if not caminho.is_absolute():
        caminho = RAIZ_REPOSITORIO / caminho
    with open(caminho, encoding="utf-8") as f:
        nomes = json.load(f)
    return _validar_class_names(nomes)


def _validar_class_names(nomes):
    if (not isinstance(nomes, (list, tuple)) or len(nomes) != N_CLASSES
            or not all(isinstance(n, str) for n in nomes)):
        raise ContratoInvalido(f"class_names deve ser uma lista de {N_CLASSES} textos")
    if len(set(nomes)) != N_CLASSES:
        raise ContratoInvalido("class_names tem nomes repetidos")
    if set(nomes) != set(MAPEAMENTO_DATASET):
        raise ContratoInvalido("class_names não coincide com as 13 classes do dataset: "
                               f"{sorted(set(nomes) ^ set(MAPEAMENTO_DATASET))}")
    return tuple(nomes)


def montar_cnn_prior(probabilidades, class_names):
    """Vetor da CNN -> {nome_da_classe: p}, na ordem de class_names."""
    nomes = _validar_class_names(class_names)
    if isinstance(probabilidades, np.ndarray) and probabilidades.ndim != 1:
        raise ContratoInvalido(f"probabilidades deve ser 1-D, veio {probabilidades.ndim}-D")
    try:
        valores = list(probabilidades)
    except TypeError:
        raise ContratoInvalido("probabilidades deve ser uma sequência de números") from None
    if len(valores) != len(nomes):
        raise ContratoInvalido(
            f"esperadas {len(nomes)} probabilidades, vieram {len(valores)}")
    p = [_numero(v, f"probabilidade[{i}]") for i, v in enumerate(valores)]
    if min(p) < 0:
        raise ContratoInvalido("probabilidade negativa")
    if max(p) > 1 + 1e-6:
        raise ContratoInvalido("probabilidade maior que 1")
    soma = sum(p)
    if abs(soma - 1.0) > TOLERANCIA_SOMA_CNN:
        raise ContratoInvalido(
            f"as probabilidades somam {soma:.4f}, longe de 1 (tolerância {TOLERANCIA_SOMA_CNN})")
    return dict(zip(nomes, p))


def montar_medidas(medidas, qualidade):
    """MedidasLesao + QualidadeMascara -> (medidas_da_kb, sem_manchas_marrons, avisos)."""
    if qualidade is None:
        motivo = ("qualidade da máscara não avaliada" if medidas is not None
                  else "medidas de imagem não fornecidas")
        return {}, False, (f"medidas da imagem desconhecidas: {motivo}",)
    if not qualidade.confiavel:
        return {}, False, (f"medidas da imagem desconhecidas: {qualidade.motivo}",)
    if medidas is None:
        raise ContratoInvalido("máscara confiável, mas sem medidas")

    saida = {
        "frac_marrom": _numero(medidas.frac_marrom, "frac_marrom"),
        "frac_lesao_total": _numero(medidas.frac_lesao_total, "frac_lesao_total"),
        "frac_amarelo": _numero(medidas.frac_amarelo, "frac_amarelo"),
        "n_lesoes_pequenas": _numero(medidas.n_lesoes_pequenas, "n_lesoes_pequenas"),
    }
    if any(v < 0 for v in saida.values()):
        raise ContratoInvalido("medida negativa")
    dist = medidas.dist_borda_media_lesoes
    if medidas.n_lesoes == 0:
        if dist is not None:
            raise ContratoInvalido("n_lesoes == 0, mas dist_borda_media_lesoes foi informada")
        return saida, True, (AVISO_SEM_MANCHAS,)
    if dist is None:
        raise ContratoInvalido("n_lesoes > 0, mas dist_borda_media_lesoes é None")
    saida["dist_borda_media_lesoes"] = _numero(dist, "dist_borda_media_lesoes")
    return saida, False, ()


def _validar_sintomas_usuario(sintomas, kb):
    saida = {}
    for nome, valor in (sintomas or {}).items():
        sintoma = kb.sintomas.get(nome)
        if sintoma is None:
            raise ContratoInvalido(f"sintoma inexistente na KB: {nome!r}")
        if sintoma.origem != "usuario":
            raise ContratoInvalido(
                f"sintoma {nome!r} vem da imagem, não pode ser informado pelo usuário")
        valor = _numero(valor, f"sintoma {nome}")
        if not 0.0 <= valor <= 1.0:
            raise ContratoInvalido(f"sintoma {nome}: grau {valor} fora de [0, 1]")
        saida[nome] = valor
    return saida


def montar_fatos(*, probabilidades=None, class_names=None, medidas=None, qualidade=None,
                 cultura=None, sintomas_usuario=None, kb=None):
    """Monta os fatos do sistema especialista. Tudo é opcional; o que falta fica desconhecido."""
    avisos = []
    fatos = {}

    prior = None
    if probabilidades is not None:
        if class_names is None:
            class_names = carregar_class_names()
        prior = montar_cnn_prior(probabilidades, class_names)
        fatos["cnn_prior"] = prior
    else:
        avisos.append(AVISO_SEM_CNN)

    medidas_kb, sem_manchas, avisos_medidas = montar_medidas(medidas, qualidade)
    avisos.extend(avisos_medidas)
    if medidas_kb:
        fatos["medidas"] = medidas_kb

    if sintomas_usuario:
        sintomas = _validar_sintomas_usuario(sintomas_usuario, kb or carregar_kb())
        if sintomas:
            fatos["sintomas"] = sintomas

    if cultura is not None:
        try:
            fatos["cultura"] = normalizar_cultura(cultura)
        except ValueError as erro:
            raise ContratoInvalido(str(erro)) from None
        origem = "informada"
    elif prior is not None:
        origem = "a_inferir_pela_cnn"
    else:
        origem = "desconhecida"

    return FatosEspecialista(fatos, tuple(avisos), origem, sem_manchas)

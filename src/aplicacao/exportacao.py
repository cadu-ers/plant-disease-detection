"""Exportação das figuras de uma análise, chamada pela CLI DEPOIS do diagnóstico.

    resultado = salvar_figuras(original, segmentada, mascara, diagnostico)
    if resultado.ok: print(resultado.pasta) else: print(resultado.erro)

Grava, numa pasta NOVA por análise em rotina_demo/cli/ (ignorada pelo Git), até 4 PNGs:
original.png, mascara.png, segmentada.png e painel.png (os três lado a lado, montado com
numpy, sem texto sobre a imagem). A pasta se chama AAAAMMDD_HHMMSS_<8 hex>: sem classe real
nem nome do arquivo original. Nada é apagado nem sobrescrito (mkdir com exist_ok=False e
novas tentativas), e nunca se grava em data/, data_split/ nem data_processed/.

A função recebe SÓ arrays e o ResultadoDiagnostico (prova de que o diagnóstico terminou;
aceita status "erro"): não recebe a AmostraSorteada, o caminho nem a classe real. O
resultado da gravação é um valor SEPARADO do diagnóstico (que é imutável). Falha de disco
(OSError, imencode) vira ResultadoGravacao(ok=False, erro=...) e nunca levanta exceção, para
não impedir a comparação nem derrubar a CLI; só argumentos inválidos levantam TypeError ou
ValueError.

ATENÇÃO: a imagem salva é a segmentada 224x224 (SEG_IMG_SIZE), NÃO a entrada exata da CNN,
que ainda passa por codificação JPEG e redimensionamento para 160x160.

Os PNGs são gravados com cv2.imencode + write_bytes (sem cv2.imwrite).
"""

import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from aplicacao.pipeline import ResultadoDiagnostico
from common.config import CLI_DEMO_DIR, DATA_DIR, PROCESSED_DIR, SEG_IMG_SIZE, SPLIT_DIR

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]
PASTAS_PROTEGIDAS = (DATA_DIR, SPLIT_DIR, PROCESSED_DIR)
TENTATIVAS_PASTA = 10


@dataclass(frozen=True)
class ResultadoGravacao:
    ok: bool
    pasta: Optional[Path]          # None se a pasta não chegou a ser criada
    arquivos: tuple                # os arquivos realmente gravados
    erro: Optional[str]


def _altura_largura():
    largura, altura = SEG_IMG_SIZE
    return altura, largura


def _validar(original, segmentada, mascara, diagnostico):
    if not isinstance(diagnostico, ResultadoDiagnostico):
        raise TypeError("diagnostico deve ser um ResultadoDiagnostico (diagnóstico já concluído)")
    altura, largura = _altura_largura()
    for nome, imagem, forma in (("original", original, (altura, largura, 3)),
                                ("segmentada", segmentada, (altura, largura, 3)),
                                ("mascara", mascara, (altura, largura))):
        if not isinstance(imagem, np.ndarray) or imagem.dtype != np.uint8:
            raise ValueError(f"{nome} deve ser um array numpy uint8")
        if imagem.shape != forma:
            raise ValueError(f"{nome} deve ter formato {forma}, veio {imagem.shape}")


def _protegida(pasta):
    """True se `pasta` é (ou está dentro de) data/, data_split/ ou data_processed/."""
    alvo = os.path.normcase(os.path.realpath(pasta))
    for nome in PASTAS_PROTEGIDAS:
        protegida = os.path.normcase(os.path.realpath(RAIZ_REPOSITORIO / nome))
        try:
            if os.path.commonpath([alvo, protegida]) == protegida:
                return True
        except ValueError:         # discos diferentes no Windows
            continue
    return False


def _sufixo():
    return secrets.token_hex(4)


def _nova_pasta(base):
    """Cria uma pasta nova (nunca reaproveita uma existente). Levanta OSError se não conseguir."""
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(TENTATIVAS_PASTA):
        pasta = base / f"{carimbo}_{_sufixo()}"
        try:
            pasta.mkdir(parents=True, exist_ok=False)
            return pasta
        except FileExistsError:
            continue
    raise OSError(f"não foi possível criar uma pasta nova em {TENTATIVAS_PASTA} tentativas")


def _painel(original, mascara_255, segmentada):
    return np.hstack([original, cv2.cvtColor(mascara_255, cv2.COLOR_GRAY2BGR), segmentada])


def _gravar_png(caminho, imagem):
    ok, buffer = cv2.imencode(".png", imagem)
    if not ok:
        raise OSError(f"não foi possível codificar {caminho.name} como PNG")
    caminho.write_bytes(buffer.tobytes())


def salvar_figuras(original, segmentada, mascara, diagnostico, pasta_base=None):
    """Grava os PNGs da análise numa pasta nova. Nunca levanta OSError (ver o docstring do módulo)."""
    _validar(original, segmentada, mascara, diagnostico)
    base = Path(pasta_base) if pasta_base is not None else RAIZ_REPOSITORIO / CLI_DEMO_DIR
    if _protegida(base):
        raise ValueError("não é permitido gravar em data/, data_split/ nem data_processed/")

    mascara_255 = ((mascara > 0) * 255).astype(np.uint8)       # a máscara pode ser 0/1 ou 0/255
    figuras = (("original.png", original), ("mascara.png", mascara_255),
               ("segmentada.png", segmentada),
               ("painel.png", _painel(original, mascara_255, segmentada)))

    try:
        pasta = _nova_pasta(base)
    except OSError as erro:
        return ResultadoGravacao(False, None, (), f"não foi possível criar a pasta das figuras: {erro}")

    gravados = []
    for nome, imagem in figuras:
        try:
            _gravar_png(pasta / nome, imagem)
        except OSError as erro:
            salvos = ", ".join(gravados) if gravados else "nenhum"
            return ResultadoGravacao(
                False, pasta, tuple(pasta / n for n in gravados),
                f"falha ao gravar {nome} ({erro}); arquivos salvos antes da falha: {salvos}")
        gravados.append(nome)
    return ResultadoGravacao(True, pasta, tuple(pasta / n for n in gravados), None)

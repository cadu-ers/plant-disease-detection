"""Sorteio de uma imagem do split de TESTE para a demonstração.

Lê data_split/split_manifest.csv UMA vez (ao criar o SorteadorTeste) e sorteia só linhas
do split "test", usando SÓ a coluna original_path. O processed_path (imagem já processada)
nunca é usado nem guardado. Não varre pastas, não copia, não move e não baixa imagens.

    sorteador = SorteadorTeste()                 # lê o manifest uma vez por sessão
    amostra = sorteador.sortear(seed=7)          # tenta até abrir e segmentar uma imagem
    amostra.segmentada, amostra.mascara          # entram na inferência e nas medidas
    amostra.imagem.caminho / .classe_real        # SÓ para a seção de comparação final

O caminho e a classe real ficam em ImagemSorteada, fora do repr/str: imprimir o objeto
por engano mostra apenas "divisão: test". Isso impede vazamento ACIDENTAL; quem lê
.caminho ou .classe_real de propósito continua tendo acesso. Eles nunca devem ser
repassados ao contrato, à CNN ou ao sistema especialista.

Com a mesma seed, a primeira imagem sorteada é a mesma de common.utils.pick_test_image
(mesma ordenação e mesmo random.Random); as retentativas seguem a sequência do mesmo
gerador, sem repetir linhas já tentadas.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.config import SPLIT_MANIFEST
from processamento_imagens.preprocessing import segment_image

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]
COLUNAS_OBRIGATORIAS = ("class", "filename", "split", "original_path")
MAX_TENTATIVAS_PADRAO = 20


class SorteioInvalido(RuntimeError):
    """Não foi possível sortear uma imagem de teste (manifest inválido ou imagens ilegíveis)."""


@dataclass(frozen=True, eq=False)
class ImagemSorteada:
    """Imagem sorteada. Só a divisão aparece no repr/str; comparação por identidade.

    eq=False faz a comparação ser por identidade. compare=False nos campos sensíveis é
    necessário além disso: o pytest explica uma falha de "==" listando os campos do
    dataclass com compare=True, mesmo com eq=False, e imprimiria o caminho e a classe.
    """
    caminho: Path = field(repr=False, compare=False)
    classe_real: str = field(repr=False, compare=False)
    split: str = "test"

    def __repr__(self):
        return f"ImagemSorteada(divisão: {self.split})"

    __str__ = __repr__


@dataclass(frozen=True, eq=False)
class AmostraSorteada:
    """Imagem sorteada e já segmentada. Nada além da divisão aparece no repr/str."""
    imagem: ImagemSorteada = field(repr=False, compare=False)
    original: object = field(repr=False, compare=False)       # np.ndarray BGR 224x224 (antes da suavização)
    segmentada: object = field(repr=False, compare=False)     # np.ndarray BGR, entrada da CNN e das medidas
    mascara: object = field(repr=False, compare=False)        # np.ndarray uint8, 0 = fundo

    def __repr__(self):
        return f"AmostraSorteada(divisão: {self.imagem.split})"

    __str__ = __repr__


class SorteadorTeste:
    """Sorteia imagens do split de teste. O manifest é lido uma única vez, aqui."""

    def __init__(self, manifest_path=SPLIT_MANIFEST, raiz=RAIZ_REPOSITORIO):
        self._raiz = Path(raiz)
        manifest = self._resolver(manifest_path)
        if not manifest.is_file():
            raise SorteioInvalido(
                f"manifest não encontrado ({manifest.name}): rode antes python src/run_pi.py")
        try:
            tabela = pd.read_csv(manifest, usecols=list(COLUNAS_OBRIGATORIAS))
        except (ValueError, OSError, pd.errors.ParserError) as erro:
            raise SorteioInvalido(
                f"manifest inválido: precisa das colunas {COLUNAS_OBRIGATORIAS} ({erro})") from None
        teste = tabela[tabela["split"] == "test"]
        if teste.empty:
            raise SorteioInvalido("o manifest não tem imagens do split 'test'")
        if teste[["class", "original_path"]].isna().any().any():
            raise SorteioInvalido("o manifest tem linhas de teste sem classe ou sem original_path")
        teste = teste.sort_values(["class", "filename"])
        self._linhas = list(zip(teste["class"], teste["original_path"]))

    def _resolver(self, caminho):
        caminho = Path(caminho)
        return caminho if caminho.is_absolute() else self._raiz / caminho

    def __len__(self):
        return len(self._linhas)

    def sortear(self, seed=None, max_tentativas=MAX_TENTATIVAS_PADRAO):
        """Sorteia uma imagem de teste que abra e segmente; tenta outra se não abrir."""
        if max_tentativas < 1:
            raise ValueError("max_tentativas deve ser pelo menos 1")
        rng = random.Random(seed)
        tentadas = set()
        limite = min(max_tentativas, len(self._linhas))
        while len(tentadas) < limite:
            i = rng.randrange(len(self._linhas))
            if i in tentadas:
                continue
            tentadas.add(i)
            classe, original_path = self._linhas[i]
            caminho = self._resolver(original_path)
            resultado = segment_image(str(caminho)) if caminho.is_file() else None
            if resultado is not None:
                original, segmentada, mascara = resultado
                return AmostraSorteada(ImagemSorteada(caminho, classe), original, segmentada,
                                       mascara)
        raise SorteioInvalido(
            f"nenhuma imagem de teste pôde ser aberta em {len(tentadas)} tentativas; "
            "confira se a pasta data/ está completa")

"""Testes do sorteio de imagem do split de teste (aplicacao/sorteio.py).

Usam manifests e imagens sintéticas em tmp_path. O teste do manifest real só LÊ o CSV
(não abre imagens) e é pulado se ele não existir.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from aplicacao.sorteio import (
    AmostraSorteada, ImagemSorteada, SorteadorTeste, SorteioInvalido,
)
from common.config import SPLIT_MANIFEST
from common.utils import pick_test_image

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = Path(SRC_DIR).parent
CLASSES = ["Tomato_healthy", "Tomato_Early_blight"]


def folha(tom):
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.ellipse(img, (32, 32), (24, 20), 0, 0, 360, (40, tom, 60), -1)
    return img


def criar_projeto(raiz, por_split=None, ilegiveis=(), ausentes=()):
    """Cria data/<classe>/*.jpg e data_split/split_manifest.csv (caminhos relativos)."""
    por_split = por_split or {"train": 3, "val": 2, "test": 6}
    linhas = []
    for split, n in por_split.items():
        for classe in CLASSES:
            (raiz / "data" / classe).mkdir(parents=True, exist_ok=True)
            for k in range(n):
                nome = f"{split}_{k}.jpg"
                rel = f"data/{classe}/{nome}"
                if nome in ilegiveis:
                    (raiz / rel).write_bytes(b"isto nao e uma imagem")
                elif nome not in ausentes:
                    assert cv2.imwrite(str(raiz / rel), folha(120 + 10 * k))
                linhas.append(dict(**{"class": classe}, filename=nome, split=split,
                                   processed_path=f"data_processed/{classe}/{nome}",
                                   original_path=rel))
    (raiz / "data_split").mkdir(exist_ok=True)
    pd.DataFrame(linhas).to_csv(raiz / "data_split" / "split_manifest.csv", index=False)
    return raiz


@pytest.fixture
def projeto(tmp_path):
    return criar_projeto(tmp_path)


def sorteador(raiz):
    return SorteadorTeste("data_split/split_manifest.csv", raiz)


# ----------------------------------------------------------------------- sorteio
def test_so_sorteia_do_split_test(projeto):
    s = sorteador(projeto)
    assert len(s) == 12                                   # 6 por classe, só do teste
    vistos = set()
    for _ in range(200):
        a = s.sortear()
        vistos.add(a.imagem.caminho.name)
        assert a.imagem.caminho.name.startswith("test_") and a.imagem.split == "test"
    assert len(vistos) > 1


def test_seed_reproduzivel_e_igual_ao_pick_test_image(projeto):
    s = sorteador(projeto)
    manifest = pd.read_csv(projeto / "data_split" / "split_manifest.csv")
    for seed in range(5):
        a, b = s.sortear(seed=seed), s.sortear(seed=seed)
        assert a.imagem.caminho == b.imagem.caminho
        caminho, classe, split = pick_test_image(manifest, None, seed)
        assert a.imagem.caminho == projeto / caminho and a.imagem.classe_real == classe
    assert len({s.sortear(seed=k).imagem.caminho for k in range(30)}) > 1


def test_caminho_relativo_resolvido_de_qualquer_pasta(projeto, tmp_path_factory, monkeypatch):
    outra = tmp_path_factory.mktemp("outra_pasta")
    monkeypatch.chdir(outra)
    a = sorteador(projeto).sortear(seed=1)
    assert a.imagem.caminho.is_absolute() and a.imagem.caminho.is_file()
    assert projeto in a.imagem.caminho.parents
    assert a.original.shape == (224, 224, 3) and a.segmentada.shape == a.original.shape
    assert a.mascara.shape == (224, 224)


def test_nunca_usa_o_processed_path(projeto):
    # data_processed/ não existe: se o código a usasse, nada abriria.
    assert not (projeto / "data_processed").exists()
    a = sorteador(projeto).sortear(seed=3)
    assert "data_processed" not in str(a.imagem.caminho)


def test_imagem_ilegivel_ou_ausente_e_pulada(tmp_path):
    ruins = {f"test_{k}.jpg" for k in range(5)}                  # só test_5 abre, em cada classe
    criar_projeto(tmp_path, ilegiveis={"test_0.jpg", "test_1.jpg", "test_2.jpg"},
                  ausentes={"test_3.jpg", "test_4.jpg"})
    s = sorteador(tmp_path)
    for seed in range(15):
        a = s.sortear(seed=seed)
        assert a.imagem.caminho.name not in ruins


def test_todas_ilegiveis_esgotam_as_tentativas_com_mensagem_clara(tmp_path):
    pasta = tmp_path / "ISCA_PASTA"
    pasta.mkdir()
    criar_projeto(pasta, ilegiveis={f"test_{k}.jpg" for k in range(6)})
    with pytest.raises(SorteioInvalido, match="tentativas") as erro:
        sorteador(pasta).sortear(seed=0)
    assert "ISCA_PASTA" not in str(erro.value) and "test_" not in str(erro.value)


def test_limite_de_tentativas(tmp_path):
    criar_projeto(tmp_path, ilegiveis={f"test_{k}.jpg" for k in range(5)})   # 2 de 12 abrem
    s = sorteador(tmp_path)
    falhas = acertos = 0
    for seed in range(20):
        try:
            s.sortear(seed=seed, max_tentativas=1)
            acertos += 1
        except SorteioInvalido as erro:
            assert "1 tentativas" in str(erro)
            falhas += 1
    assert falhas > 0 and acertos > 0          # com 1 tentativa, depende da sorte; com 20 não
    assert all(s.sortear(seed=k).imagem.caminho.name == "test_5.jpg" for k in range(20))
    with pytest.raises(ValueError):
        s.sortear(max_tentativas=0)


# ---------------------------------------------------------------------- manifest
def test_manifest_ausente_vazio_ou_sem_test(tmp_path):
    with pytest.raises(SorteioInvalido, match="não encontrado"):
        sorteador(tmp_path)
    (tmp_path / "data_split").mkdir()
    pd.DataFrame(columns=["class", "filename", "split", "processed_path", "original_path"]
                 ).to_csv(tmp_path / "data_split" / "split_manifest.csv", index=False)
    with pytest.raises(SorteioInvalido, match="split 'test'"):
        sorteador(tmp_path)
    criar_projeto(tmp_path, por_split={"train": 2, "val": 2})
    with pytest.raises(SorteioInvalido, match="split 'test'"):
        sorteador(tmp_path)


def test_manifest_sem_coluna_obrigatoria_ou_com_nulos(tmp_path):
    criar_projeto(tmp_path)
    caminho = tmp_path / "data_split" / "split_manifest.csv"
    tabela = pd.read_csv(caminho)
    tabela.drop(columns=["original_path"]).to_csv(caminho, index=False)
    with pytest.raises(SorteioInvalido, match="colunas"):
        sorteador(tmp_path)
    tabela.loc[tabela["split"] == "test", "original_path"] = np.nan
    tabela.to_csv(caminho, index=False)
    with pytest.raises(SorteioInvalido, match="sem classe ou sem original_path"):
        sorteador(tmp_path)


def test_manifest_e_lido_uma_unica_vez(projeto, monkeypatch):
    chamadas = []
    original = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: chamadas.append(1) or original(*a, **k))
    s = sorteador(projeto)
    for seed in range(10):
        s.sortear(seed=seed)
    assert len(chamadas) == 1


# ---------------------------------------------------------------------- vazamento
def imagem_isca(n=1):
    return ImagemSorteada(Path(f"ISCA_CAMINHO_{n}/arquivo_isca.jpg"), f"ISCA_CLASSE_{n}")


def sem_isca(texto):
    return "ISCA" not in texto and "arquivo_isca" not in texto


def test_repr_str_e_fstring_so_mostram_a_divisao():
    img = imagem_isca()
    amostra = AmostraSorteada(img, np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.zeros((2, 2)))
    for obj in (img, amostra):
        for texto in (repr(obj), str(obj), f"{obj}", f"{obj!r}", repr([obj]), repr({"x": obj}),
                      repr((obj, 1)), "%s" % obj, format(obj)):
            assert sem_isca(texto) and "divisão: test" in texto
    assert "caminho" not in repr(img) and "classe_real" not in repr(img)
    assert "[[" not in repr(amostra) and "array" not in repr(amostra)


def test_campos_sensiveis_estao_fora_do_repr():
    for classe in (ImagemSorteada, AmostraSorteada):
        assert classe.__dataclass_params__.frozen and not classe.__dataclass_params__.eq
    campos = {f.name: f for f in dataclasses.fields(ImagemSorteada)}
    assert not campos["caminho"].repr and not campos["classe_real"].repr
    assert not campos["caminho"].compare and not campos["classe_real"].compare
    assert all(not f.repr and not f.compare for f in dataclasses.fields(AmostraSorteada))
    assert imagem_isca().caminho.name == "arquivo_isca.jpg"        # acesso explícito continua


def test_duas_imagens_diferentes_nao_sao_iguais():
    a, b = imagem_isca(1), imagem_isca(2)
    assert a != b and a == a
    copia = ImagemSorteada(a.caminho, a.classe_real)
    assert copia != a                                              # igualdade é por identidade
    assert len({a, b, copia}) == 3                                 # e o hash também


def test_falha_de_comparacao_nao_imprime_caminho_nem_classe():
    a, b = imagem_isca(1), imagem_isca(2)
    with pytest.raises(AssertionError) as erro:
        assert a == b
    assert sem_isca(str(erro.value))
    with pytest.raises(AssertionError) as erro:
        assert [a] == [b]
    assert sem_isca(str(erro.value))


def test_mensagens_de_erro_nao_trazem_caminho(tmp_path):
    pasta = tmp_path / "ISCA_PASTA"
    pasta.mkdir()
    with pytest.raises(SorteioInvalido) as erro:
        sorteador(pasta)
    assert "ISCA_PASTA" not in str(erro.value)


# ---------------------------------------------------------- manifest real e imports
@pytest.mark.skipif(not os.path.exists(RAIZ / SPLIT_MANIFEST),
                    reason="data_split/split_manifest.csv não existe")
def test_manifest_real_so_leitura():
    s = SorteadorTeste()                                           # resolve pela raiz do repositório
    assert len(s) == 2725
    manifest = pd.read_csv(RAIZ / SPLIT_MANIFEST)
    teste = manifest[manifest["split"] == "test"]
    assert teste["class"].nunique() == 13
    assert not teste["original_path"].map(os.path.isabs).any()
    assert teste["original_path"].str.startswith("data/").all()
    assert not teste["original_path"].str.contains("data_processed").any()
    for seed in range(5):                                          # só a escolha da linha, sem abrir imagem
        classe, original_path = s._linhas[np.random.default_rng(seed).integers(len(s))]
        assert original_path.startswith("data/") and (RAIZ / original_path).is_file()


def test_importar_o_modulo_nao_carrega_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import aplicacao.sorteio; "
              "assert 'tensorflow' not in sys.modules, 'importou tensorflow'")
    resultado = subprocess.run([sys.executable, "-c", codigo, SRC_DIR],
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr

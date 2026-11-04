"""Testes da exportação das figuras (aplicacao/exportacao.py).

Tudo é gravado em tmp_path (pasta_base injetada): nenhum teste escreve em rotina_demo/.
"""

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from aplicacao import exportacao
from aplicacao.exportacao import ResultadoGravacao, salvar_figuras
from aplicacao.pipeline import ResultadoDiagnostico
from common.config import CLI_DEMO_DIR, SEG_IMG_SIZE

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = Path(SRC_DIR).parent
LARGURA, ALTURA = SEG_IMG_SIZE
FORMATO_PASTA = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")


def diag(status="ok", **extra):
    base = dict(status=status, mensagem_erro=None if status == "ok" else "erro de teste",
                divisao="test", modo="sem_cnn", top3_cnn=(), qualidade=None, medidas=None,
                avisos=(), regras_disparadas=(), pendencias=(), conclusao=None, explicacao="",
                resultado=None)
    base.update(extra)
    return ResultadoDiagnostico(**base)


def arrays():
    rng = np.random.default_rng(0)
    original = rng.integers(0, 256, (ALTURA, LARGURA, 3), dtype=np.uint8)
    segmentada = rng.integers(0, 256, (ALTURA, LARGURA, 3), dtype=np.uint8)
    mascara = np.zeros((ALTURA, LARGURA), np.uint8)
    mascara[50:150, 40:180] = 255
    return original, segmentada, mascara


def salvar(base, **kw):
    original, segmentada, mascara = arrays()
    return salvar_figuras(original, segmentada, mascara, diag(), pasta_base=base, **kw)


def ler(caminho):
    return cv2.imdecode(np.frombuffer(Path(caminho).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)


# ------------------------------------------------------------------------ gravação
def test_grava_os_4_pngs_com_os_tamanhos_certos(tmp_path):
    original, segmentada, mascara = arrays()
    r = salvar_figuras(original, segmentada, mascara, diag(), pasta_base=tmp_path)
    assert isinstance(r, ResultadoGravacao) and r.ok and r.erro is None
    assert [a.name for a in r.arquivos] == ["original.png", "mascara.png", "segmentada.png",
                                            "painel.png"]
    assert sorted(p.name for p in r.pasta.iterdir()) == sorted(a.name for a in r.arquivos)
    assert np.array_equal(ler(r.arquivos[0]), original)
    assert np.array_equal(ler(r.arquivos[2]), segmentada)
    assert ler(r.arquivos[1]).shape == (ALTURA, LARGURA)
    painel = ler(r.arquivos[3])
    assert painel.shape == (ALTURA, 3 * LARGURA, 3)
    assert np.array_equal(painel[:, :LARGURA], original)
    assert np.array_equal(painel[:, 2 * LARGURA:], segmentada)
    assert np.array_equal(painel[:, LARGURA:2 * LARGURA, 0], mascara)    # mesma máscara, em cinza


def test_mascara_0_1_e_normalizada_para_0_255(tmp_path):
    original, segmentada, mascara = arrays()
    r = salvar_figuras(original, segmentada, (mascara > 0).astype(np.uint8), diag(),
                       pasta_base=tmp_path)
    salva = ler(r.arquivos[1])
    assert set(np.unique(salva)) == {0, 255} and np.array_equal(salva, mascara)
    painel = ler(r.arquivos[3])
    assert set(np.unique(painel[:, LARGURA:2 * LARGURA])) == {0, 255}


def test_aceita_diagnostico_com_status_erro(tmp_path):
    original, segmentada, mascara = arrays()
    assert salvar_figuras(original, segmentada, mascara, diag("erro"), pasta_base=tmp_path).ok


def test_cada_analise_ganha_uma_pasta_nova_e_nada_e_apagado(tmp_path):
    (tmp_path / "antiga").mkdir()
    (tmp_path / "antiga" / "arquivo.txt").write_text("conteúdo", encoding="utf-8")
    r1, r2 = salvar(tmp_path), salvar(tmp_path)
    assert r1.pasta != r2.pasta and FORMATO_PASTA.match(r1.pasta.name)
    assert (tmp_path / "antiga" / "arquivo.txt").read_text(encoding="utf-8") == "conteúdo"
    assert len(list(tmp_path.iterdir())) == 3


def test_colisao_de_nome_tenta_outro_sufixo(tmp_path, monkeypatch):
    sufixos = iter(["aaaaaaaa", "aaaaaaaa", "bbbbbbbb"])
    monkeypatch.setattr(exportacao, "_sufixo", lambda: next(sufixos))
    monkeypatch.setattr(exportacao, "datetime", type("D", (), {
        "now": staticmethod(lambda: __import__("datetime").datetime(2026, 1, 2, 3, 4, 5))}))
    primeira = salvar(tmp_path)
    segunda = salvar(tmp_path)                               # 1ª tentativa colide, 2ª usa "bbbbbbbb"
    assert primeira.pasta.name == "20260102_030405_aaaaaaaa"
    assert segunda.ok and segunda.pasta.name == "20260102_030405_bbbbbbbb"


def test_colisao_sempre_esgota_as_tentativas_sem_excecao(tmp_path, monkeypatch):
    monkeypatch.setattr(exportacao, "_sufixo", lambda: "cccccccc")
    monkeypatch.setattr(exportacao, "datetime", type("D", (), {
        "now": staticmethod(lambda: __import__("datetime").datetime(2026, 1, 2, 3, 4, 5))}))
    assert salvar(tmp_path).ok
    r = salvar(tmp_path)
    assert not r.ok and r.pasta is None and r.arquivos == () and "tentativas" in r.erro


# ---------------------------------------------------------------------------- falhas
def test_imencode_falso_nao_levanta_e_diz_o_que_falhou(tmp_path, monkeypatch):
    monkeypatch.setattr(exportacao.cv2, "imencode", lambda *a, **k: (False, None))
    r = salvar(tmp_path)
    assert not r.ok and r.arquivos == () and "codificar original.png" in r.erro
    assert "nenhum" in r.erro                                # nenhum arquivo foi salvo antes


def test_gravacao_parcial_lista_os_arquivos_salvos_antes_da_falha(tmp_path, monkeypatch):
    real = Path.write_bytes

    def falha_na_terceira(self, dados):
        if self.name == "segmentada.png":
            raise OSError("disco cheio")
        return real(self, dados)

    monkeypatch.setattr(Path, "write_bytes", falha_na_terceira)
    r = salvar(tmp_path)
    assert not r.ok and r.pasta is not None
    assert [a.name for a in r.arquivos] == ["original.png", "mascara.png"]
    assert "segmentada.png" in r.erro and "disco cheio" in r.erro
    assert "original.png, mascara.png" in r.erro             # diz o que foi salvo antes
    assert all(a.exists() for a in r.arquivos)               # nada foi apagado


def test_erro_ao_criar_a_pasta_nao_levanta(tmp_path, monkeypatch):
    def negado(self, *a, **k):
        raise PermissionError("acesso negado")

    monkeypatch.setattr(Path, "mkdir", negado)
    r = salvar(tmp_path)
    assert not r.ok and r.pasta is None and "acesso negado" in r.erro


# --------------------------------------------------------------------------- validação
@pytest.mark.parametrize("campo, valor", [
    ("original", np.zeros((10, 10, 3), np.uint8)),
    ("original", np.zeros((ALTURA, LARGURA, 3), np.float32)),
    ("original", None),
    ("segmentada", np.zeros((ALTURA, LARGURA), np.uint8)),
    ("mascara", np.zeros((ALTURA, LARGURA, 3), np.uint8)),
    ("mascara", np.zeros((ALTURA, LARGURA), bool)),
    ("mascara", "mascara.png"),
])
def test_arrays_invalidos_sao_rejeitados(tmp_path, campo, valor):
    args = dict(zip(("original", "segmentada", "mascara"), arrays()))
    args[campo] = valor
    with pytest.raises(ValueError):
        salvar_figuras(**args, diagnostico=diag(), pasta_base=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("ruim", [None, "ok", {"status": "ok"}])
def test_diagnostico_de_tipo_errado_e_rejeitado(tmp_path, ruim):
    with pytest.raises(TypeError):
        salvar_figuras(*arrays(), ruim, pasta_base=tmp_path)


# ------------------------------------------------------------------ pastas protegidas
@pytest.mark.parametrize("nome", ["data", "DATA", "Data_Split", "DATA_PROCESSED",
                                  "data_split/test", "data/Tomato_healthy/DATA"])
def test_recusa_gravar_em_pastas_protegidas_em_qualquer_caixa(nome):
    alvo = RAIZ / nome
    existia = alvo.exists()
    with pytest.raises(ValueError, match="não é permitido"):
        salvar_figuras(*arrays(), diag(), pasta_base=alvo)
    assert alvo.exists() == existia                          # nada foi criado


def test_pastas_parecidas_nao_sao_bloqueadas(tmp_path):
    assert not exportacao._protegida(tmp_path / "data_cli")
    assert not exportacao._protegida(RAIZ / "rotina_demo" / "cli")
    assert exportacao._protegida(RAIZ / "data")


# -------------------------------------------------------------------------- vazamento
def test_assinatura_so_tem_arrays_diagnostico_e_pasta_base():
    assert list(inspect.signature(salvar_figuras).parameters) == [
        "original", "segmentada", "mascara", "diagnostico", "pasta_base"]


def test_nomes_nao_trazem_classe_real_nem_nome_de_arquivo(tmp_path):
    r = salvar(tmp_path)
    nomes = [r.pasta.name, *(a.name for a in r.arquivos)]
    assert FORMATO_PASTA.match(nomes[0])
    assert all(re.fullmatch(r"[a-z]+\.png", n) for n in nomes[1:])
    for texto in nomes + [repr(r)]:
        assert "ISCA" not in texto and "Tomato" not in texto and "Potato" not in texto


# ------------------------------------------------------------------- caminho padrão
def test_caminho_padrao_parte_da_raiz_do_repositorio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gravadas = []
    monkeypatch.setattr(exportacao, "_nova_pasta", lambda base: gravadas.append(base) or tmp_path / "x")
    monkeypatch.setattr(exportacao, "_gravar_png", lambda *a: None)
    salvar_figuras(*arrays(), diag())
    assert gravadas == [RAIZ / CLI_DEMO_DIR] and gravadas[0].is_absolute()


def test_os_testes_nao_gravam_em_rotina_demo_cli():
    assert not (RAIZ / CLI_DEMO_DIR).exists() or not any((RAIZ / CLI_DEMO_DIR).iterdir())


def test_importar_o_modulo_nao_carrega_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import aplicacao.exportacao; "
              "assert 'tensorflow' not in sys.modules, 'importou tensorflow'")
    resultado = subprocess.run([sys.executable, "-c", codigo, SRC_DIR],
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr

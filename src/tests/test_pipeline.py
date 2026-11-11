"""Testes do pipeline (aplicacao/pipeline.py).

Usam folhas sintéticas, preditores falsos e a KB real. Não usam TensorFlow nem o modelo.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from aplicacao import pipeline
from aplicacao.contrato import carregar_class_names
from aplicacao.pipeline import (
    Comparacao, ResultadoDiagnostico, comparar_com_classe_real, diagnosticar_amostra, obter_kb,
)
from aplicacao.sorteio import AmostraSorteada, ImagemSorteada
from ia.expert_system import carregar_kb
from ia.inferencia import NOTA_CNN_DEMONSTRACAO, ModeloIndisponivel
from processamento_imagens.preprocessing import segment_image

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = list(carregar_class_names())
ISCAS = ("ISCA_CAMINHO", "ISCA_CLASSE", "arquivo_isca")


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


@pytest.fixture(scope="module")
def folha(tmp_path_factory):
    """(segmentada, mascara) de uma folha verde com manchas marrons, pela segmentação real."""
    img = np.full((224, 224, 3), (150, 150, 150), np.uint8)
    cv2.ellipse(img, (112, 112), (85, 70), 0, 0, 360, (40, 160, 60), -1)
    for x, y in ((90, 100), (130, 120), (110, 80)):
        cv2.circle(img, (x, y), 9, (30, 70, 120), -1)
    arquivo = str(tmp_path_factory.mktemp("folha") / "folha.png")
    assert cv2.imwrite(arquivo, img)
    _, segmentada, mascara = segment_image(arquivo)
    return segmentada, mascara


class PreditorFalso:
    """Chamável como o PreditorCNN: guarda o que recebeu."""

    def __init__(self, vetor, class_names=CLASSES):
        self.vetor, self.chamadas = vetor, []
        if class_names is not None:
            self.class_names = tuple(class_names)

    def __call__(self, segmentada_bgr, extensao=".jpg"):
        self.chamadas.append((segmentada_bgr, extensao))
        return self.vetor


def vetor(nomes=CLASSES, **pesos):
    p = np.zeros(len(nomes), np.float32)
    for nome, valor in pesos.items():
        p[nomes.index(nome)] = valor
    return p


def requeima():
    return vetor(Tomato_Late_blight=0.7, Tomato_Early_blight=0.2, Tomato__Target_Spot=0.1)


# --------------------------------------------------------------------- modo sem CNN
def test_sem_preditor_roda_so_com_medidas_e_regras(folha, kb):
    r = diagnosticar_amostra(*folha, "test", kb=kb)
    assert r.status == "ok" and r.modo == "sem_cnn" and r.top3_cnn == ()
    assert not r.resultado.usou_cnn and r.medidas is not None
    assert any("sem CNN" in a for a in r.avisos)
    assert NOTA_CNN_DEMONSTRACAO not in r.avisos                   # a nota nunca é um aviso


def test_inconclusivo_nunca_vira_saudavel(kb):
    folha_limpa = np.zeros((224, 224, 3), np.uint8)
    cv2.ellipse(folha_limpa, (112, 112), (85, 70), 0, 0, 360, (40, 160, 60), -1)
    mascara = np.zeros((224, 224), np.uint8)
    cv2.ellipse(mascara, (112, 112), (85, 70), 0, 0, 360, 255, -1)
    r = diagnosticar_amostra(folha_limpa, mascara, "test", kb=kb)  # sem CNN, sem sintomas
    assert r.conclusao.hipotese_id is None and not r.conclusao.conclusivo
    assert "INCONCLUSIVO" in r.conclusao.texto and "saudável" not in r.conclusao.texto.lower()


# ------------------------------------------------------------------------- máscaras
def test_mascara_suspeita_nao_tem_medidas_mas_segue_com_cnn(kb):
    mascara = np.zeros((224, 224), np.uint8)
    mascara[:, :140] = 255                                         # colada nas bordas
    r = diagnosticar_amostra(np.zeros((224, 224, 3), np.uint8), mascara, "test",
                             preditor=PreditorFalso(requeima()), kb=kb)
    assert r.status == "ok" and r.medidas is None and not r.qualidade.confiavel
    assert any("fundo vazado" in a for a in r.avisos)
    assert r.modo == "com_cnn" and r.conclusao.hipotese_id == "tomate_late_blight"


def test_mascara_vazia_nao_tem_medidas(kb):
    r = diagnosticar_amostra(np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224), np.uint8),
                             "test", kb=kb)
    assert r.status == "ok" and r.medidas is None and r.qualidade.vazia
    assert any("máscara vazia" in a for a in r.avisos)


def test_tamanhos_diferentes_viram_erro_sem_excecao(kb):
    mascara = np.zeros((224, 224), np.uint8)
    mascara[40:184, 40:184] = 255
    r = diagnosticar_amostra(np.zeros((100, 100, 3), np.uint8), mascara, "test", kb=kb)
    assert r.status == "erro" and "tamanhos diferentes" in r.mensagem_erro


# ---------------------------------------------------------------------------- CNN
class Quebrado:
    class_names = tuple(CLASSES)

    def __init__(self, erro):
        self.erro = erro

    def __call__(self, segmentada_bgr, extensao=".jpg"):
        raise self.erro


@pytest.mark.parametrize("preditor, trecho", [
    (PreditorFalso(np.full(12, 1 / 12, np.float32)), "saída inválida"),
    (PreditorFalso(np.array([np.nan] + [1 / 12] * 12, np.float32)), "saída inválida"),
    (PreditorFalso(np.full(13, 0.5 / 13, np.float32)), "saída inválida"),
    (PreditorFalso(None), "CNN ignorada"),
    (Quebrado(RuntimeError("falha do TensorFlow")), "falhou ao executar"),
    (Quebrado(ModeloIndisponivel("modelo da CNN não encontrado")), "CNN indisponível"),
    (PreditorFalso(requeima(), class_names=["a"] * 13), "saída inválida"),
])
def test_cnn_com_falha_cai_para_o_modo_sem_cnn(folha, kb, preditor, trecho):
    r = diagnosticar_amostra(*folha, "test", preditor=preditor, kb=kb)
    assert r.status == "ok" and r.modo == "sem_cnn" and r.top3_cnn == ()
    assert any(trecho in a for a in r.avisos) and any("sem CNN" in a for a in r.avisos)
    assert not r.resultado.usou_cnn and NOTA_CNN_DEMONSTRACAO not in r.avisos


def test_cnn_usada_traz_top3_sem_aviso_falso_dos_pesos(folha, kb):
    r = diagnosticar_amostra(*folha, "test", preditor=PreditorFalso(requeima()), kb=kb)
    assert r.modo == "com_cnn" and NOTA_CNN_DEMONSTRACAO not in r.avisos
    assert not any("pesos" in a for a in r.avisos)               # a nota só vai à revelação
    assert [t[0] for t in r.top3_cnn] == ["Tomato_Late_blight", "Tomato_Early_blight",
                                          "Tomato__Target_Spot"]
    assert r.top3_cnn[0][1] == kb.doencas["tomate_late_blight"].nome
    assert r.top3_cnn[0][2] == pytest.approx(0.7)
    assert any("cultura inferida pela CNN: tomate" in a for a in r.avisos)


def test_ordem_das_classes_trocada(folha, kb):
    ordem = np.random.default_rng(1).permutation(13)
    nomes2 = [CLASSES[i] for i in ordem]
    juntos = diagnosticar_amostra(*folha, "test", kb=kb,
                                  preditor=PreditorFalso(requeima()[ordem], nomes2))
    normal = diagnosticar_amostra(*folha, "test", kb=kb, preditor=PreditorFalso(requeima()))
    assert juntos.top3_cnn == normal.top3_cnn
    assert juntos.conclusao == normal.conclusao
    # vetor embaralhado, mas com a ordem original declarada: a CNN "diz" outra coisa
    errado = diagnosticar_amostra(*folha, "test", kb=kb,
                                  preditor=PreditorFalso(requeima()[ordem], CLASSES))
    assert errado.top3_cnn[0][0] != "Tomato_Late_blight"


def test_class_names_vem_do_parametro_quando_o_preditor_nao_tem(folha, kb):
    sem_atributo = PreditorFalso(requeima(), class_names=None)
    assert not hasattr(sem_atributo, "class_names")
    r = diagnosticar_amostra(*folha, "test", preditor=sem_atributo, kb=kb, class_names=CLASSES)
    assert r.modo == "com_cnn" and r.top3_cnn[0][0] == "Tomato_Late_blight"
    r2 = diagnosticar_amostra(*folha, "test", preditor=sem_atributo, kb=kb)   # lê o JSON real
    assert r2.top3_cnn == r.top3_cnn


# ----------------------------------------------------------------- erro do contrato
def test_sintoma_do_usuario_invalido_vira_erro_sem_excecao(folha, kb):
    r = diagnosticar_amostra(*folha, "test", kb=kb, sintomas_usuario={"inventado": 1.0})
    assert r.status == "erro" and "inventado" in r.mensagem_erro
    assert r.conclusao is None and r.resultado is None
    with pytest.raises(ValueError):
        comparar_com_classe_real(r, ImagemSorteada(Path("x.jpg"), "Tomato_healthy"))


# ------------------------------------------------------------------------ fluxo completo
def test_fluxo_completo_com_preditor_falso_e_kb_real(folha, kb):
    r = diagnosticar_amostra(
        *folha, "test", preditor=PreditorFalso(requeima()), kb=kb,
        sintomas_usuario={"aspecto_encharcado": 1.0, "umidade_alta": 1.0})
    assert isinstance(r, ResultadoDiagnostico) and r.status == "ok" and r.divisao == "test"
    assert len(r.top3_cnn) == 3 and r.medidas is not None and r.qualidade.confiavel
    assert r.regras_disparadas and all(x.porque and x.hipoteses for x in r.regras_disparadas)
    ids = [x.regra_id for x in r.regras_disparadas]
    assert len(ids) == len(set(ids))                               # sem regra duplicada
    assert r.conclusao.nome and r.conclusao.cf is not None and "calibrada" in r.conclusao.texto
    assert "Diagnóstico mais provável" in r.explicacao
    assert isinstance(r.pendencias, tuple)


def test_pendencias_listam_o_que_ajudaria(folha, kb):
    r = diagnosticar_amostra(*folha, "test", preditor=PreditorFalso(requeima()), kb=kb)
    assert r.pendencias and r.pendencias[0].startswith("Pergunta sugerida:")


# --------------------------------------------------------------------- KB e pureza
def test_a_kb_e_carregada_uma_unica_vez(folha, monkeypatch):
    chamadas = []
    real = carregar_kb
    monkeypatch.setattr(pipeline, "_KB_PADRAO", None)
    monkeypatch.setattr(pipeline, "carregar_kb", lambda *a: chamadas.append(1) or real(*a))
    diagnosticar_amostra(*folha, "test")
    diagnosticar_amostra(*folha, "test")
    assert len(chamadas) == 1 and obter_kb() is obter_kb()


def test_kb_injetada_nao_e_recarregada(folha, kb, monkeypatch):
    monkeypatch.setattr(pipeline, "carregar_kb", lambda *a: pytest.fail("não deveria carregar"))
    assert diagnosticar_amostra(*folha, "test", kb=kb).status == "ok"


def test_diagnosticar_nao_cria_arquivos(folha, kb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def arquivos(raiz):
        return sorted(str(p.relative_to(raiz)) for p in Path(raiz).rglob("*"))

    antes, repo_antes = arquivos(tmp_path), (Path(SRC_DIR).parent / "rotina_demo").exists()
    diagnosticar_amostra(*folha, "test", preditor=PreditorFalso(requeima()), kb=kb)
    diagnosticar_amostra(*folha, "test", kb=kb)
    assert arquivos(tmp_path) == antes == []
    assert (Path(SRC_DIR).parent / "rotina_demo").exists() == repo_antes


# --------------------------------------------------------------------- vazamento
class ImagemArmadilha:
    """Falha se alguém ler o caminho ou a classe real; só a divisão é permitida."""
    split = "test"

    @property
    def caminho(self):
        raise AssertionError("o pipeline leu o caminho")

    @property
    def classe_real(self):
        raise AssertionError("o pipeline leu a classe real")


def varrer(obj, achados, vistos=None):
    """Textos de todos os atributos/itens, recursivamente, para procurar a isca."""
    vistos = vistos if vistos is not None else set()
    if id(obj) in vistos:
        return achados
    vistos.add(id(obj))
    if isinstance(obj, (str, Path)):
        achados.append(str(obj))
    elif isinstance(obj, np.ndarray):
        pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            varrer(k, achados, vistos), varrer(v, achados, vistos)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            varrer(v, achados, vistos)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            varrer(v, achados, vistos)
    elif hasattr(obj, "__dataclass_fields__"):
        for nome in obj.__dataclass_fields__:
            varrer(getattr(obj, nome), achados, vistos)
    return achados


def test_assinatura_so_aceita_arrays_divisao_e_dependencias():
    assert list(inspect.signature(diagnosticar_amostra).parameters) == [
        "segmentada", "mascara", "divisao", "preditor", "kb", "class_names", "sintomas_usuario"]


def test_armadilha_o_pipeline_nunca_le_caminho_nem_classe_real(folha, kb):
    amostra = AmostraSorteada(ImagemArmadilha(), None, *folha)
    r = diagnosticar_amostra(amostra.segmentada, amostra.mascara, amostra.imagem.split,
                             preditor=PreditorFalso(requeima()), kb=kb)
    assert r.status == "ok" and r.divisao == "test"


def test_espioes_preditor_e_contrato_so_recebem_arrays(folha, kb, monkeypatch):
    imagem = ImagemSorteada(Path("ISCA_CAMINHO/arquivo_isca.jpg"), "ISCA_CLASSE")
    amostra = AmostraSorteada(imagem, None, *folha)
    capturado = {}
    real = pipeline.montar_fatos

    def espiao(**kwargs):
        capturado.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(pipeline, "montar_fatos", espiao)
    preditor = PreditorFalso(requeima())
    r = diagnosticar_amostra(amostra.segmentada, amostra.mascara, amostra.imagem.split,
                             preditor=preditor, kb=kb)
    assert len(preditor.chamadas) == 1
    argumentos = preditor.chamadas[0]
    assert isinstance(argumentos[0], np.ndarray) and argumentos[1] == ".jpg"
    assert set(capturado) == {"probabilidades", "class_names", "medidas", "qualidade",
                              "sintomas_usuario", "kb"}
    assert "cultura" not in capturado                               # opção A: o motor infere
    textos = varrer(capturado, []) + varrer(r, []) + [repr(r), r.explicacao, *r.avisos]
    for isca in ISCAS:
        assert not any(isca in t for t in textos), isca


# ------------------------------------------------------------------------ comparação
def test_comparacao_so_depois_do_diagnostico(folha, kb):
    r = diagnosticar_amostra(*folha, "test", preditor=PreditorFalso(requeima()), kb=kb)
    assert not hasattr(r, "caminho") and not hasattr(r, "classe_real")
    certo = comparar_com_classe_real(
        r, ImagemSorteada(Path("ISCA_CAMINHO/arquivo_isca.jpg"), "Tomato_Late_blight"))
    assert isinstance(certo, Comparacao)
    assert certo.cnn_top1_coincide is True and certo.especialista_coincide is True
    assert certo.classe_real == "Tomato_Late_blight" and certo.caminho.name == "arquivo_isca.jpg"
    errado = comparar_com_classe_real(r, ImagemSorteada(Path("x.jpg"), "Tomato_healthy"))
    assert errado.cnn_top1_coincide is False and errado.especialista_coincide is False
    for comp in (certo, errado):                                   # nem a comparação vaza no repr
        assert "ISCA" not in repr(comp) and "Tomato" not in repr(comp)


def test_comparacao_sem_cnn_e_classe_sem_traducao(folha, kb):
    sem_cnn = diagnosticar_amostra(*folha, "test", kb=kb)
    c = comparar_com_classe_real(sem_cnn, ImagemSorteada(Path("x.jpg"), "Tomato_Late_blight"))
    assert c.cnn_top1_coincide is None
    com_cnn = diagnosticar_amostra(*folha, "test", preditor=PreditorFalso(requeima()), kb=kb)
    desconhecida = comparar_com_classe_real(com_cnn, ImagemSorteada(Path("x.jpg"), "Milho_ferrugem"))
    assert desconhecida.classe_real_canonica is None
    assert desconhecida.cnn_top1_coincide is None and desconhecida.especialista_coincide is None


def test_comparacao_exige_um_diagnostico_pronto():
    with pytest.raises(ValueError):
        comparar_com_classe_real(None, ImagemSorteada(Path("x.jpg"), "Tomato_healthy"))


def test_importar_o_modulo_nao_carrega_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import aplicacao.pipeline; "
              "assert 'tensorflow' not in sys.modules, 'importou tensorflow'")
    resultado = subprocess.run([sys.executable, "-c", codigo, SRC_DIR],
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr

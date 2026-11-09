"""Testes da CLI (cli/formatacao.py, cli/menu.py e diagnostico_cli.py).

Sem teclado e sem TensorFlow: entradas simuladas, preditor falso, KB real e um projeto
sintético em tmp_path (manifest e imagens). As figuras vão para tmp_path (pasta_base): nenhum
teste grava em rotina_demo/.
"""

import builtins
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from aplicacao.contrato import carregar_class_names
from aplicacao.exportacao import ResultadoGravacao
from aplicacao.pipeline import ResultadoDiagnostico, diagnosticar_amostra
from aplicacao.sorteio import AmostraSorteada, ImagemSorteada, SorteadorTeste, SorteioInvalido
from cli import formatacao as fmt
from cli import menu
from cli.menu import Dependencias, executar
from ia.expert_system import carregar_kb
from common.config import MIN_LESION_AREA_PX
from ia.inferencia import AVISO_PESOS_LOCAIS, ModeloIndisponivel

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = list(carregar_class_names())
CLASSE_REAL = "Tomato_Early_blight"
ARQUIVO = "ISCA_ARQUIVO_00.JPG"
SEED = 731904528


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


# ------------------------------------------------------------------------------ apoio
def desenhar_folha():
    img = np.full((224, 224, 3), (150, 150, 150), np.uint8)
    cv2.ellipse(img, (112, 112), (85, 70), 0, 0, 360, (40, 160, 60), -1)
    for x, y in ((90, 100), (130, 120), (110, 80)):
        cv2.circle(img, (x, y), 9, (30, 70, 120), -1)
    return img


def criar_projeto(raiz):
    """data/<classe>/ISCA_ARQUIVO_00.JPG e data_split/split_manifest.csv (caminhos relativos)."""
    rel = f"data/{CLASSE_REAL}/{ARQUIVO}"
    (raiz / "data" / CLASSE_REAL).mkdir(parents=True)
    assert cv2.imwrite(str(raiz / rel), desenhar_folha())
    (raiz / "data_split").mkdir()
    pd.DataFrame([{"class": CLASSE_REAL, "filename": ARQUIVO, "split": "test",
                   "processed_path": "data_processed/x.jpg", "original_path": rel}]
                 ).to_csv(raiz / "data_split" / "split_manifest.csv", index=False)
    return raiz


@pytest.fixture
def projeto(tmp_path):
    return criar_projeto(tmp_path)


class PreditorFalso:
    """Chamável como o PreditorCNN. Aposta em Late blight; a classe real do projeto é Early blight."""

    def __init__(self):
        self.class_names = tuple(CLASSES)
        p = np.zeros(len(CLASSES), np.float32)
        p[CLASSES.index("Tomato_Late_blight")] = 0.7
        p[CLASSES.index("Tomato__Target_Spot")] = 0.2
        p[CLASSES.index("Tomato_Leaf_Mold")] = 0.1
        self.vetor = p

    def __call__(self, segmentada_bgr, extensao=".jpg"):
        return self.vetor


class Terminal:
    """Entrada e saída simuladas. Cada item da fila é um texto ou KeyboardInterrupt;
    fila vazia = fim da entrada (EOFError). O prompt vai para a tela, como no input() real."""

    def __init__(self, *entradas):
        self.fila = list(entradas)
        self.linhas = []
        self.marcas = []          # posição (em linhas) de cada prompt "Pressione Enter..."
        self.outras = []          # posição de cada prompt "Analisar outra?"

    def saida(self, texto=""):
        self.linhas.append(str(texto))

    def entrada(self, prompt=""):
        self.linhas.append(prompt)
        if prompt == fmt.PROMPT_REVELAR:
            self.marcas.append(len(self.linhas))
        if prompt == fmt.PROMPT_OUTRA:
            self.outras.append(len(self.linhas))
        if not self.fila:
            raise EOFError
        valor = self.fila.pop(0)
        if valor is KeyboardInterrupt:
            raise KeyboardInterrupt
        return valor

    # Os textos longos são quebrados em linhas pela formatação; para procurar uma frase
    # inteira, as quebras e os recuos viram um espaço só.
    @property
    def texto(self):
        return " ".join("\n".join(self.linhas).split())

    def antes_do_enter(self, k=0):
        """Tudo o que foi mostrado antes do k-ésimo Enter de revelação (da análise k)."""
        inicio = self.outras[k - 1] if k > 0 else 0
        return " ".join("\n".join(self.linhas[inicio:self.marcas[k]]).split())

    def depois_do_enter(self, k=0):
        fim = self.outras[k] if len(self.outras) > k else len(self.linhas)
        return " ".join("\n".join(self.linhas[self.marcas[k]:fim]).split())


def deps_de_teste(raiz, kb, preditor="padrao", seed=SEED, **troca):
    preditor = PreditorFalso() if preditor == "padrao" else preditor
    base = dict(
        criar_sorteador=lambda: SorteadorTeste("data_split/split_manifest.csv", raiz),
        carregar_preditor=lambda: preditor,
        obter_kb=lambda: kb,
        gerar_seed=lambda: seed,
        pasta_base=raiz / "figuras",
        raiz=raiz,
    )
    base.update(troca)
    return Dependencias(**base)


def rodar(terminal, deps):
    return executar(entrada=terminal.entrada, saida=terminal.saida, deps=deps)


FLUXO = ["1", "", "n", "0"]          # opção 1, Enter, não analisar outra, sair


# ---------------------------------------------------------------------- fluxo completo
def test_fluxo_completo_mostra_as_sete_secoes_e_a_comparacao(projeto, kb):
    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb)) == 0
    assert t.fila == []                                       # consumiu exatamente as 4 entradas
    antes, depois = t.antes_do_enter(), t.depois_do_enter()
    for titulo in ("DIAGNÓSTICO", "[1] CNN", "[2] Medidas da folha", "[3] Avisos",
                   "[4] Regras disparadas", "[5] Informações que ajudariam",
                   "[6] Conclusão", "[7] Explicação do sistema especialista"):
        assert titulo in antes
    assert "Divisão: test" in antes and "1) Analisar imagem aleatória" in antes
    assert "0) Sair" in antes
    assert "COMPARAÇÃO COM A CLASSE REAL" in depois and "Até logo." in t.texto


def test_secoes_trazem_o_que_o_plano_promete(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    antes = t.antes_do_enter()
    assert "softmax" in antes and "não é confiança calibrada" in antes        # top 3
    assert "Requeima do tomateiro" in antes and "70.0%" in antes
    assert "Aviso sobre a CNN" in antes and AVISO_PESOS_LOCAIS.split(":")[0] in antes
    assert "cultura inferida pela CNN: tomate." in antes
    assert "a CLI não pergunta" in antes                                       # pendências
    assert "não é uma probabilidade calibrada" in antes                        # CF
    assert "Carregando modelo" in antes and "Modelo carregado." in antes


def test_legendas_das_medidas_seguem_o_codigo_de_medidas_lesao(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    antes = t.antes_do_enter()
    assert "Fora do verde saudável" in antes and "inclui marrom, amarelo" in antes
    assert "área da folha segmentada, não da imagem inteira" in antes
    assert "0 = na borda da folha, 1 = ponto mais interno da folha" in antes
    assert "(média dos pixels das regiões marrons)" in antes          # média por pixel
    assert "não é uma segmentação de lesão validada" in antes
    assert f"com pelo menos {MIN_LESION_AREA_PX} pixels" in antes        # lido do config
    assert "qualquer cor" not in antes and "centro" not in antes


def test_o_minimo_de_pixels_da_legenda_vem_do_config(projeto, kb, monkeypatch):
    monkeypatch.setattr(fmt, "MIN_LESION_AREA_PX", 987)
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    assert "com pelo menos 987 pixels" in t.antes_do_enter()


def test_a_seed_so_identifica_o_sorteio_sem_prometer_reproducao(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    depois = t.depois_do_enter()
    assert f"Seed: {SEED} (identifica este sorteio)" in depois
    assert "repete" not in depois and "demo_ia" not in depois


def test_conclusao_diz_se_e_conclusiva_e_explica_o_motivo(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    antes = t.antes_do_enter()
    amostra = SorteadorTeste("data_split/split_manifest.csv", projeto).sortear(seed=SEED)
    diag = diagnosticar_amostra(amostra.segmentada, amostra.mascara, "test", PreditorFalso(), kb)
    assert ("NÃO CONCLUSIVA" in antes) == (not diag.conclusao.conclusivo)
    if not diag.conclusao.conclusivo:
        assert "Motivo:" in antes
    if diag.regras_disparadas:
        assert "Evidência:" in antes


def test_motivos_de_inconclusivo_vem_dos_limiares_da_kb(kb):
    def hip(hid, cf):
        return SimpleNamespace(id=hid, cf_final=cf)

    def res(*hs):
        return SimpleNamespace(hipoteses=list(hs))

    so_cf = fmt.motivo_inconclusivo(res(hip("tomate_late_blight", 0.49),
                                        hip("tomate_early_blight", 0.10)), kb)
    assert len(so_cf) == 1 and "0.49" in so_cf[0] and "0.60" in so_cf[0]
    so_margem = fmt.motivo_inconclusivo(res(hip("tomate_late_blight", 0.70),
                                            hip("tomate_early_blight", 0.65)), kb)
    assert len(so_margem) == 1 and "diferença" in so_margem[0] and "0.15" in so_margem[0]
    assert "Pinta-preta" in so_margem[0]
    os_dois = fmt.motivo_inconclusivo(res(hip("tomate_late_blight", 0.50),
                                          hip("tomate_early_blight", 0.45)), kb)
    assert len(os_dois) == 2


# ----------------------------------------------------------------------------- vazamento
ISCAS_ANTES = (ARQUIVO, "ISCA_ARQUIVO", CLASSE_REAL, f"data/{CLASSE_REAL}", f"data\\{CLASSE_REAL}",
               str(SEED))


def test_antes_do_enter_nao_aparece_caminho_arquivo_classe_nem_seed(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    antes = t.antes_do_enter()
    for isca in ISCAS_ANTES + (str(projeto),):
        assert isca not in antes, isca


def test_depois_do_enter_aparecem_caminho_classe_real_e_seed(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    depois = t.depois_do_enter()
    assert f"data/{CLASSE_REAL}/{ARQUIVO}" in depois                      # caminho relativo
    assert str(SEED) in depois
    # classe real em formato do dataset E legível, a partir do classe_real_canonica
    assert f"{CLASSE_REAL} - Pinta-preta do tomateiro (Early blight)" in depois
    assert "CNN (1a opção)" in depois and "Requeima do tomateiro (Late blight) - NÃO coincide" in depois
    assert "Especialista" in depois and "Figuras salvas em" in depois


def test_o_texto_de_revelacao_vem_depois_do_pedido_de_enter(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb))
    texto = t.texto
    assert texto.index(fmt.PROMPT_REVELAR) < texto.index(ARQUIVO) < texto.index("Seed")


def test_funcoes_do_diagnostico_nao_recebem_a_imagem_sorteada():
    import inspect
    for funcao in (fmt.formatar_sorteio, fmt.formatar_diagnostico):
        assert not {"imagem", "comparacao", "caminho", "classe_real"} & set(
            inspect.signature(funcao).parameters)
    assert "imagem" in inspect.signature(fmt.formatar_revelacao).parameters


def test_formatar_sorteio_mostra_so_a_divisao():
    assert fmt.formatar_sorteio("test") == "Imagem sorteada do conjunto de teste.\nDivisão: test"


@pytest.mark.parametrize("onde", ["criar_sorteador", "sortear", "obter_kb", "diagnosticar"])
@pytest.mark.parametrize("excecao", [SorteioInvalido, RuntimeError])
def test_erros_antes_do_enter_nao_vazam_a_isca_no_texto_da_excecao(projeto, kb, onde, excecao):
    isca = f"falha em C:/ISCA_CAMINHO/{CLASSE_REAL}/{ARQUIVO}"

    def levanta(*a, **k):
        raise excecao(isca)

    class SorteadorQueFalha:
        def sortear(self, seed=None):
            levanta()

    troca = {
        "criar_sorteador": {"criar_sorteador": levanta},
        "sortear": {"criar_sorteador": SorteadorQueFalha},
        "obter_kb": {"obter_kb": levanta},
        "diagnosticar": {"diagnosticar": levanta},
    }[onde]
    t = Terminal("1", "0")
    assert rodar(t, deps_de_teste(projeto, kb, **troca)) == 0
    assert "ISCA" not in t.texto and CLASSE_REAL not in t.texto and ARQUIVO not in t.texto
    assert "Até logo." in t.texto                                    # voltou ao menu e saiu
    if excecao is SorteioInvalido and onde in ("criar_sorteador", "sortear"):
        assert fmt.MSG_SORTEIO_INVALIDO in t.texto
    else:
        assert f"erro inesperado ({excecao.__name__})" in t.texto


def test_sorteio_invalido_real_mostra_mensagem_generica(tmp_path, kb):
    # sem manifest: o SorteioInvalido real cita o arquivo do manifest, a tela não
    t = Terminal("1", "0")
    deps = deps_de_teste(tmp_path, kb)
    assert rodar(t, deps) == 0
    assert fmt.MSG_SORTEIO_INVALIDO in t.texto
    assert "split_manifest" not in t.texto and str(tmp_path) not in t.texto
    assert t.texto.count("0) Sair") == 2                              # o menu voltou a aparecer


# ------------------------------------------------------------------------------ menu
def test_opcao_invalida_pede_de_novo_e_aceita_espacos(projeto, kb):
    criados = []
    t = Terminal("x", "", "2", "11", "  0 ")
    deps = deps_de_teste(projeto, kb, criar_sorteador=lambda: criados.append(1))
    assert rodar(t, deps) == 0
    assert t.texto.count(fmt.MSG_OPCAO_INVALIDA) == 4 and criados == []
    assert t.texto.count("1) Analisar imagem aleatória") == 5       # o menu voltou a cada vez
    assert "Até logo." in t.texto


@pytest.mark.parametrize("entradas, revelou", [
    ([KeyboardInterrupt], False),                    # Ctrl+C no menu
    ([], False),                                     # fim da entrada no menu
    (["1", KeyboardInterrupt], False),               # Ctrl+C no Enter de revelação
    (["1"], False),                                  # EOF no Enter de revelação
    (["1", "", KeyboardInterrupt], True),            # Ctrl+C em "Analisar outra?"
    (["1", ""], True),                               # EOF em "Analisar outra?"
    (["1", "", "talvez", KeyboardInterrupt], True),  # Ctrl+C depois de resposta inválida
])
def test_ctrl_c_e_eof_saem_sem_traceback(projeto, kb, entradas, revelou):
    t = Terminal(*entradas)
    assert rodar(t, deps_de_teste(projeto, kb)) == 0
    assert "Encerrando." in t.texto and "Até logo." not in t.texto
    assert ("COMPARAÇÃO COM A CLASSE REAL" in t.texto) is revelou
    if not revelou:                                  # saiu antes da revelação: nada foi revelado
        assert ARQUIVO not in t.texto and CLASSE_REAL not in t.texto and str(SEED) not in t.texto
        assert not (projeto / "figuras").exists()


def test_resposta_invalida_em_analisar_outra_pergunta_de_novo(projeto, kb):
    t = Terminal("1", "", "talvez", "", "n", "0")
    assert rodar(t, deps_de_teste(projeto, kb)) == 0
    assert fmt.MSG_RESPOSTA_INVALIDA in t.texto and "Até logo." in t.texto


def test_analisar_outra_repete_e_aceita_sim_nao(projeto, kb):
    t = Terminal("1", "", "SIM", "", "não", "0")
    assert rodar(t, deps_de_teste(projeto, kb)) == 0
    assert t.texto.count("COMPARAÇÃO COM A CLASSE REAL") == 2 and len(t.marcas) == 2


# ---------------------------------------------------------------------- uma vez por sessão
def test_sorteador_kb_e_modelo_sao_criados_uma_vez_e_a_seed_vai_ao_sorteio(projeto, kb):
    chamadas = {"sorteador": 0, "kb": 0, "modelo": 0}
    seeds_pedidas, seeds_geradas = [], iter([111111111, 222222222])
    reais = {}

    def criar():
        chamadas["sorteador"] += 1
        reais["s"] = SorteadorTeste("data_split/split_manifest.csv", projeto)
        original = reais["s"].sortear
        reais["s"].sortear = lambda seed=None: (seeds_pedidas.append(seed), original(seed=seed))[1]
        return reais["s"]

    def carregar():
        chamadas["modelo"] += 1
        return PreditorFalso()

    def obter():
        chamadas["kb"] += 1
        return kb

    t = Terminal("1", "", "s", "", "n", "0")
    deps = deps_de_teste(projeto, kb, criar_sorteador=criar, carregar_preditor=carregar,
                         obter_kb=obter, gerar_seed=lambda: next(seeds_geradas))
    assert rodar(t, deps) == 0
    assert chamadas == {"sorteador": 1, "kb": 1, "modelo": 1}
    assert seeds_pedidas == [111111111, 222222222]
    assert t.texto.count("Carregando modelo") == 1                  # só na primeira análise
    for k, seed in enumerate(seeds_pedidas):                        # seed só depois do Enter
        assert str(seed) not in t.antes_do_enter(k)
        assert str(seed) in t.depois_do_enter(k)


def test_o_menu_nao_cria_nada_pesado(projeto, kb):
    def proibido(*a, **k):
        raise AssertionError("nada pesado deveria ser criado só para abrir o menu")

    t = Terminal("0")
    deps = deps_de_teste(projeto, kb, criar_sorteador=proibido, carregar_preditor=proibido,
                         obter_kb=proibido)
    assert rodar(t, deps) == 0 and "Carregando modelo" not in t.texto


# --------------------------------------------------------------------------- modo sem CNN
def test_modelo_indisponivel_segue_sem_cnn_com_aviso_e_nao_tenta_de_novo(projeto, kb):
    tentativas = []

    def carregar():
        tentativas.append(1)
        raise ModeloIndisponivel("modelo da CNN não encontrado: models/plant_disease_model.keras")

    t = Terminal("1", "", "s", "", "n", "0")
    assert rodar(t, deps_de_teste(projeto, kb, carregar_preditor=carregar)) == 0
    assert len(tentativas) == 1 and t.texto.count("COMPARAÇÃO COM A CLASSE REAL") == 2
    antes = t.antes_do_enter()
    assert "AVISO: a CNN não está disponível" in antes and "SEM a CNN" in antes
    assert "A CNN não foi usada nesta análise" in antes
    assert "softmax" not in antes and AVISO_PESOS_LOCAIS not in antes
    assert "CNN (1a opção): não utilizada nesta análise" in t.depois_do_enter() and "não utilizada nesta análise" in t.texto
    assert t.texto.count("AVISO: a CNN não está disponível") == 1


def test_erro_qualquer_ao_carregar_o_modelo_tambem_vira_modo_sem_cnn(projeto, kb):
    def carregar():
        raise ImportError("DLL do tensorflow ISCA_DLL")

    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, carregar_preditor=carregar)) == 0
    assert "(ImportError)" in t.texto and "ISCA_DLL" not in t.texto
    assert "A CNN não foi usada nesta análise" in t.texto


def test_sem_cnn_nunca_inventa_probabilidade_nem_trata_inconclusivo_como_saudavel(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb, preditor=None))
    antes = t.antes_do_enter()
    assert "%" not in antes.split("[2]")[0].split("[1]")[1]           # seção da CNN sem porcentagens
    conclusao = antes.split("[6] Conclusão")[1].split("[7]")[0]
    assert "saudável" not in conclusao.lower() or "CONCLUSIVA" in conclusao


# ------------------------------------------------------------------ máscara suspeita e erro
def amostra_falsa(projeto, mascara):
    original = desenhar_folha()
    imagem = ImagemSorteada(projeto / "data" / CLASSE_REAL / ARQUIVO, CLASSE_REAL)
    return AmostraSorteada(imagem, original, original.copy(), mascara)


class SorteadorFalso:
    def __init__(self, amostra):
        self.amostra = amostra

    def sortear(self, seed=None):
        return self.amostra


def test_mascara_suspeita_mostra_o_motivo_e_nenhuma_medida(projeto, kb):
    amostra = amostra_falsa(projeto, np.full((224, 224), 255, np.uint8))    # folha = imagem toda
    t = Terminal(*FLUXO)
    deps = deps_de_teste(projeto, kb, criar_sorteador=lambda: SorteadorFalso(amostra))
    assert rodar(t, deps) == 0
    antes = t.antes_do_enter()
    assert "Máscara: NÃO confiável" in antes and "possível vazamento do fundo" in antes
    assert "Medidas não calculadas" in antes and "nunca viram zero" in antes
    assert "Fora do verde saudável" not in antes and "Distância média" not in antes
    assert "COMPARAÇÃO COM A CLASSE REAL" in t.depois_do_enter()           # a análise continua


def test_diagnostico_com_status_erro_mostra_a_mensagem_e_nao_compara(projeto, kb):
    erro = ResultadoDiagnostico(
        status="erro", mensagem_erro="dados inválidos de teste", divisao="test", modo="sem_cnn",
        top3_cnn=(), qualidade=None, medidas=None, avisos=("um aviso qualquer",),
        regras_disparadas=(), pendencias=(), conclusao=None, explicacao="", resultado=None)
    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, diagnosticar=lambda *a, **k: erro)) == 0
    antes, depois = t.antes_do_enter(), t.depois_do_enter()
    assert "Não foi possível concluir o diagnóstico" in antes and "dados inválidos de teste" in antes
    assert "um aviso qualquer" in antes and ARQUIVO not in antes
    assert "Comparação indisponível" in depois and ARQUIVO in depois and CLASSE_REAL in depois


# ------------------------------------------------------------------------ gravação
def test_falha_de_gravacao_vira_mensagem_e_a_comparacao_continua(projeto, kb):
    ruim = ResultadoGravacao(False, None, (), "falha ao gravar original.png (disco cheio)")
    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, salvar_figuras=lambda *a, **k: ruim)) == 0
    depois = t.depois_do_enter()
    assert "As figuras NÃO foram salvas: falha ao gravar original.png (disco cheio)" in depois
    assert "Classe real" in depois and "Seed" in depois and "Até logo." in t.texto


def test_excecao_inesperada_na_gravacao_tambem_nao_derruba_a_cli(projeto, kb):
    def quebra(*a, **k):
        raise RuntimeError("explodiu")

    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, salvar_figuras=quebra)) == 0
    depois = t.depois_do_enter()
    assert "falha inesperada (RuntimeError)" in depois and "Classe real" in depois


def test_gravacao_parcial_mostra_a_pasta_criada(projeto, kb):
    parcial = ResultadoGravacao(False, projeto / "figuras" / "20260101_000000_abcd1234", (),
                                "falha ao gravar mascara.png (disco cheio)")
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb, salvar_figuras=lambda *a, **k: parcial))
    assert "Pasta criada" in t.depois_do_enter() and "20260101_000000_abcd1234" in t.texto


def test_a_pasta_gravada_corresponde_a_amostra_do_diagnostico(projeto, kb):
    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb)) == 0
    pastas = list((projeto / "figuras").iterdir())
    assert len(pastas) == 1
    assert sorted(p.name for p in pastas[0].iterdir()) == [
        "mascara.png", "original.png", "painel.png", "segmentada.png"]
    amostra = SorteadorTeste("data_split/split_manifest.csv", projeto).sortear(seed=SEED)
    ler = lambda nome: cv2.imdecode(np.frombuffer((pastas[0] / nome).read_bytes(), np.uint8),
                                    cv2.IMREAD_UNCHANGED)
    assert np.array_equal(ler("original.png"), amostra.original)
    assert np.array_equal(ler("segmentada.png"), amostra.segmentada)
    assert np.array_equal(ler("mascara.png"), (amostra.mascara > 0).astype(np.uint8) * 255)
    assert pastas[0].name in t.depois_do_enter()                    # a pasta mostrada é esta
    for nome in [pastas[0].name, *(p.name for p in pastas[0].iterdir())]:
        assert CLASSE_REAL not in nome and "ISCA" not in nome


# ------------------------------------------------------------------ formatação e texto
def test_funcoes_de_formatacao_sao_puras_nao_abrem_arquivo(projeto, kb, monkeypatch):
    amostra = SorteadorTeste("data_split/split_manifest.csv", projeto).sortear(seed=SEED)
    diag = diagnosticar_amostra(amostra.segmentada, amostra.mascara, "test", PreditorFalso(), kb)

    def proibido(*a, **k):
        raise AssertionError("a formatação não deve abrir arquivos")

    monkeypatch.setattr(builtins, "open", proibido)
    assert "[1] CNN" in fmt.formatar_diagnostico(diag, kb)
    assert fmt.formatar_menu() and fmt.formatar_sorteio("test")


def todos_os_textos(projeto, kb):
    """Telas de vários cenários (com CNN, sem CNN, máscara suspeita, falha) e as constantes."""
    saidas = []
    cenarios = [
        dict(),
        dict(preditor=None),
        dict(criar_sorteador=lambda: SorteadorFalso(
            amostra_falsa(projeto, np.full((224, 224), 255, np.uint8)))),
        dict(salvar_figuras=lambda *a, **k: ResultadoGravacao(False, None, (), "falha ao gravar x")),
    ]
    for troca in cenarios:
        t = Terminal("x", *FLUXO)
        rodar(t, deps_de_teste(projeto, kb, **troca))
        saidas.append(t.texto)
    t = Terminal("1", "", KeyboardInterrupt)
    rodar(t, deps_de_teste(projeto, kb))
    saidas.append(t.texto)
    saidas += [getattr(fmt, nome) for nome in dir(fmt)
               if nome.startswith(("MSG_", "PROMPT_")) and isinstance(getattr(fmt, nome), str)]
    saidas.append(fmt.formatar_aviso_modelo("motivo de teste"))
    saidas.append(fmt.formatar_erro_inesperado(RuntimeError("x")))
    return saidas


def test_todo_o_texto_cabe_em_cp850_e_cp1252(projeto, kb):
    for texto in todos_os_textos(projeto, kb):
        for codificacao in ("cp850", "cp1252"):
            texto.encode(codificacao)                                    # não pode levantar


def test_simbolos_so_ascii_e_acentos_so_em_letras(projeto, kb):
    for texto in todos_os_textos(projeto, kb):
        ruins = {c for c in texto if ord(c) >= 128 and not c.isalpha()}
        assert not ruins, ruins


def test_o_diagnostico_e_o_menu_cabem_em_78_colunas(projeto, kb):
    amostra = SorteadorTeste("data_split/split_manifest.csv", projeto).sortear(seed=SEED)
    diag = diagnosticar_amostra(amostra.segmentada, amostra.mascara, "test", PreditorFalso(), kb)
    for tela in (fmt.formatar_diagnostico(diag, kb), fmt.formatar_menu(),
                 fmt.formatar_aviso_modelo("motivo")):
        assert max(len(linha) for linha in tela.splitlines()) <= fmt.LARGURA


@pytest.mark.parametrize("prompt", [fmt.PROMPT_MENU, fmt.PROMPT_REVELAR, fmt.PROMPT_OUTRA])
def test_os_prompts_cabem_em_78_colunas(prompt):
    assert len(prompt) <= fmt.LARGURA


def test_banners_tem_acentos():
    assert "DIAGNÓSTICO" in fmt.formatar_diagnostico(
        SimpleNamespace(status="erro", mensagem_erro="x", avisos=()), None)
    assert "Diagnóstico de doenças" in fmt.formatar_menu()
    assert "demonstração" in fmt.formatar_menu()


# --------------------------------------------------------------- imports e ponto de entrada
def rodar_python(codigo, env_extra=None, entrada=None):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    env.pop("TF_CPP_MIN_LOG_LEVEL", None)
    env.pop("OPENCV_LOG_LEVEL", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", codigo, SRC_DIR], capture_output=True,
                          text=True, encoding="utf-8", env=env, input=entrada)


def test_importar_os_modulos_da_cli_nao_carrega_tensorflow():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import cli, cli.formatacao, cli.menu, "
              "diagnostico_cli; assert 'tensorflow' not in sys.modules, 'importou tensorflow'; "
              "assert cli.__file__.startswith(sys.argv[1])")
    resultado = rodar_python(codigo)
    assert resultado.returncode == 0, resultado.stderr


@pytest.mark.parametrize("previa, esperado", [(None, "2"), ("0", "0")])
def test_abrir_o_menu_e_sair_nao_carrega_tensorflow_e_ajusta_o_log(previa, esperado):
    codigo = ("import io, os, sys; sys.path.insert(0, sys.argv[1]); sys.stdin = io.StringIO('0\\n'); "
              "import diagnostico_cli; "
              f"assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == {esperado!r}; "
              "codigo = diagnostico_cli.principal(); "
              "assert codigo == 0 and 'tensorflow' not in sys.modules")
    extra = {"TF_CPP_MIN_LOG_LEVEL": previa} if previa else None
    resultado = rodar_python(codigo, extra)
    assert resultado.returncode == 0, resultado.stderr
    assert "Até logo." in resultado.stdout


@pytest.mark.parametrize("previa, esperado", [(None, "ERROR"), ("INFO", "INFO")])
def test_o_ponto_de_entrada_define_o_log_do_opencv_antes_de_importar_cv2(previa, esperado):
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); "
              "assert 'cv2' not in sys.modules; import diagnostico_cli, os; "
              f"assert os.environ['OPENCV_LOG_LEVEL'] == {esperado!r}; "
              "assert 'cv2' not in sys.modules, 'importou cv2 antes da hora'")
    extra = {"OPENCV_LOG_LEVEL": previa} if previa else None
    resultado = rodar_python(codigo, extra)
    assert resultado.returncode == 0, resultado.stderr


def test_com_o_ponto_de_entrada_o_imread_nao_escreve_o_caminho_no_stderr(tmp_path):
    # Sem OPENCV_LOG_LEVEL, cv2.imread de um arquivo que não abre escreve o caminho no stderr.
    isca = tmp_path / "Tomato_ISCA" / "ISCA_ARQUIVO.JPG"               # não existe: nada é criado
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import diagnostico_cli, cv2; "
              f"assert cv2.imread({str(isca)!r}) is None")
    resultado = rodar_python(codigo)
    assert resultado.returncode == 0, resultado.stderr
    assert "ISCA" not in resultado.stderr and "imread" not in resultado.stderr


def test_rodar_o_script_direto_abre_o_menu_e_sai_com_zero():
    script = os.path.join(SRC_DIR, "diagnostico_cli.py")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    resultado = subprocess.run([sys.executable, script], input="0\n", capture_output=True,
                               text=True, encoding="utf-8", env=env)
    assert resultado.returncode == 0, resultado.stderr
    assert "1) Analisar imagem aleatória" in resultado.stdout and "Até logo." in resultado.stdout
    assert "Traceback" not in resultado.stderr


def test_com_entrada_vazia_o_script_encerra_sem_traceback():
    script = os.path.join(SRC_DIR, "diagnostico_cli.py")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    resultado = subprocess.run([sys.executable, script], input="", capture_output=True,
                               text=True, encoding="utf-8", env=env)
    assert resultado.returncode == 0 and "Encerrando." in resultado.stdout
    assert "Traceback" not in resultado.stderr


def test_entrada_redirecionada_repete_o_texto_lido():
    script = os.path.join(SRC_DIR, "diagnostico_cli.py")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    resultado = subprocess.run([sys.executable, script], input="9\n0\n", capture_output=True,
                               text=True, encoding="utf-8", env=env)
    assert "Escolha uma opção: 9" in resultado.stdout and "Escolha uma opção: 0" in resultado.stdout


def test_o_modulo_menu_usa_so_dependencias_injetaveis():
    campos = set(Dependencias.__dataclass_fields__)
    assert {"criar_sorteador", "carregar_preditor", "obter_kb", "gerar_seed", "diagnosticar",
            "salvar_figuras", "pasta_base", "raiz", "gerar_laudo"} == campos
    assert menu.Dependencias().gerar_laudo is None                     # padrão: sem API
    assert menu.Dependencias().pasta_base is None                      # padrão: rotina_demo/cli


# ------------------------------------------------------------- laudo da IA generativa
def _laudo(texto="Texto do laudo para o agricultor.", fonte="llm", **kw):
    from ia.generativa.laudo import Laudo
    return Laudo(texto=texto, fonte=fonte, modelo=kw.pop("modelo", "m-teste"), **kw)


def test_sem_gerar_laudo_o_laudo_nao_aparece(projeto, kb):
    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb)) == 0
    assert "LAUDO EM LINGUAGEM SIMPLES" not in t.texto


def test_com_gerar_laudo_o_laudo_aparece_antes_do_enter(projeto, kb):
    recebidos = []

    def gerar(diag, kb_):
        recebidos.append(diag)
        return _laudo()

    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, gerar_laudo=gerar)) == 0
    antes = t.antes_do_enter()
    assert "LAUDO EM LINGUAGEM SIMPLES (IA GENERATIVA)" in antes
    assert "Texto do laudo para o agricultor." in antes and "m-teste" in antes
    assert antes.index("[7] Explicação") < antes.index("LAUDO EM LINGUAGEM SIMPLES")
    assert len(recebidos) == 1 and isinstance(recebidos[0], ResultadoDiagnostico)
    assert "COMPARAÇÃO COM A CLASSE REAL" in t.depois_do_enter()


def test_laudo_indisponivel_mostra_o_motivo_e_segue(projeto, kb):
    gerar = lambda d, k: _laudo(texto="explicação", fonte="motor", modelo=None,
                                aviso="variável de ambiente GEMINI_API_KEY não definida")
    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, gerar_laudo=gerar)) == 0
    assert "IA generativa indisponível" in t.antes_do_enter()
    assert "GEMINI_API_KEY" in t.antes_do_enter()
    assert "COMPARAÇÃO COM A CLASSE REAL" in t.depois_do_enter()


def test_laudo_que_falha_nao_derruba_a_analise_nem_vaza_a_mensagem(projeto, kb):
    def gerar(diag, kb_):
        raise RuntimeError("ISCA_CAMINHO data/Tomato_ISCA/arquivo_isca.jpg")

    t = Terminal(*FLUXO)
    assert rodar(t, deps_de_teste(projeto, kb, gerar_laudo=gerar)) == 0
    antes = t.antes_do_enter()
    assert "Não foi possível gerar o laudo da IA generativa (RuntimeError)" in antes
    assert "ISCA" not in antes
    assert "COMPARAÇÃO COM A CLASSE REAL" in t.depois_do_enter()


def test_laudo_com_doenca_fora_dos_fatos_mostra_alerta():
    tela = fmt.formatar_laudo(_laudo(doencas_fora_dos_fatos=("Ácaro-rajado (Two-spotted spider mite)",)))
    texto = " ".join(tela.split())
    assert "ATENÇÃO: o texto abaixo cita doença(s) que NÃO estão no diagnóstico" in texto
    assert "Ácaro-rajado" in texto


LAUDO_V2 = """O que o sistema encontrou:
O sistema não tem certeza. A hipótese mais forte é o mosaico do tomateiro, mas faltam
sinais para confirmar.

Por quê:
A análise da foto aponta mosaico, mas os sinais medidos na folha ainda não confirmam.

Perguntas para você:
- A folha tem um mosqueado de verde claro e escuro?
- As folhas estão deformadas, estreitas ou menores que o normal, principalmente as mais novas da planta?

O que fazer:
Se o diagnóstico for confirmado por um técnico, remova as plantas infectadas. Procure um agrônomo."""


def test_formatar_laudo_por_blocos_com_recuo_4_e_6():
    tela = fmt.formatar_laudo(_laudo(texto=LAUDO_V2))
    linhas = tela.splitlines()
    for titulo in ("O que o sistema encontrou:", "Por quê:", "Perguntas para você:",
                   "O que fazer:"):
        assert "    " + titulo in linhas, titulo                  # título sozinho, recuo 4
    corpo = linhas[linhas.index("    O que o sistema encontrou:"):]
    assert all(len(l) <= fmt.LARGURA for l in corpo)
    # só três recuos no corpo: 4 (título), 6 (parágrafo / "- ") e 8 (continuação de pergunta)
    recuos = {len(l) - len(l.lstrip()) for l in corpo if l}
    assert recuos == {4, 6, 8}
    # parágrafo de várias linhas do LLM vira um só, quebrado de novo, todo com recuo 6
    i = linhas.index("    O que o sistema encontrou:")
    assert linhas[i + 1].startswith("      O sistema não tem certeza.")
    assert all(l.startswith("      ") and l[6] != " " for l in linhas[i + 1:linhas.index("    Por quê:") - 1])
    assert "      - A folha tem um mosqueado de verde claro e escuro?" in linhas
    # pergunta longa: continuação alinhada ao texto da pergunta, não ao título
    j = next(k for k, l in enumerate(linhas) if l.startswith("      - As folhas"))
    assert linhas[j + 1].startswith("        ") and linhas[j + 1][8] != " "
    # uma linha em branco antes de cada título (exceto o primeiro)
    assert linhas[linhas.index("    Por quê:") - 1] == ""


def test_formatar_laudo_aceita_texto_na_linha_do_titulo_e_por_que_sem_acento():
    tela = fmt.formatar_laudo(_laudo(texto="Por que: a foto aponta mosaico.\nO QUE FAZER:\nProcure um agrônomo."))
    linhas = tela.splitlines()
    assert "    Por quê:" in linhas and "      a foto aponta mosaico." in linhas
    assert "    O que fazer:" in linhas and "      Procure um agrônomo." in linhas


def test_linha_do_guardrail_aprovado_e_corrigido():
    aprovado = fmt.formatar_laudo(_laudo(texto=LAUDO_V2, status_guardrail="aprovado",
                                         tentativas=1))
    assert "Guardrail: aprovado na 1a tentativa" in aprovado.splitlines()
    corrigido = fmt.formatar_laudo(_laudo(
        texto=LAUDO_V2, status_guardrail="corrigido", tentativas=2,
        violacoes=("cita doença fora do diagnóstico: Ácaro-rajado (Two-spotted spider mite)",
                   'usa "%" (força da evidência não é porcentagem)')))
    texto = " ".join(corrigido.split())
    assert ("Guardrail: corrigido na 2a tentativa (violações: cita doença fora do diagnóstico: "
            "Ácaro-rajado (Two-spotted spider mite); usa \"%\"") in texto
    assert "Ácaro-rajado" in corrigido                      # nome com hífen não é quebrado
    assert "    O que o sistema encontrou:" in corrigido.splitlines()   # o laudo do LLM aparece


def test_guardrail_reprovado_mostra_o_texto_do_sistema_especialista():
    tela = fmt.formatar_laudo(_laudo(texto="Explicação do motor.", fonte="motor", modelo=None,
                                     aviso="o guardrail reprovou", status_guardrail="reprovado",
                                     tentativas=2, violacoes=("indica dose ou produto (\"dose\")",)))
    texto = " ".join(tela.split())
    assert ("Guardrail: reprovado: mostrando o texto do sistema especialista (violações: "
            "indica dose ou produto (\"dose\"))") in texto
    assert "    Explicação do motor." in tela.splitlines()
    assert "Texto gerado pelo Gemini" not in tela and "indisponível" not in tela


def test_linha_do_guardrail_aprovado_com_aviso():
    from ia.generativa.guardrail import Violacao
    aviso = Violacao('texto sem acentos ("nao", "diagnostico"); escreva em português com '
                     "acentos", nivel="qualidade", curto="texto sem acentos")
    tela = fmt.formatar_laudo(_laudo(texto=LAUDO_V2, status_guardrail="aprovado_com_aviso",
                                     tentativas=2, violacoes=(aviso,)))
    linhas = tela.splitlines()
    assert "Guardrail: aprovado com aviso (texto sem acentos)" in linhas
    assert "Texto gerado pelo Gemini" in tela                   # o texto do LLM é mostrado
    assert "    O que o sistema encontrou:" in linhas
    tela.encode("cp850")


def test_sem_guardrail_nao_ha_linha_de_guardrail():
    assert "Guardrail" not in fmt.formatar_laudo(_laudo(texto=LAUDO_V2))
    indisponivel = fmt.formatar_laudo(_laudo(texto="x", fonte="motor", modelo=None,
                                             aviso="sem chave"))
    assert "Guardrail" not in indisponivel and "IA generativa indisponível" in indisponivel


def test_texto_da_cli_do_guardrail_cabe_no_console_do_windows():
    tela = fmt.formatar_laudo(_laudo(
        texto=LAUDO_V2, status_guardrail="corrigido", tentativas=2,
        violacoes=('"o_que_fazer" manda destruir, arrancar, eliminar, remover, retirar ou '
                   "queimar plantas sem a condição de confirmação (o caso não é conclusivo)",
                   'chama a força das regras de "fraca", "baixa" ou "pequena", mas nenhuma '
                   "regra disparou")))
    tela.encode("cp850")
    tela.encode("cp1252")
    assert all(len(l) <= fmt.LARGURA for l in tela.splitlines())


@pytest.mark.parametrize("argv, esperado", [
    ([], (False, None, False)),
    (["--sem-llm"], (True, None, False)),
    (["--seed", "594525008"], (False, 594525008, False)),
    (["--seed=42", "--sem-llm"], (True, 42, False)),
    (["--sem-llm", "--seed", "0"], (True, 0, False)),
    (["--sem-cache"], (False, None, True)),
    (["--seed", "7", "--sem-cache"], (False, 7, True)),
])
def test_ler_opcoes_aceita_sem_llm_sem_cache_e_seed(argv, esperado):
    import diagnostico_cli
    opcoes = diagnostico_cli.ler_opcoes(argv)
    assert opcoes == esperado
    assert (opcoes.sem_llm, opcoes.seed, opcoes.sem_cache) == esperado


@pytest.mark.parametrize("argv, usar_cache", [([], True), (["--sem-cache"], False)])
def test_sem_cache_repassa_usar_cache_false_ao_cliente_do_laudo(monkeypatch, argv, usar_cache):
    import diagnostico_cli
    from ia.generativa import laudo as modulo_laudo
    pedidos = []
    monkeypatch.setattr(modulo_laudo, "criar_gerador_de_laudo",
                        lambda **kw: pedidos.append(kw) or "gerador")
    capturadas = []
    monkeypatch.setattr(menu, "executar", lambda entrada, saida, deps: capturadas.append(deps) or 0)
    assert diagnostico_cli.principal(argv) == 0
    assert pedidos == [{"usar_cache": usar_cache}] and capturadas[0].gerar_laudo == "gerador"


def test_sem_cache_com_sem_llm_nao_cria_o_laudo(monkeypatch):
    import diagnostico_cli
    from ia.generativa import laudo as modulo_laudo
    monkeypatch.setattr(modulo_laudo, "criar_gerador_de_laudo",
                        lambda **kw: pytest.fail("não devia criar o gerador de laudo"))
    capturadas = []
    monkeypatch.setattr(menu, "executar", lambda entrada, saida, deps: capturadas.append(deps) or 0)
    assert diagnostico_cli.principal(["--sem-llm", "--sem-cache"]) == 0
    assert capturadas[0].gerar_laudo is None


@pytest.mark.parametrize("argv, trecho", [
    (["--xyz"], "Opção desconhecida: --xyz."),
    (["--seed"], "Falta o valor de --seed"),
    (["--seed", "abc"], "Seed inválida: 'abc'"),
    (["--seed=1.5"], "Seed inválida: '1.5'"),
    (["--seed", "-3"], "Seed inválida: -3"),
])
def test_ler_opcoes_recusa_opcao_desconhecida_e_seed_invalida(argv, trecho):
    import diagnostico_cli
    with pytest.raises(ValueError, match=re.escape(trecho)):
        diagnostico_cli.ler_opcoes(argv)


def test_seed_da_linha_de_comando_vai_para_as_dependencias(monkeypatch):
    import diagnostico_cli
    recebidas = []
    monkeypatch.setattr(menu, "executar", lambda entrada, saida, deps: recebidas.append(deps) or 0)
    assert diagnostico_cli.principal(["--seed", "594525008", "--sem-llm"]) == 0
    assert diagnostico_cli.principal(["--sem-llm"]) == 0
    com_seed, sem_seed = recebidas
    assert com_seed.gerar_seed() == 594525008 and com_seed.gerar_seed() == 594525008
    assert com_seed.gerar_laudo is None                               # --sem-llm continua valendo
    assert sem_seed.gerar_seed is menu._gerar_seed                    # sem --seed: sorteio normal


def test_com_seed_fixa_o_sorteio_usa_essa_seed_e_ela_so_aparece_depois_do_enter(projeto, kb):
    t = Terminal(*FLUXO)
    rodar(t, deps_de_teste(projeto, kb, seed=594525008))
    assert "594525008" not in t.antes_do_enter()
    assert "Seed: 594525008 (identifica este sorteio)" in t.depois_do_enter()


def test_ponto_de_entrada_recusa_opcao_desconhecida():
    codigo = ("import sys; sys.path.insert(0, sys.argv[1]); import diagnostico_cli; "
              "sys.exit(diagnostico_cli.principal(['--xyz']))")
    resultado = rodar_python(codigo)
    assert resultado.returncode == 2
    assert "Opção desconhecida: --xyz." in resultado.stderr
    assert "--sem-llm" in resultado.stderr and "--seed N" in resultado.stderr
    assert "--sem-cache" in resultado.stderr

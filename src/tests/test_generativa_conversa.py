"""Testes da conversa (ia/generativa/conversa.py): resposta livre -> sintomas -> o sistema
especialista recalcula. Transporte falso: nenhuma chamada à API do Gemini."""

import json

import cv2
import numpy as np
import pytest

from aplicacao.contrato import carregar_class_names
from aplicacao.pipeline import diagnosticar_amostra
from ia.expert_system import carregar_kb
from ia.generativa.cliente import ClienteLLM
from ia.generativa.conversa import (
    ID_DESCONHECIDO, MOTIVO_FORMATO, MOTIVO_NAO_PERGUNTADO, MOTIVO_SEM_TRECHO, MOTIVO_TRECHO,
    MOTIVO_VALOR, Interpretacao, interpretar_resposta, montar_prompt_conversa, reavaliar,
    registrar_conversa, schema_conversa, sintomas_para_perguntar, trecho_esta_na_resposta,
    validar_interpretacao,
)
from ia.generativa.guardrail import _apelidos, sem_acentos
from ia.generativa.prompts import SISTEMA_CONVERSA, VERSAO_PROMPT_CONVERSA
from processamento_imagens.preprocessing import segment_image

CLASSES = list(carregar_class_names())
RESPOSTA_MOSAICO = "Sim, tem umas manchas verde claro e escuro, tipo mosaico"


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


@pytest.fixture(scope="module")
def folha_verde(tmp_path_factory):
    """Folha sintética toda verde: nenhuma regra de imagem aponta doença."""
    img = np.full((224, 224, 3), (150, 150, 150), np.uint8)
    cv2.ellipse(img, (112, 112), (85, 70), 0, 0, 360, (40, 160, 60), -1)
    arquivo = str(tmp_path_factory.mktemp("folha") / "folha.png")
    assert cv2.imwrite(arquivo, img)
    _, segmentada, mascara = segment_image(arquivo)
    return segmentada, mascara


class PreditorMosaico:
    """CNN falsa com ~100% para mosaico (o caso do primeiro laudo real)."""
    class_names = tuple(CLASSES)

    def __init__(self):
        self.chamadas = 0

    def __call__(self, segmentada_bgr, extensao=".jpg"):
        self.chamadas += 1
        v = np.zeros(len(CLASSES), np.float32)
        v[CLASSES.index("Tomato__Tomato_mosaic_virus")] = 0.99
        v[CLASSES.index("Tomato__Tomato_YellowLeaf__Curl_Virus")] = 0.01
        return v


@pytest.fixture(scope="module")
def caso(folha_verde, kb):
    preditor = PreditorMosaico()
    diag = diagnosticar_amostra(*folha_verde, "test", preditor=preditor, kb=kb)
    return diag, preditor


class TransporteFalso:
    def __init__(self, resposta):
        self.resposta, self.chamadas = resposta, []

    def __call__(self, modelo, prompt, imagens, opcoes):
        self.chamadas.append((modelo, prompt, imagens, opcoes))
        return self.resposta if isinstance(self.resposta, str) else json.dumps(
            self.resposta, ensure_ascii=False)


def cliente_falso(resposta):
    t = TransporteFalso(resposta)
    return ClienteLLM(modelo="m-teste", modelo_reserva=None, transporte=t, usar_cache=False), t


def json_llm(**itens):
    """{"id": {"valor": v, "trecho": t}} a partir de id=(v, t)."""
    return {sid: {"valor": v, "trecho": t} for sid, (v, t) in itens.items()}


# --------------------------------------------------------------------- o que perguntar
def test_diagnostico_do_mosaico_comeca_inconclusivo(caso):
    diag, _ = caso
    assert diag.conclusao.hipotese_id == "tomate_mosaic"
    assert round(diag.conclusao.cf, 2) == 0.49 and not diag.conclusao.conclusivo


def test_sintomas_para_perguntar_so_devolve_origem_usuario(caso, kb):
    diag, _ = caso
    sintomas = sintomas_para_perguntar(diag, kb)
    assert sintomas and all(s.origem == "usuario" for s in sintomas)
    assert [s.id for s in sintomas] == ["mosaico_claro_escuro", "folha_deformada"]
    assert len({s.id for s in sintomas}) == len(sintomas)          # sem repetir


def test_sintomas_para_perguntar_respeita_o_maximo_e_a_ordem(caso, kb):
    diag, _ = caso
    assert [s.id for s in sintomas_para_perguntar(diag, kb, maximo=1)] == ["mosaico_claro_escuro"]


def test_sem_hipotese_ou_com_erro_nao_pergunta_nada(caso, kb):
    from dataclasses import replace
    diag, _ = caso
    assert sintomas_para_perguntar(replace(diag, status="erro"), kb) == ()
    sem = replace(diag, conclusao=replace(diag.conclusao, hipotese_id=None))
    assert sintomas_para_perguntar(sem, kb) == ()


def test_sintoma_de_imagem_nunca_e_perguntado(caso, kb, monkeypatch):
    from ia.generativa import conversa
    diag, _ = caso
    monkeypatch.setattr(conversa, "proxima_pergunta", lambda r, k: kb.sintomas["halo_amarelo"])
    assert all(s.origem == "usuario" for s in conversa.sintomas_para_perguntar(diag, kb))


# ------------------------------------------------------------------ fluxo completo
def test_resposta_com_mosqueado_deixa_o_mosaico_conclusivo(caso, kb, folha_verde):
    diag, preditor = caso
    sintomas = sintomas_para_perguntar(diag, kb)
    cliente, t = cliente_falso(json_llm(mosaico_claro_escuro=(1, "manchas verde claro e escuro"),
                                        folha_deformada=(None, None)))
    interp = interpretar_resposta(RESPOSTA_MOSAICO, sintomas, cliente)
    assert interp.sintomas == {"mosaico_claro_escuro": 1.0}
    assert interp.trechos == {"mosaico_claro_escuro": "manchas verde claro e escuro"}
    assert interp.descartados == () and interp.aviso is None
    assert interp.modelo == "m-teste" and interp.veio_do_cache is False
    _, prompt, imagens, opcoes = t.chamadas[0]
    assert opcoes["sistema"] == SISTEMA_CONVERSA and imagens == ()
    assert opcoes["schema"] == schema_conversa(sintomas) and RESPOSTA_MOSAICO in prompt

    antes = preditor.chamadas
    novo = reavaliar(diag, *folha_verde, sintomas=interp.sintomas, preditor=preditor, kb=kb)
    assert preditor.chamadas == antes + 1                    # reaproveita o mesmo preditor
    assert novo.conclusao.hipotese_id == "tomate_mosaic"
    assert novo.conclusao.conclusivo and round(novo.conclusao.cf, 2) == 0.90
    assert "R30" in {r.regra_id for r in novo.regras_disparadas}
    assert novo.divisao == diag.divisao


@pytest.mark.parametrize("sintomas, cf, conclusivo", [
    ({}, 0.49, False),
    ({"mosaico_claro_escuro": 1}, 0.90, True),
    ({"mosaico_claro_escuro": 1, "folha_deformada": 1}, 0.96, True),
    ({"mosaico_claro_escuro": 0}, 0.49, False),     # a base não tem regra contrária
])
def test_tabela_do_motor_com_as_respostas(caso, kb, folha_verde, sintomas, cf, conclusivo):
    diag, preditor = caso
    novo = reavaliar(diag, *folha_verde, sintomas=sintomas, preditor=preditor, kb=kb)
    assert novo.conclusao.hipotese_id == "tomate_mosaic"
    assert round(novo.conclusao.cf, 2) == cf and novo.conclusao.conclusivo is conclusivo


# --------------------------------------------------------------------------- guardrail
def test_llm_inventa_sintoma_com_trecho_que_nao_esta_na_resposta(caso, kb):
    diag, _ = caso
    cliente, _ = cliente_falso(json_llm(
        mosaico_claro_escuro=(1, "manchas verde claro e escuro"),
        folha_deformada=(1, "as folhas estão deformadas")))
    interp = interpretar_resposta(RESPOSTA_MOSAICO, sintomas_para_perguntar(diag, kb), cliente)
    assert interp.sintomas == {"mosaico_claro_escuro": 1.0}
    assert interp.descartados == (("folha_deformada", MOTIVO_TRECHO),)


def test_id_nao_perguntado_valor_invalido_e_null():
    ids = ("mosaico_claro_escuro", "folha_deformada", "teia_fina")
    dados = json_llm(mosaico_claro_escuro=(1.7, "mosaico"), folha_deformada=(None, None),
                     teia_fina=("sim", "mosaico"), aspecto_encharcado=(1, "mosaico"))
    dados["Tomato_mosaic_virus"] = {"valor": 1, "trecho": "mosaico"}
    aceitos, trechos, descartados = validar_interpretacao(dados, ids, RESPOSTA_MOSAICO)
    assert aceitos == {} and trechos == {}
    assert descartados == (("mosaico_claro_escuro", MOTIVO_VALOR),
                           ("teia_fina", MOTIVO_VALOR),
                           ("aspecto_encharcado", MOTIVO_NAO_PERGUNTADO),
                           (ID_DESCONHECIDO, MOTIVO_NAO_PERGUNTADO))  # id estranho não é repetido


@pytest.mark.parametrize("valor", [True, -0.1, float("nan"), float("inf"), "1", [1]])
def test_valor_nao_numerico_ou_fora_do_intervalo_e_descartado(valor):
    _, _, descartados = validar_interpretacao(
        {"teia_fina": {"valor": valor, "trecho": "tem"}}, ("teia_fina",), "tem sim")
    assert descartados == (("teia_fina", MOTIVO_VALOR),)


def test_valor_sem_trecho_e_item_fora_do_formato_sao_descartados():
    ids = ("teia_fina", "folha_deformada")
    dados = {"teia_fina": {"valor": 1, "trecho": None}, "folha_deformada": 1}
    _, _, descartados = validar_interpretacao(dados, ids, "tem teia")
    assert descartados == (("teia_fina", MOTIVO_SEM_TRECHO), ("folha_deformada", MOTIVO_FORMATO))
    assert validar_interpretacao([1, 2], ids, "x") == ({}, {}, ((ID_DESCONHECIDO, MOTIVO_FORMATO),))


def test_valores_intermediarios_e_zero_com_trecho_sao_aceitos():
    ids = ("mosaico_claro_escuro", "folha_deformada")
    texto = "Não tem mosqueado nenhum, mas algumas folhas estão um pouco deformadas."
    aceitos, trechos, descartados = validar_interpretacao(
        json_llm(mosaico_claro_escuro=(0, "Não tem mosqueado"),
                 folha_deformada=(0.5, "um pouco deformadas")), ids, texto)
    assert aceitos == {"mosaico_claro_escuro": 0.0, "folha_deformada": 0.5}
    assert descartados == ()


@pytest.mark.parametrize("trecho, ok", [
    ("manchas verde claro e escuro", True),
    ("MANCHAS   Verde  claro", True),                     # caixa e espaços extras
    ('"tipo mosaico."', True),                            # aspas e pontuação nas pontas
    ("nao tem", False),
    ("", False),
    ("   ", False),
])
def test_trecho_esta_na_resposta(trecho, ok):
    assert trecho_esta_na_resposta(trecho, RESPOSTA_MOSAICO) is ok


def test_trecho_compara_sem_acento():
    assert trecho_esta_na_resposta("nao ha teia", "Não há teia nenhuma.")
    assert trecho_esta_na_resposta("Não há", "nao ha teia")


# ------------------------------------------------------------- sem LLM / sem resposta
@pytest.mark.parametrize("texto", ["", "   ", "\n\t "])
def test_resposta_vazia_nao_chama_o_transporte(caso, kb, texto):
    diag, _ = caso
    cliente, t = cliente_falso(json_llm())
    interp = interpretar_resposta(texto, sintomas_para_perguntar(diag, kb), cliente)
    assert t.chamadas == [] and interp.sintomas == {} and interp.aviso is None


def test_sem_chave_avisa_e_nao_levanta(caso, kb, monkeypatch, tmp_path):
    for nome in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(nome, raising=False)
    diag, _ = caso
    interp = interpretar_resposta(RESPOSTA_MOSAICO, sintomas_para_perguntar(diag, kb),
                                  ClienteLLM(cache_dir=tmp_path))
    assert interp.sintomas == {} and "GEMINI_API_KEY" in interp.aviso


def test_json_invalido_avisa_sem_repetir_a_resposta(caso, kb):
    diag, _ = caso
    cliente, _ = cliente_falso("isto não é JSON {")
    interp = interpretar_resposta(RESPOSTA_MOSAICO, sintomas_para_perguntar(diag, kb), cliente)
    assert interp.sintomas == {} and "JSON" in interp.aviso and "isto" not in interp.aviso


# ------------------------------------------------------------------- o que vai ao LLM
def test_prompt_nao_tem_doenca_classe_caminho_nem_cnn(caso, kb):
    diag, _ = caso
    sintomas = sintomas_para_perguntar(diag, kb)
    cliente, t = cliente_falso(json_llm())
    interpretar_resposta(RESPOSTA_MOSAICO.replace("tipo mosaico", "acho"), sintomas, cliente)
    _, prompt, _, opcoes = t.chamadas[0]
    tudo = sem_acentos(prompt + "\n" + opcoes["sistema"] + json.dumps(opcoes["schema"])).lower()
    for d in kb.doencas.values():
        for apelido in _apelidos(sem_acentos(d.nome)):
            assert apelido not in tudo, apelido
    for proibido in ("tomato_", "potato_", "cnn", "data/", "data_split", ".jpg", "split",
                     "hipotese_principal", "tomate_mosaic"):
        assert proibido not in tudo, proibido


def test_prompt_lista_so_id_descricao_e_pergunta(caso, kb):
    diag, _ = caso
    sintomas = sintomas_para_perguntar(diag, kb)
    prompt = montar_prompt_conversa("  sim  ", sintomas)
    inicio = prompt.index("["), prompt.index("]") + 1
    lista = json.loads(prompt[inicio[0]:inicio[1]])
    assert lista == [{"id": s.id, "descricao": s.descricao, "pergunta": s.pergunta}
                     for s in sintomas]


def test_schema_tem_um_campo_por_sintoma_perguntado(caso, kb):
    diag, _ = caso
    sintomas = sintomas_para_perguntar(diag, kb)
    schema = schema_conversa(sintomas)
    assert schema["required"] == [s.id for s in sintomas] == list(schema["properties"])
    item = schema["properties"][sintomas[0].id]
    assert item["required"] == ["valor", "trecho"]


def test_tarefa_da_conversa_tem_a_escala_e_o_trecho_literal():
    texto = " ".join(SISTEMA_CONVERSA.split())
    assert VERSAO_PROMPT_CONVERSA == 1
    for trecho in ("1 = sim", "0 = não", '"um pouco"', '"talvez"', '"em algumas folhas"',
                   "Na dúvida, null", "LITERALMENTE", "Não invente ids"):
        assert trecho in texto, trecho


# --------------------------------------------------------------------------------- log
def test_log_tem_os_campos_e_nao_tem_o_texto_do_usuario(caso, kb, tmp_path):
    diag, _ = caso
    log = tmp_path / "conversa_log.jsonl"
    cliente, _ = cliente_falso(json_llm(mosaico_claro_escuro=(1, "manchas verde claro e escuro"),
                                        folha_deformada=(1, "invenção do modelo")))
    interpretar_resposta(RESPOSTA_MOSAICO, sintomas_para_perguntar(diag, kb), cliente, log=log)
    conteudo = log.read_text(encoding="utf-8")
    linha = json.loads(conteudo)
    assert set(linha) == {"data_hora", "modelo", "versao_prompt", "perguntados", "aceitos",
                          "descartados", "do_cache"}
    assert linha["perguntados"] == ["mosaico_claro_escuro", "folha_deformada"]
    assert linha["aceitos"] == {"mosaico_claro_escuro": 1.0} and linha["do_cache"] is False
    assert linha["descartados"] == [{"sintoma": "folha_deformada", "motivo": MOTIVO_TRECHO}]
    for proibido in ("manchas verde", "Sim, tem", "invenção", "Tomato_", "Potato_", "seed",
                     "data/", str(tmp_path)):
        assert proibido not in conteudo, proibido


def test_log_desligado_por_padrao_e_falha_ao_gravar_nao_quebra(caso, kb, tmp_path):
    import inspect
    diag, _ = caso
    assert inspect.signature(interpretar_resposta).parameters["log"].default is None
    pasta = tmp_path / "ocupado"
    pasta.mkdir()
    cliente, _ = cliente_falso(json_llm(mosaico_claro_escuro=(1, "tipo mosaico")))
    interp = interpretar_resposta(RESPOSTA_MOSAICO, sintomas_para_perguntar(diag, kb), cliente,
                                  log=pasta)
    assert interp.sintomas == {"mosaico_claro_escuro": 1.0}
    assert registrar_conversa(None, Interpretacao()) is False


def test_caminho_padrao_do_log_fica_em_rotina_demo():
    from common.config import CONVERSA_LOG
    assert CONVERSA_LOG.replace("\\", "/") == "rotina_demo/conversa_log.jsonl"

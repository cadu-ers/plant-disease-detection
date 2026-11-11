"""Testes do laudo da IA generativa (ia/generativa/laudo.py), sem rede: LLM falso."""

import copy
import json
import os

import cv2
import numpy as np
import pytest

from aplicacao.contrato import carregar_class_names
from aplicacao.pipeline import diagnosticar_amostra
from ia.expert_system import carregar_kb
from ia.generativa.cliente import ClienteLLM
from ia.generativa.laudo import (
    SCHEMA_LAUDO, VIOLACAO_JSON_INVALIDO, Laudo, criar_gerador_de_laudo,
    doencas_fora_dos_fatos, gerar_laudo, limpar_texto, montar_prompt, resumir_diagnostico,
)
from ia.generativa.prompts import (SISTEMA_BASE, SISTEMA_LAUDO, SISTEMA_LAUDO_V1,
                                   TAREFA_LAUDO, TAREFA_LAUDO_V2, TITULOS_LAUDO,
                                   VERSAO_PROMPT_LAUDO)
from processamento_imagens.preprocessing import segment_image

CLASSES = list(carregar_class_names())


@pytest.fixture(scope="module")
def kb():
    return carregar_kb()


@pytest.fixture(scope="module")
def folha(tmp_path_factory):
    img = np.full((224, 224, 3), (150, 150, 150), np.uint8)
    cv2.ellipse(img, (112, 112), (85, 70), 0, 0, 360, (40, 160, 60), -1)
    for x, y in ((90, 100), (130, 120), (110, 80)):
        cv2.circle(img, (x, y), 9, (30, 70, 120), -1)
    arquivo = str(tmp_path_factory.mktemp("folha") / "folha.png")
    assert cv2.imwrite(arquivo, img)
    _, segmentada, mascara = segment_image(arquivo)
    return segmentada, mascara


class PreditorFalso:
    class_names = tuple(CLASSES)

    def __init__(self, **pesos):
        self.pesos = pesos

    def __call__(self, segmentada_bgr, extensao=".jpg"):
        v = np.zeros(len(CLASSES), np.float32)
        for nome, p in self.pesos.items():
            v[CLASSES.index(nome)] = p
        return v


@pytest.fixture(scope="module")
def diag(folha, kb):
    preditor = PreditorFalso(Tomato_Late_blight=0.7, Tomato_Early_blight=0.2, Tomato__Target_Spot=0.1)
    return diagnosticar_amostra(*folha, "test", preditor=preditor, kb=kb)


class TransporteFalso:
    def __init__(self, resposta):
        self.resposta, self.chamadas = resposta, []

    def __call__(self, modelo, prompt, imagens, opcoes):
        self.chamadas.append((modelo, prompt, imagens, opcoes))
        return self.resposta


def cliente_falso(resposta, tmp_path=None):
    t = TransporteFalso(resposta)
    c = ClienteLLM(modelo="m-teste", modelo_reserva=None, transporte=t,
                   usar_cache=tmp_path is not None, cache_dir=tmp_path)
    return c, t


# ------------------------------------------------------------------ resumo (o que vai ao LLM)
def test_resumo_tem_os_fatos_do_motor(diag, kb):
    r = resumir_diagnostico(diag, kb)
    assert r["situacao"] in ("conclusivo", "inconclusivo", "sem_hipotese")
    assert r["hipotese_principal"]["nome"] == diag.conclusao.nome
    assert r["hipotese_principal"]["cf"] == round(diag.conclusao.cf, 2)
    assert [c["classe"] for c in r["rede_neural_top3"]] == [x[1] for x in diag.top3_cnn]
    assert r["recomendacoes"] == list(kb.doencas[diag.conclusao.hipotese_id].recomendacoes)
    assert r["rede_neural_concorda_com_a_hipotese"] is True
    json.dumps(r, ensure_ascii=False)                       # serializável


def test_inconclusivo_leva_motivo_e_perguntas(diag, kb):
    r = resumir_diagnostico(diag, kb)
    assert r["situacao"] == "inconclusivo"
    assert r["motivo_da_incerteza"] and r["perguntas_pendentes"]
    assert all(not p.startswith("Pergunta sugerida") and "_" not in p.split(" ")[0]
               for p in r["perguntas_pendentes"])          # só o texto da pergunta


def test_prompt_nao_tem_nome_de_classe_do_dataset_nem_caminho(diag, kb):
    prompt = montar_prompt(resumir_diagnostico(diag, kb))
    for proibido in ("Tomato_", "Potato_", "data/", ".jpg", ".JPG", "split", "classe_real"):
        assert proibido not in prompt


def test_resumo_diz_de_onde_veio_a_evidencia(diag, kb):
    origem = resumir_diagnostico(diag, kb)["origem_da_evidencia"]
    hip = diag.resultado.hipotese(diag.conclusao.hipotese_id)
    a_favor = [e for e in hip.evidencias if e.fonte == "regra" and e.cf > 0]
    assert origem == {"forca_pela_rede_neural": round(hip.cf_cnn, 2),
                      "maximo_pela_rede_neural": round(kb.config["peso_cnn"], 2),
                      "forca_pelas_regras": round(hip.cf_regras, 2),
                      "regras_que_dispararam": len(a_favor)}
    # preditor falso: 0,7 para a requeima do tomate -> força pela rede = peso_cnn * 0,7
    assert origem["forca_pela_rede_neural"] == round(kb.config["peso_cnn"] * 0.7, 2)
    assert '"origem_da_evidencia"' in montar_prompt(resumir_diagnostico(diag, kb))


def test_regra_contra_a_hipotese_nao_conta_como_regra_que_disparou(diag, kb):
    from dataclasses import replace
    from ia.expert_system.engine import Evidencia
    hid = diag.conclusao.hipotese_id
    resultado = copy.deepcopy(diag.resultado)
    hip = resultado.hipotese(hid)
    antes = sum(1 for e in hip.evidencias if e.fonte == "regra" and e.cf > 0)
    hip.evidencias += [Evidencia("regra", "RX", 0.4, 1.0, "a favor"),
                       Evidencia("regra", "RY", -0.3, 1.0, "contra")]
    origem = resumir_diagnostico(replace(diag, resultado=resultado), kb)["origem_da_evidencia"]
    assert origem["regras_que_dispararam"] == antes + 1


def test_sem_hipotese_a_origem_da_evidencia_e_nula(diag, kb):
    from dataclasses import replace
    sem = replace(diag, conclusao=replace(diag.conclusao, hipotese_id=None, nome=None, cf=None,
                                          conclusivo=False))
    r = resumir_diagnostico(sem, kb)
    assert r["situacao"] == "sem_hipotese" and r["origem_da_evidencia"] is None
    json.dumps(r, ensure_ascii=False)


def test_resumo_exige_diagnostico_ok(diag, kb):
    from dataclasses import replace
    with pytest.raises(ValueError):
        resumir_diagnostico(replace(diag, status="erro"), kb)


def test_sem_cnn_o_resumo_nao_inventa_rede_neural(folha, kb):
    d = diagnosticar_amostra(*folha, "test", kb=kb)
    r = resumir_diagnostico(d, kb)
    assert r["rede_neural_top3"] == [] and r["rede_neural_concorda_com_a_hipotese"] is None


# ------------------------------------------------------------------ prompt de sistema (v3)
def test_prompt_separa_base_e_tarefa_e_guarda_as_versoes_anteriores():
    assert VERSAO_PROMPT_LAUDO == 4
    assert SISTEMA_LAUDO.startswith(SISTEMA_BASE) and SISTEMA_LAUDO.endswith(TAREFA_LAUDO)
    assert SISTEMA_LAUDO_V1 != SISTEMA_LAUDO and "Regras obrigatórias" in SISTEMA_LAUDO_V1
    assert TAREFA_LAUDO_V2 != TAREFA_LAUDO and "título SOZINHO numa linha" in TAREFA_LAUDO_V2


def test_base_tem_os_limites_que_valem_para_qualquer_tarefa():
    texto = " ".join(SISTEMA_BASE.split())
    for trecho in ("não diagnostica", "Nunca indique produto", "dose", "marca",
                   "agrônomo", "Não saia do tema", "diga que não sabe"):
        assert trecho in texto, trecho


def test_tarefa_do_laudo_tem_as_regras_e_pede_o_json_do_guardrail():
    texto = " ".join(TAREFA_LAUDO.split())
    # inconclusivo: recomendações condicionais, irreversível só com condição, perguntas antes
    assert 'NÃO for "conclusivo"' in texto
    assert "Se o diagnóstico for confirmado por um técnico" in texto
    assert "nunca ordens" in texto
    assert "destruir, arrancar, eliminar, remover, retirar ou queimar plantas" in texto
    assert "a prioridade é responder às perguntas" in texto
    # zero regras: força zero não é "fraca"
    assert 'se "regras_que_dispararam" for 0, a força pelas regras é ZERO, não "fraca"' in texto
    assert "nenhuma regra confirmou a hipótese" in texto
    assert 'nunca chame as regras de "fracas", "baixas" ou "pequenas"' in texto
    # Por quê: de onde veio a evidência
    for campo in ("origem_da_evidencia", "forca_pela_rede_neural", "forca_pelas_regras",
                  "regras_que_dispararam", "maximo_pela_rede_neural"):
        assert campo in texto, campo
    # o que o guardrail confere também está pedido no prompt
    assert 'Nunca use o símbolo "%"' in texto and '"dose", "fungicida"' in texto
    assert "SOMENTE um objeto JSON" in texto
    for campo in SCHEMA_LAUDO["required"]:
        assert f'"{campo}"' in texto, campo
    for titulo in TITULOS_LAUDO:
        assert titulo in texto, titulo


# ----------------------------------------------------------------------------- geração
class TransporteSequencia(TransporteFalso):
    """Devolve as respostas em ordem (a última se repete)."""

    def __init__(self, *respostas):
        super().__init__(None)
        self.respostas = list(respostas)

    def __call__(self, modelo, prompt, imagens, opcoes):
        self.chamadas.append((modelo, prompt, imagens, opcoes))
        return self.respostas[min(len(self.chamadas), len(self.respostas)) - 1]


def cliente_seq(*respostas, tmp_path=None):
    t = TransporteSequencia(*[r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
                              for r in respostas])
    c = ClienteLLM(modelo="m-teste", modelo_reserva=None, transporte=t,
                   usar_cache=tmp_path is not None, cache_dir=tmp_path)
    return c, t


def resposta_valida(diag, kb, **troca):
    """JSON que passa no guardrail para o diag de teste (requeima do tomate, inconclusivo)."""
    r = resumir_diagnostico(diag, kb)
    dados = {
        "o_que_encontrou": "**O sistema não tem certeza**: a hipótese mais forte é a “requeima” – confira.",
        # o diag de teste não tem regra a favor: o guardrail exige "nenhuma regra confirmou"
        "por_que": "A análise da foto aponta requeima, mas nenhuma regra confirmou.",
        "perguntas": [r["perguntas_pendentes"][0]],
        "o_que_fazer": "Se o diagnóstico for confirmado por um técnico, remova e destrua as "
                       "plantas muito afetadas. Procure um agrônomo.",
        "hipotese_citada": r["hipotese_principal"]["nome"],
        "situacao_declarada": r["situacao"],
        "recomendacoes_usadas": [1],
        "perguntas_usadas": [0],
    }
    dados.update(troca)
    return dados


def test_resposta_valida_e_aprovada_na_1a_tentativa(diag, kb, tmp_path):
    cliente, t = cliente_seq(resposta_valida(diag, kb))
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, cliente, log=log)
    assert laudo.fonte == "llm" and laudo.modelo == "m-teste"
    assert (laudo.status_guardrail, laudo.tentativas, laudo.violacoes) == ("aprovado", 1, ())
    assert len(t.chamadas) == 1
    _, prompt, imagens, opcoes = t.chamadas[0]
    assert opcoes["sistema"] == SISTEMA_LAUDO and opcoes["schema"] == SCHEMA_LAUDO
    assert imagens == () and '"hipotese_principal"' in prompt
    assert "Sua resposta anterior violou" not in prompt
    # texto montado em código, com os títulos e limpo (sem markdown nem aspas curvas)
    assert laudo.texto.startswith('O que o sistema encontrou:\nO sistema não tem certeza: a '
                                  'hipótese mais forte é a "requeima" - confira.')
    assert "\n\nPor quê:\n" in laudo.texto and "\n\nPerguntas para você:\n- " in laudo.texto
    linha = json.loads(log.read_text(encoding="utf-8"))
    assert linha["status_guardrail"] == "aprovado" and linha["tentativas"] == 1
    assert linha["versao_prompt"] == VERSAO_PROMPT_LAUDO and linha["modelo"] == "m-teste"


def test_doenca_fora_dos_fatos_e_corrigida_na_2a_tentativa(diag, kb, tmp_path):
    ruim = resposta_valida(diag, kb, por_que="Nenhuma regra confirmou. Pode ser septoriose "
                                             "do tomateiro.")
    cliente, t = cliente_seq(ruim, resposta_valida(diag, kb))
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, cliente, log=log)
    assert laudo.fonte == "llm" and laudo.status_guardrail == "corrigido"
    assert laudo.tentativas == 2 and len(t.chamadas) == 2
    assert laudo.violacoes == (
        "cita doença fora do diagnóstico: Septoriose do tomateiro (Septoria leaf spot)",)
    segundo_prompt = t.chamadas[1][1]
    assert "Sua resposta anterior violou:\n- cita doença fora do diagnóstico" in segundo_prompt
    assert "septoriose" not in laudo.texto.lower()
    linha = json.loads(log.read_text(encoding="utf-8"))
    assert linha["status_guardrail"] == "corrigido" and linha["tentativas"] == 2
    assert linha["violacoes"] == [[{"violacao": v, "nivel": "seguranca"}
                                   for v in laudo.violacoes], []]


def test_duas_respostas_ruins_sao_reprovadas_e_o_texto_e_do_motor(diag, kb, tmp_path):
    ruim1 = resposta_valida(diag, kb, o_que_fazer="Destrua as plantas agora.")
    ruim2 = resposta_valida(diag, kb, o_que_fazer="Aplique 20 ml de fungicida.")
    cliente, t = cliente_seq(ruim1, ruim2)
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, cliente, log=log)
    assert laudo.fonte == "motor" and laudo.texto == diag.explicacao and laudo.modelo is None
    assert laudo.status_guardrail == "reprovado" and laudo.tentativas == 2
    assert laudo.violacoes == ("indica dose ou produto (quantidade com unidade)",
                               "indica dose ou produto (nome de classe de produto)")
    assert "reprovou as duas respostas" in laudo.aviso
    linha = json.loads(log.read_text(encoding="utf-8"))
    assert linha["status_guardrail"] == "reprovado"
    assert len(linha["violacoes"]) == 2
    assert any("destruir" in v["violacao"] for v in linha["violacoes"][0])
    assert {v["nivel"] for t in linha["violacoes"] for v in t} == {"seguranca"}


ACENTUADO = dict(
    o_que_encontrou="O sistema não tem certeza: a força das evidências não basta.",
    por_que="A análise da foto aponta o Vírus, mas nenhuma regra confirmou.",
    o_que_fazer="Se o diagnóstico for confirmado por um técnico, destrua as plantas. "
                "Procure um agrônomo.")


def test_acentos_do_llm_chegam_a_tela(diag, kb):
    from cli.formatacao import formatar_laudo
    cliente, _ = cliente_seq(resposta_valida(diag, kb, **ACENTUADO))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "aprovado"
    tela = formatar_laudo(laudo)
    for palavra in ("não", "força", "Vírus", "agrônomo", "análise", "diagnóstico", "técnico"):
        assert palavra in laudo.texto and palavra in tela, palavra


def test_acentos_separados_do_llm_sao_juntados_e_nao_somem(diag, kb):
    import unicodedata
    decomposto = {k: unicodedata.normalize("NFD", v) for k, v in ACENTUADO.items()}
    cliente, _ = cliente_seq(resposta_valida(diag, kb, **decomposto))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "aprovado"
    assert "não" in laudo.texto and "Vírus" in laudo.texto and "agrônomo" in laudo.texto


def test_com_acentos_o_guardrail_ainda_pega_destrua_as_plantas_sem_condicao(diag, kb):
    ruim = resposta_valida(diag, kb, **dict(ACENTUADO, o_que_fazer="Destrua as plantas. "
                                                                    "Procure um agrônomo."))
    cliente, _ = cliente_seq(ruim, resposta_valida(diag, kb, **ACENTUADO))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "corrigido"
    assert any("sem a condição de confirmação" in v for v in laudo.violacoes)
    assert "Vírus" in laudo.texto and "agrônomo" in laudo.texto


def test_resposta_sem_acentos_e_corrigida(diag, kb):
    sem = resposta_valida(diag, kb, o_que_encontrou="O sistema nao tem certeza sobre o "
                                                    "diagnostico.")
    cliente, t = cliente_seq(sem, resposta_valida(diag, kb, **ACENTUADO))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "corrigido"
    assert laudo.violacoes == ('texto sem acentos ("nao", "diagnostico"); escreva em '
                               "português com acentos",)
    assert "texto sem acentos" in t.chamadas[1][1] and "não tem certeza" in laudo.texto


def test_corrigido_so_por_acentos_tela_curta_e_log_completo(diag, kb, tmp_path):
    from cli.formatacao import formatar_laudo
    sem = resposta_valida(diag, kb, o_que_encontrou="O sistema nao tem certeza sobre o "
                                                    "diagnostico.")
    cliente, _ = cliente_seq(sem, resposta_valida(diag, kb, **ACENTUADO))
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, cliente, log=log)
    # tela: linha curta, sem o detalhe técnico
    tela = formatar_laudo(laudo)
    assert "Guardrail: aprovado (texto revisado automaticamente)" in tela
    assert "sem acentos" not in tela
    # log: não muda (status "corrigido", tentativas e a violação completa com o nível)
    linha = json.loads(log.read_text(encoding="utf-8"))
    assert linha["status_guardrail"] == "corrigido" and linha["tentativas"] == 2
    assert linha["violacoes"] == [[{"violacao": 'texto sem acentos ("nao", "diagnostico"); '
                                                "escreva em português com acentos",
                                    "nivel": "qualidade"}], []]


# ------------------------------------------------- níveis: segurança x qualidade
SEM_ACENTO = dict(o_que_encontrou="O sistema nao tem certeza sobre o diagnostico.")
AVISO_ACENTO = ('texto sem acentos ("nao", "diagnostico"); escreva em português com acentos')


def test_duas_vezes_sem_acentos_e_aprovado_com_aviso(diag, kb, tmp_path):
    sem = resposta_valida(diag, kb, **SEM_ACENTO)
    cliente, t = cliente_seq(sem, sem)
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, cliente, log=log)
    assert len(t.chamadas) == 2                         # qualidade também pede a 2a tentativa
    assert laudo.fonte == "llm" and laudo.status_guardrail == "aprovado_com_aviso"
    assert laudo.tentativas == 2 and laudo.violacoes == (AVISO_ACENTO,)
    assert laudo.violacoes[0].nivel == "qualidade" and laudo.texto.startswith(
        "O que o sistema encontrou:\nO sistema nao tem certeza")
    linha = json.loads(log.read_text(encoding="utf-8"))
    assert linha["status_guardrail"] == "aprovado_com_aviso"
    assert linha["violacoes"] == [[{"violacao": AVISO_ACENTO, "nivel": "qualidade"}]] * 2


def test_seguranca_que_sobra_na_2a_reprova_mesmo_com_qualidade_junto(diag, kb, tmp_path):
    sem = resposta_valida(diag, kb, **SEM_ACENTO)
    perigosa = resposta_valida(diag, kb, **dict(SEM_ACENTO, o_que_fazer="Destrua as plantas."))
    cliente, _ = cliente_seq(sem, perigosa)
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, cliente, log=log)
    assert laudo.fonte == "motor" and laudo.status_guardrail == "reprovado"
    assert {v.nivel for v in laudo.violacoes} == {"seguranca", "qualidade"}
    niveis = [sorted(v["nivel"] for v in t)
              for t in json.loads(log.read_text(encoding="utf-8"))["violacoes"]]
    assert niveis == [["qualidade"], ["qualidade", "seguranca"]]


def test_seguranca_na_1a_e_so_qualidade_na_2a_e_aprovado_com_aviso(diag, kb):
    perigosa = resposta_valida(diag, kb, o_que_fazer="Aplique uma dose de fungicida.")
    cliente, t = cliente_seq(perigosa, resposta_valida(diag, kb, **SEM_ACENTO))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "aprovado_com_aviso" and laudo.violacoes == (AVISO_ACENTO,)
    assert "fungicida" not in laudo.texto and "indica dose" in t.chamadas[1][1]


def test_llm_cai_na_correcao_de_qualidade_e_a_1a_sai_com_aviso(diag, kb):
    from ia.generativa.cliente import LLMIndisponivel
    respostas = iter([json.dumps(resposta_valida(diag, kb, **SEM_ACENTO))])

    def transporte(modelo, prompt, imagens, opcoes):
        try:
            return next(respostas)
        except StopIteration:
            raise LLMIndisponivel("sem rede") from None

    cliente = ClienteLLM(modelo="m-teste", modelo_reserva=None, transporte=transporte,
                         usar_cache=False)
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.fonte == "llm" and laudo.status_guardrail == "aprovado_com_aviso"
    assert laudo.tentativas == 1 and laudo.violacoes == (AVISO_ACENTO,)


def test_niveis_das_violacoes(diag, kb):
    from ia.generativa.guardrail import Violacao, nivel_de, so_qualidade, validar_laudo
    r = resumir_diagnostico(diag, kb)
    ruim = resposta_valida(diag, kb, hipotese_citada="Outra", situacao_declarada="conclusivo",
                           recomendacoes_usadas=[99], por_que="Pode ser septoriose do "
                           "tomateiro; as regras são fracas.",
                           o_que_fazer="Destrua as plantas com 10 ml de fungicida, 30%.",
                           **SEM_ACENTO)
    violacoes = validar_laudo(ruim, r, kb)
    assert [nivel_de(v) for v in violacoes].count("qualidade") == 1
    assert all(nivel_de(v) == "seguranca" for v in violacoes if "acentos" not in v)
    assert len(violacoes) >= 9                          # todas as de segurança da lista
    assert nivel_de(VIOLACAO_JSON_INVALIDO) == "seguranca"    # texto comum: segurança
    assert so_qualidade([Violacao("x", "qualidade")]) and not so_qualidade([])
    assert not so_qualidade([Violacao("x", "qualidade"), "y"])
    with pytest.raises(ValueError):
        Violacao("x", "leve")


def test_prompt_pede_acentos_e_nao_pede_ascii():
    texto = " ".join(SISTEMA_LAUDO.split())
    assert "COM todos os acentos" in texto and "nunca troque uma letra acentuada" in texto
    assert "ASCII" not in SISTEMA_LAUDO and "sem acentos" not in texto


def test_json_invalido_conta_como_violacao(diag, kb):
    cliente, t = cliente_seq("isto não é JSON {", resposta_valida(diag, kb))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "corrigido"
    assert laudo.violacoes == (VIOLACAO_JSON_INVALIDO,)
    # a resposta inválida do LLM não é repetida no prompt de correção
    assert "isto não é JSON" not in t.chamadas[1][1]


def test_dois_json_invalidos_sao_reprovados_sem_repetir_a_resposta(diag, kb):
    cliente, _ = cliente_seq("lixo {")
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.fonte == "motor" and laudo.status_guardrail == "reprovado"
    assert laudo.violacoes == (VIOLACAO_JSON_INVALIDO,) and "lixo" not in laudo.aviso


def test_bloco_vazio_depois_da_limpeza_e_violacao(diag, kb):
    cliente, _ = cliente_seq(resposta_valida(diag, kb, o_que_encontrou="\U0001F331\U0001F331"))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "reprovado"
    assert laudo.violacoes == ('campo "o_que_encontrou" ficou vazio depois da limpeza',)


def test_sem_chave_volta_para_a_explicacao_do_motor_sem_guardrail(diag, kb, monkeypatch, tmp_path):
    for nome in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(nome, raising=False)
    log = tmp_path / "guardrail_log.jsonl"
    laudo = gerar_laudo(diag, kb, ClienteLLM(cache_dir=tmp_path), log=log)
    assert laudo.fonte == "motor" and laudo.texto == diag.explicacao
    assert "GEMINI_API_KEY" in laudo.aviso
    assert laudo.status_guardrail is None and laudo.tentativas == 0
    assert not log.exists()                     # o guardrail não rodou: nada a registrar


def test_llm_cai_na_correcao_e_o_laudo_e_reprovado(diag, kb):
    from ia.generativa.cliente import LLMIndisponivel
    ruim = json.dumps(resposta_valida(diag, kb, perguntas=[], perguntas_usadas=[]))
    respostas = iter([ruim])

    def transporte(modelo, prompt, imagens, opcoes):
        try:
            return next(respostas)
        except StopIteration:
            raise LLMIndisponivel("sem rede") from None

    cliente = ClienteLLM(modelo="m-teste", modelo_reserva=None, transporte=transporte,
                         usar_cache=False)
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.fonte == "motor" and laudo.status_guardrail == "reprovado"
    assert laudo.tentativas == 1 and "sem rede" in laudo.aviso
    assert laudo.violacoes == ('"perguntas" está vazia, mas há perguntas pendentes',)


def test_falha_ao_gravar_o_log_nao_quebra_o_laudo(diag, kb, tmp_path):
    pasta = tmp_path / "ocupado"
    pasta.mkdir()                                # um diretório no lugar do arquivo de log
    cliente, _ = cliente_seq(resposta_valida(diag, kb))
    laudo = gerar_laudo(diag, kb, cliente, log=pasta)
    assert laudo.fonte == "llm" and laudo.status_guardrail == "aprovado"


def test_log_nao_tem_caminho_classe_do_dataset_nem_seed(diag, kb, tmp_path):
    log = tmp_path / "guardrail_log.jsonl"
    ruim = resposta_valida(diag, kb, hipotese_citada="Tomato_Late_blight",
                           o_que_fazer="Veja data/Tomato_Late_blight/abc.JPG")
    for respostas in ([resposta_valida(diag, kb)], [ruim, resposta_valida(diag, kb)], [ruim]):
        gerar_laudo(diag, kb, cliente_seq(*respostas)[0], log=log)
    conteudo = log.read_text(encoding="utf-8")
    assert len(conteudo.splitlines()) == 3
    for proibido in ("Tomato_", "Potato_", "Pepper", "data/", "data_split", ".jpg", ".JPG",
                     "abc", "seed", "classe_real", "caminho", str(tmp_path)):
        assert proibido not in conteudo, proibido
    for linha in map(json.loads, conteudo.splitlines()):
        assert set(linha) == {"data_hora", "modelo", "versao_prompt", "situacao",
                              "status_guardrail", "tentativas", "violacoes", "do_cache"}


def test_cache_reusa_o_laudo_e_o_log_marca_do_cache(diag, kb, tmp_path):
    ruim = resposta_valida(diag, kb, por_que="A foto aponta requeima; as regras são fracas.")
    cliente, t = cliente_seq(ruim, resposta_valida(diag, kb), tmp_path=tmp_path / "cache")
    log = tmp_path / "guardrail_log.jsonl"
    primeiro = gerar_laudo(diag, kb, cliente, log=log)
    segundo = gerar_laudo(diag, kb, cliente, log=log)
    assert len(t.chamadas) == 2                          # a 2a rodada veio toda do cache
    assert primeiro.texto == segundo.texto and segundo.status_guardrail == "corrigido"
    linhas = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert [l["do_cache"] for l in linhas] == [[False, False], [True, True]]


def test_regras_chamadas_de_fracas_sem_nenhuma_regra_sao_corrigidas(diag, kb):
    ruim = resposta_valida(diag, kb, por_que="A foto aponta requeima e a evidência das regras "
                                             "é fraca.")
    cliente, t = cliente_seq(ruim, resposta_valida(diag, kb))
    laudo = gerar_laudo(diag, kb, cliente)
    assert laudo.status_guardrail == "corrigido"
    assert laudo.violacoes == (
        'chama a força das regras de "fraca", "baixa" ou "pequena", mas nenhuma regra disparou',
        '"por_que" não diz que nenhuma regra confirmou a hipótese')
    assert "nenhuma regra confirmou" in laudo.texto


def test_gerador_sem_cache_cria_cliente_que_nao_usa_o_cache(monkeypatch):
    from ia.generativa import laudo as modulo
    criados = []

    class ClienteEspiao:
        def __init__(self, **kw):
            criados.append(kw)

    monkeypatch.setattr(modulo, "ClienteLLM", ClienteEspiao)
    modulo.criar_gerador_de_laudo(usar_cache=False)
    modulo.criar_gerador_de_laudo()
    assert criados == [{"usar_cache": False}, {"usar_cache": True}]


def test_gerador_de_sessao_usa_um_cliente_e_liga_o_log(diag, kb, tmp_path):
    cliente, t = cliente_seq(resposta_valida(diag, kb))
    log = tmp_path / "guardrail_log.jsonl"
    gerar = criar_gerador_de_laudo(cliente, log=log)
    assert isinstance(gerar(diag, kb), Laudo) and isinstance(gerar(diag, kb), Laudo)
    assert len(t.chamadas) == 2 and len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_o_log_padrao_do_gerador_fica_em_rotina_demo():
    import inspect
    from common.config import GUARDRAIL_LOG
    assert inspect.signature(criar_gerador_de_laudo).parameters["log"].default == GUARDRAIL_LOG
    assert inspect.signature(gerar_laudo).parameters["log"].default is None   # testes: sem log
    assert GUARDRAIL_LOG.replace("\\", "/") == "rotina_demo/guardrail_log.jsonl"


# ---------------------------------------------------------------- checagem de alucinação
def test_doenca_do_resumo_nao_e_alerta_mas_outra_e(diag, kb):
    r = resumir_diagnostico(diag, kb)
    assert doencas_fora_dos_fatos("Parece requeima do tomateiro (late blight).", r, kb) == ()
    assert doencas_fora_dos_fatos("Pode ser Early blight.", r, kb) == ()      # está no resumo
    fora = doencas_fora_dos_fatos("Também pode ser ácaro-rajado ou mosaico. Leaf mold?", r, kb)
    assert "Ácaro-rajado (Two-spotted spider mite)" in fora
    assert "Mofo-da-folha do tomateiro (Leaf mold)" in fora


def test_apelido_compartilhado_nao_gera_alerta_falso(diag, kb):
    r = resumir_diagnostico(diag, kb)
    # "late blight" também é apelido da requeima da BATATA, mas está no resumo (tomate)
    assert doencas_fora_dos_fatos("sinais de late blight", r, kb) == ()
    assert doencas_fora_dos_fatos("como a requeima da batata", r, kb) == ("Requeima da batata (Late blight)",)


# ----------------------------------------------------------------------------- limpeza
def test_limpar_texto():
    bruto = "# Título\n\n\n\n**Negrito** e * item\n* outro\nseta → fim… \U0001F600"
    assert limpar_texto(bruto) == "Título\n\nNegrito e * item\n- outro\nseta -> fim..."

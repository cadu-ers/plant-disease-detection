"""Testes do cliente do LLM (ia.generativa.cliente), sem rede: o transporte é falso."""

import json

import pytest

from ia.generativa.cliente import (
    ClienteLLM, Imagem, LLMIndisponivel, RespostaInvalida, ler_chave,
)


class ErroAPI(Exception):
    def __init__(self, code, msg="erro"):
        super().__init__(msg)
        self.code = code


class TransporteFalso:
    """Devolve os resultados na ordem (texto ou exceção a levantar); repete o último.

    `por_modelo` permite resultados diferentes por modelo: {"m1": [...], "m2": [...]}.
    """

    def __init__(self, resultados=("ok",), por_modelo=None):
        self.resultados = list(resultados)
        self.por_modelo = {m: list(r) for m, r in (por_modelo or {}).items()}
        self.chamadas = []

    def __call__(self, modelo, prompt, imagens, opcoes):
        self.chamadas.append({"modelo": modelo, "prompt": prompt, "imagens": tuple(imagens),
                              "opcoes": dict(opcoes)})
        fila = self.por_modelo.get(modelo, self.resultados)
        resultado = fila.pop(0) if len(fila) > 1 else fila[0]
        if isinstance(resultado, BaseException):
            raise resultado
        return resultado


class Relogio:
    """Substitui time.sleep: registra as esperas sem esperar de verdade."""

    def __init__(self):
        self.esperas = []

    def __call__(self, segundos):
        self.esperas.append(segundos)


def novo_cliente(tmp_path, transporte=None, **kw):
    kw.setdefault("modelo", "m1")
    kw.setdefault("modelo_reserva", None)
    kw.setdefault("dormir", Relogio())
    return ClienteLLM(cache_dir=tmp_path, transporte=transporte, **kw)


@pytest.fixture
def sem_variaveis(monkeypatch):
    for nome in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_MODEL_RESERVA"):
        monkeypatch.delenv(nome, raising=False)


# ---------- chamada básica ----------

def test_gerar_texto_repassa_prompt_sistema_e_temperatura(tmp_path):
    t = TransporteFalso(["  resposta  "])
    cliente = novo_cliente(tmp_path, t)
    assert cliente.gerar_texto("oi", sistema="sys", temperatura=0.2) == "resposta"
    chamada = t.chamadas[0]
    assert chamada["modelo"] == "m1"
    assert chamada["prompt"] == "oi"
    assert chamada["opcoes"] == {"sistema": "sys", "temperatura": 0.2, "schema": None}
    assert cliente.modelo_usado == "m1"


def test_gerar_json_decodifica_e_repassa_schema(tmp_path):
    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    t = TransporteFalso([json.dumps({"a": 0.5})])
    assert novo_cliente(tmp_path, t).gerar_json("p", schema=schema) == {"a": 0.5}
    assert t.chamadas[0]["opcoes"]["schema"] == schema
    assert t.chamadas[0]["opcoes"]["temperatura"] == 0.0


def test_gerar_json_invalido(tmp_path):
    with pytest.raises(RespostaInvalida, match="JSON"):
        novo_cliente(tmp_path, TransporteFalso(["não é json"])).gerar_json("p", schema={"type": "object"})


def test_gerar_json_exige_schema(tmp_path):
    with pytest.raises(ValueError):
        novo_cliente(tmp_path, TransporteFalso()).gerar_json("p", schema={})


def test_imagem_so_aceita_bytes_nunca_caminho(tmp_path):
    c = novo_cliente(tmp_path, TransporteFalso())
    with pytest.raises(TypeError):
        c.gerar_texto("p", imagens=["data/Tomato_Early_blight/foto.jpg"])
    with pytest.raises(ValueError):
        c.gerar_texto("   ")


@pytest.mark.parametrize("resposta", [None, "", "   "])
def test_resposta_vazia_e_invalida_e_nao_e_repetida(tmp_path, resposta):
    t = TransporteFalso([resposta])
    with pytest.raises(RespostaInvalida):
        novo_cliente(tmp_path, t, modelo_reserva="m2").gerar_texto("p")
    assert len(t.chamadas) == 1


# ---------- cache ----------

def test_cache_evita_segunda_chamada_e_funciona_sem_chave(tmp_path, sem_variaveis):
    t = TransporteFalso(["primeira"])
    assert novo_cliente(tmp_path, t).gerar_texto("p") == "primeira"
    # Novo cliente, SEM transporte e SEM chave: a resposta vem do cache.
    assert novo_cliente(tmp_path).gerar_texto("p") == "primeira"
    assert len(t.chamadas) == 1


def test_cache_diferencia_prompt_imagem_schema_e_modelo(tmp_path):
    t = TransporteFalso(["a", "b", "c", '{"x": 1}', "e"])
    c = novo_cliente(tmp_path, t)
    c.gerar_texto("p")
    c.gerar_texto("p2")
    c.gerar_texto("p", imagens=[Imagem(b"\x01\x02")])
    c.gerar_json("p", schema={"type": "object"})
    novo_cliente(tmp_path, t, modelo="outro").gerar_texto("p")
    assert len(t.chamadas) == 5


def test_veio_do_cache_diz_de_onde_veio_a_ultima_resposta(tmp_path):
    t = TransporteFalso(["a", ErroAPI(403)])
    c = novo_cliente(tmp_path, t)
    assert c.veio_do_cache is None                          # nenhuma chamada ainda
    c.gerar_texto("p")
    assert c.veio_do_cache is False                         # da API
    c.gerar_texto("p")
    assert c.veio_do_cache is True and len(t.chamadas) == 1  # do cache
    with pytest.raises(LLMIndisponivel):
        c.gerar_texto("outro prompt")
    assert c.veio_do_cache is None                          # a última chamada falhou


def test_veio_do_cache_vale_tambem_para_json_invalido(tmp_path):
    c = novo_cliente(tmp_path, TransporteFalso(["não é json"]))
    with pytest.raises(RespostaInvalida):
        c.gerar_json("p", schema={"type": "object"})
    assert c.veio_do_cache is False
    with pytest.raises(RespostaInvalida):
        c.gerar_json("p", schema={"type": "object"})       # a resposta ruim ficou no cache
    assert c.veio_do_cache is True


def test_cliente_sem_cache_nunca_le_nem_grava(tmp_path):
    t = TransporteFalso(["a", "b"])
    c = novo_cliente(tmp_path, t, usar_cache=False)
    assert c.gerar_texto("p") == "a" and c.gerar_texto("p") == "b"
    assert c.veio_do_cache is False and not list(tmp_path.iterdir())


def test_usar_cache_false_sempre_chama(tmp_path):
    t = TransporteFalso(["a", "b"])
    c = novo_cliente(tmp_path, t)
    assert c.gerar_texto("p", usar_cache=False) == "a"
    assert c.gerar_texto("p", usar_cache=False) == "b"
    assert not list(tmp_path.iterdir())


def test_cache_corrompido_e_ignorado(tmp_path):
    t = TransporteFalso(["nova"])
    c = novo_cliente(tmp_path, t)
    chave = c._chave_cache("m1", "p", (), {"sistema": None, "temperatura": 0.3, "schema": None})
    (tmp_path / f"{chave}.json").write_text("{quebrado", encoding="utf-8")
    assert c.gerar_texto("p") == "nova"
    assert len(t.chamadas) == 1


# ---------- chave e modelo ----------

def test_sem_chave_levanta_indisponivel(tmp_path, sem_variaveis):
    with pytest.raises(LLMIndisponivel, match="GEMINI_API_KEY"):
        novo_cliente(tmp_path).gerar_texto("p")
    assert ler_chave() is None


def test_le_chave_de_gemini_ou_google(monkeypatch, sem_variaveis):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert ler_chave() == "g"
    monkeypatch.setenv("GEMINI_API_KEY", "  k  ")
    assert ler_chave() == "k"


def test_modelos_pelas_variaveis_de_ambiente(monkeypatch, sem_variaveis):
    monkeypatch.setenv("GEMINI_MODEL", "principal-env")
    monkeypatch.setenv("GEMINI_MODEL_RESERVA", "reserva-env")
    c = ClienteLLM(usar_cache=False)
    assert c.modelos == ("principal-env", "reserva-env")
    assert ClienteLLM(modelo="explicito", modelo_reserva=None, usar_cache=False).modelos == ("explicito",)


def test_reserva_igual_ao_principal_e_ignorada(tmp_path):
    assert novo_cliente(tmp_path, modelo="m1", modelo_reserva="m1").modelos == ("m1",)


def test_chave_nao_aparece_no_repr(tmp_path):
    c = novo_cliente(tmp_path, TransporteFalso(), api_key="SEGREDO-123")
    assert "SEGREDO" not in repr(c)


# ---------- erros, novas tentativas e modelo reserva ----------

def test_erro_temporario_e_repetido_ate_dar_certo(tmp_path):
    t = TransporteFalso([ErroAPI(503), ErroAPI(504), "ok"])
    relogio = Relogio()
    c = novo_cliente(tmp_path, t, tentativas=3, espera_inicial_s=1.0, dormir=relogio)
    assert c.gerar_texto("p") == "ok"
    assert len(t.chamadas) == 3
    assert relogio.esperas == [1.0, 2.0]


@pytest.mark.parametrize("code, trecho", [
    (429, "cota"), (403, "GEMINI_API_KEY"), (404, "GEMINI_MODEL"), (503, "servidor"),
])
def test_erro_da_api_vira_indisponivel_com_mensagem(tmp_path, code, trecho):
    c = novo_cliente(tmp_path, TransporteFalso([ErroAPI(code)]))
    with pytest.raises(LLMIndisponivel, match=trecho):
        c.gerar_texto("p")
    assert not list(tmp_path.iterdir())  # erro não vai para o cache


def test_erro_permanente_nao_e_repetido_nem_troca_de_modelo(tmp_path):
    t = TransporteFalso([ErroAPI(403)])
    with pytest.raises(LLMIndisponivel, match="403"):
        novo_cliente(tmp_path, t, modelo_reserva="m2", tentativas=3).gerar_texto("p")
    assert [ch["modelo"] for ch in t.chamadas] == ["m1"]


def test_principal_fora_do_ar_usa_reserva_e_grava_cache(tmp_path, sem_variaveis):
    t = TransporteFalso(por_modelo={"m1": [ErroAPI(503)], "m2": ["da reserva"]})
    c = novo_cliente(tmp_path, t, modelo_reserva="m2", tentativas=2)
    assert c.gerar_texto("p") == "da reserva"
    assert c.modelo_usado == "m2"
    assert [ch["modelo"] for ch in t.chamadas] == ["m1", "m1", "m2"]
    # Depois, sem rede e sem chave, a resposta da reserva vem do cache.
    sem_rede = novo_cliente(tmp_path, modelo_reserva="m2")
    assert sem_rede.gerar_texto("p") == "da reserva"
    assert sem_rede.modelo_usado == "m2"


def test_modelo_inexistente_vai_direto_para_a_reserva(tmp_path):
    t = TransporteFalso(por_modelo={"m1": [ErroAPI(404)], "m2": ["ok"]})
    c = novo_cliente(tmp_path, t, modelo_reserva="m2", tentativas=3)
    assert c.gerar_texto("p") == "ok"
    assert [ch["modelo"] for ch in t.chamadas] == ["m1", "m2"]


def test_os_dois_modelos_fora_do_ar(tmp_path):
    t = TransporteFalso([ErroAPI(503)])
    c = novo_cliente(tmp_path, t, modelo_reserva="m2", tentativas=2)
    with pytest.raises(LLMIndisponivel, match="modelos tentados: m1, m2"):
        c.gerar_texto("p")
    assert len(t.chamadas) == 4


def test_erro_de_rede_e_repetido_e_vira_indisponivel(tmp_path):
    t = TransporteFalso([TimeoutError("lento")])
    c = novo_cliente(tmp_path, t, tentativas=2)
    with pytest.raises(LLMIndisponivel, match="TimeoutError"):
        c.gerar_texto("p")
    assert len(t.chamadas) == 2

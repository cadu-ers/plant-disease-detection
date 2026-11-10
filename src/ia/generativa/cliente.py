"""Cliente do LLM (Google Gemini) usado pela camada de IA generativa.

    cliente = ClienteLLM()
    texto = cliente.gerar_texto("Explique ...", sistema="Você é ...")
    dados = cliente.gerar_json("Classifique ...", schema={...}, imagens=[Imagem(b"...")])

Chave da API: lida da variável de ambiente GEMINI_API_KEY (ou GOOGLE_API_KEY).
Ela nunca é gravada em arquivo, impressa nem incluída em mensagens de erro.

Modelo: GEMINI_MODEL (variável de ambiente) ou common.config.LLM_MODEL. Erros
temporários (429, 500, 502, 503, 504, timeout, rede) são repetidos com espera
crescente (LLM_TENTATIVAS). Se o modelo principal continuar fora do ar, ou não
existir (404), o cliente tenta o modelo reserva (GEMINI_MODEL_RESERVA ou
LLM_MODEL_RESERVA). `modelo_usado` diz qual modelo respondeu a última chamada.

Cache: cada resposta recebida é gravada em rotina_demo/llm_cache/ (não versionado),
com chave = hash de (modelo, instrução de sistema, prompt, schema, temperatura e o
conteúdo das imagens). Uma pergunta repetida não gasta cota e, se já estiver no
cache (de qualquer um dos dois modelos), funciona sem internet e sem chave: é o
plano B da apresentação. `veio_do_cache` diz se a última resposta veio do cache (True),
da API (False) ou se a chamada falhou (None). ClienteLLM(usar_cache=False) não lê nem grava
o cache (rodadas de avaliação: cada laudo é uma chamada nova).

Erros:
    LLMIndisponivel   sem chave, biblioteca não instalada, sem rede, cota esgotada,
                      modelo inexistente etc. Quem chama decide o que fazer
                      (ex.: mostrar o texto do sistema especialista sem o LLM).
    RespostaInvalida  a API respondeu, mas vazio ou com JSON inválido.

A biblioteca google-genai só é importada na primeira chamada real à API: importar
este módulo é barato, e os testes usam um `transporte` falso (sem rede).
"""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from common.config import (
    LLM_CACHE_DIR, LLM_ESPERA_INICIAL_S, LLM_MODEL, LLM_MODEL_RESERVA, LLM_TENTATIVAS,
    LLM_TIMEOUT_S,
)

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]
VARIAVEIS_CHAVE = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
VARIAVEL_MODELO = "GEMINI_MODEL"
VARIAVEL_MODELO_RESERVA = "GEMINI_MODEL_RESERVA"
VERSAO_CACHE = 1
CODIGOS_TEMPORARIOS = frozenset({429, 500, 502, 503, 504})
# Exceções de rede do httpx (usado pela google-genai), reconhecidas pelo nome da classe
# para não importar o httpx aqui.
_NOMES_ERROS_REDE = ("Timeout", "ConnectError", "NetworkError", "RemoteProtocolError")
_PADRAO = object()


class LLMIndisponivel(RuntimeError):
    """O LLM não pode ser usado agora (chave, biblioteca, rede, cota ou modelo)."""


class RespostaInvalida(RuntimeError):
    """A API respondeu, mas a resposta está vazia ou não é o JSON esperado."""


@dataclass(frozen=True)
class Imagem:
    """Bytes de uma imagem enviada ao modelo. Só os bytes: nunca o nome do arquivo."""
    dados: bytes
    mime_type: str = "image/jpeg"


# transporte(modelo, prompt, imagens, opcoes) -> texto da resposta
Transporte = Callable[[str, str, Sequence[Imagem], dict], Optional[str]]


def ler_chave():
    """Chave da API das variáveis de ambiente, ou None."""
    for nome in VARIAVEIS_CHAVE:
        valor = os.environ.get(nome, "").strip()
        if valor:
            return valor
    return None


def _caminho(caminho):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else RAIZ_REPOSITORIO / caminho


def _temporario(erro):
    """Erro que vale repetir: sobrecarga, cota momentânea, timeout ou falha de rede."""
    if getattr(erro, "code", None) in CODIGOS_TEMPORARIOS:
        return True
    if isinstance(erro, (TimeoutError, ConnectionError)):
        return True
    return any(nome in type(erro).__name__ for nome in _NOMES_ERROS_REDE)


def _vale_trocar_de_modelo(erro):
    """Erro que pode sumir com outro modelo: fora do ar (depois das tentativas) ou 404."""
    return _temporario(erro) or getattr(erro, "code", None) == 404


def _mensagem_de_erro(erro):
    """Mensagem legível para um erro da API, sem repetir dados da requisição."""
    codigo = getattr(erro, "code", None)
    if codigo == 429:
        return "cota da API esgotada ou muitas requisições seguidas (429); tente de novo em instantes"
    if codigo in (401, 403):
        return f"chave da API recusada ({codigo}); confira a variável GEMINI_API_KEY"
    if codigo == 400 and "API key" in str(erro):
        return "chave da API inválida (400); confira a variável GEMINI_API_KEY"
    if codigo == 404:
        return (f"modelo não encontrado (404); defina outro em {VARIAVEL_MODELO} "
                "ou em common/config.py (LLM_MODEL)")
    if codigo is not None and codigo >= 500:
        return f"erro no servidor do Gemini ({codigo}); tente de novo em instantes"
    return f"{type(erro).__name__}: {erro}"


class _FalhaDoModelo(Exception):
    """Uso interno: o modelo falhou de vez (o erro original fica em .erro)."""

    def __init__(self, erro):
        super().__init__(str(erro))
        self.erro = erro


class _TransporteGemini:
    """Chamada real à API do Gemini (google-genai). Criado só quando é preciso."""

    def __init__(self, api_key, timeout_s):
        try:
            from google import genai
            from google.genai import types
        except ImportError as erro:
            raise LLMIndisponivel(
                f"biblioteca google-genai não instalada ({erro}); rode pip install -r requirements.txt"
            ) from None
        self._types = types
        self._cliente = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def __call__(self, modelo, prompt, imagens, opcoes):
        types = self._types
        partes = [types.Part.from_bytes(data=img.dados, mime_type=img.mime_type) for img in imagens]
        partes.append(prompt)
        config = types.GenerateContentConfig(
            system_instruction=opcoes.get("sistema"),
            temperature=opcoes.get("temperatura"),
            response_mime_type="application/json" if opcoes.get("schema") else None,
            response_json_schema=opcoes.get("schema"),
        )
        resposta = self._cliente.models.generate_content(model=modelo, contents=partes, config=config)
        return resposta.text


class ClienteLLM:
    """Gera texto ou JSON com o Gemini, com cache em disco e erros tratados."""

    def __init__(self, modelo=None, modelo_reserva=_PADRAO, api_key=None, timeout_s=LLM_TIMEOUT_S,
                 cache_dir=LLM_CACHE_DIR, usar_cache=True, transporte: Optional[Transporte] = None,
                 tentativas=LLM_TENTATIVAS, espera_inicial_s=LLM_ESPERA_INICIAL_S, dormir=time.sleep):
        self.modelo = modelo or os.environ.get(VARIAVEL_MODELO, "").strip() or LLM_MODEL
        if modelo_reserva is _PADRAO:
            modelo_reserva = os.environ.get(VARIAVEL_MODELO_RESERVA, "").strip() or LLM_MODEL_RESERVA
        self.modelo_reserva = modelo_reserva if modelo_reserva and modelo_reserva != self.modelo else None
        self.modelo_usado = None          # modelo que respondeu a última chamada
        self.veio_do_cache = None         # a última resposta veio do cache? (None: falhou)
        self._tentativas = max(1, int(tentativas))
        self._espera_inicial_s = espera_inicial_s
        self._dormir = dormir
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._cache_dir = _caminho(cache_dir) if usar_cache and cache_dir else None
        self._transporte = transporte

    def __repr__(self):  # nunca mostrar a chave
        return (f"ClienteLLM(modelo={self.modelo!r}, reserva={self.modelo_reserva!r}, "
                f"cache={'sim' if self._cache_dir else 'não'})")

    @property
    def modelos(self):
        """Modelos na ordem em que são tentados."""
        return (self.modelo,) + ((self.modelo_reserva,) if self.modelo_reserva else ())

    # ---------- cache ----------
    def _chave_cache(self, modelo, prompt, imagens, opcoes):
        conteudo = {
            "versao": VERSAO_CACHE,
            "modelo": modelo,
            "prompt": prompt,
            "sistema": opcoes.get("sistema"),
            "schema": opcoes.get("schema"),
            "temperatura": opcoes.get("temperatura"),
            "imagens": [[hashlib.sha256(img.dados).hexdigest(), img.mime_type] for img in imagens],
        }
        texto = json.dumps(conteudo, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()

    def _ler_cache(self, chave):
        if self._cache_dir is None:
            return None
        arquivo = self._cache_dir / f"{chave}.json"
        try:
            with open(arquivo, encoding="utf-8") as f:
                return json.load(f)["resposta"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _gravar_cache(self, chave, modelo, resposta):
        if self._cache_dir is None:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            fd, temporario = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"modelo": modelo, "resposta": resposta}, f, ensure_ascii=False)
            os.replace(temporario, self._cache_dir / f"{chave}.json")
        except OSError:
            pass  # cache é conveniência: falhar ao gravar não deve quebrar o diagnóstico

    # ---------- chamada ----------
    def _obter_transporte(self):
        if self._transporte is None:
            chave = self._api_key or ler_chave()
            if not chave:
                raise LLMIndisponivel(
                    "variável de ambiente GEMINI_API_KEY não definida "
                    "(crie a chave em aistudio.google.com)")
            self._transporte = _TransporteGemini(chave, self._timeout_s)
        return self._transporte

    def _chamar(self, prompt, imagens, opcoes, usar_cache):
        self.veio_do_cache = None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("o prompt deve ser um texto não vazio")
        imagens = tuple(imagens)
        if not all(isinstance(img, Imagem) for img in imagens):
            raise TypeError("imagens devem ser objetos Imagem (bytes), nunca caminhos")

        chaves = {m: self._chave_cache(m, prompt, imagens, opcoes) for m in self.modelos}
        if usar_cache:
            for modelo in self.modelos:
                em_cache = self._ler_cache(chaves[modelo])
                if em_cache is not None:
                    self.modelo_usado = modelo
                    self.veio_do_cache = True
                    return em_cache

        transporte = self._obter_transporte()
        ultimo_erro = None
        for modelo in self.modelos:
            try:
                resposta = self._tentar_modelo(transporte, modelo, prompt, imagens, opcoes)
            except _FalhaDoModelo as falha:
                ultimo_erro = falha.erro
                if _vale_trocar_de_modelo(falha.erro):
                    continue
                break
            if usar_cache:
                self._gravar_cache(chaves[modelo], modelo, resposta)
            self.modelo_usado = modelo
            self.veio_do_cache = False
            return resposta

        mensagem = _mensagem_de_erro(ultimo_erro)
        if len(self.modelos) > 1 and _vale_trocar_de_modelo(ultimo_erro):
            mensagem += f" (modelos tentados: {', '.join(self.modelos)})"
        raise LLMIndisponivel(mensagem)

    def _tentar_modelo(self, transporte, modelo, prompt, imagens, opcoes):
        """Chama um modelo, repetindo erros temporários. Levanta _FalhaDoModelo se desistir."""
        for tentativa in range(self._tentativas):
            try:
                resposta = transporte(modelo, prompt, imagens, opcoes)
            except (LLMIndisponivel, RespostaInvalida):
                raise
            except Exception as erro:  # rede, timeout, cota, chave recusada, modelo inexistente...
                if _temporario(erro) and tentativa < self._tentativas - 1:
                    self._dormir(self._espera_inicial_s * (2 ** tentativa))
                    continue
                raise _FalhaDoModelo(erro) from None
            if not isinstance(resposta, str) or not resposta.strip():
                raise RespostaInvalida(
                    "o modelo devolveu uma resposta vazia (talvez bloqueada pelo filtro)")
            return resposta
        raise AssertionError("inalcançável")  # pragma: no cover

    def gerar_texto(self, prompt, *, sistema=None, imagens=(), temperatura=0.3, usar_cache=True):
        """Resposta em texto livre."""
        opcoes = {"sistema": sistema, "temperatura": temperatura, "schema": None}
        return self._chamar(prompt, imagens, opcoes, usar_cache).strip()

    def gerar_json(self, prompt, *, schema, sistema=None, imagens=(), temperatura=0.0, usar_cache=True):
        """Resposta em JSON, no formato do `schema` (JSON Schema). Devolve o objeto decodificado.

        O schema força o formato na API, mas o CONTEÚDO (ex.: nomes de sintomas válidos)
        deve ser conferido por quem chama.
        """
        if not isinstance(schema, dict) or not schema:
            raise ValueError("schema deve ser um JSON Schema (dict) não vazio")
        opcoes = {"sistema": sistema, "temperatura": temperatura, "schema": schema}
        texto = self._chamar(prompt, imagens, opcoes, usar_cache)
        try:
            return json.loads(texto)
        except ValueError:
            raise RespostaInvalida(f"o modelo não devolveu JSON válido: {texto[:200]!r}") from None

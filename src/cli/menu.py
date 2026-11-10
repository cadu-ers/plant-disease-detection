"""Laço da CLI de demonstração: menu, análise de uma imagem de teste e revelação.

    executar(entrada=input, saida=print, deps=None) -> código de saída (0 = normal)

Entrada e saída são injetáveis (os testes usam entradas simuladas); as dependências pesadas
também (Dependencias). Fluxo de "1) Analisar imagem aleatória":

    sorteia (seed gerada aqui) -> mostra SÓ a divisão -> diagnostica -> mostra o diagnóstico
    -> (se houver gerar_laudo) mostra o laudo da IA generativa -> espera Enter -> compara
    com a classe real -> grava as figuras -> mostra a comparação, a pasta e a seed
    -> (se houver conversar e o diagnóstico não for conclusivo) conversa: perguntas, resposta
    em texto livre, "Entendi:", "Confirma? (s/n)", o sistema especialista recalcula, ANTES x
    DEPOIS e o laudo novo -> "Analisar outra? (s/n)"

A conversa vem DEPOIS da revelação: só então o usuário pode abrir a figura salva e olhar a
folha. Quem decide é o sistema especialista; o LLM só traduz a resposta em sintomas, e o
usuário confirma o que foi entendido antes do recálculo.

Regras importantes:
    - O TensorFlow NÃO é importado ao abrir o menu: o modelo só é carregado na primeira
      análise (mensagem "Carregando modelo..."). Se faltar, a sessão segue sem CNN, com aviso.
    - O sorteador (manifest), a KB e o modelo são criados UMA vez por sessão.
    - Antes do Enter nada do que vai à tela contém caminho, nome de arquivo, classe real nem
      seed. Mensagens de erro antes da revelação são genéricas (só o tipo do erro), porque o
      texto original de uma exceção poderia conter um caminho.
    - Ctrl+C e fim da entrada (EOF) encerram sem traceback. Falha de gravação não impede a
      comparação.
"""

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aplicacao.exportacao import ResultadoGravacao, salvar_figuras
from aplicacao.pipeline import comparar_com_classe_real, diagnosticar_amostra, obter_kb
from aplicacao.sorteio import RAIZ_REPOSITORIO, SorteadorTeste, SorteioInvalido
from cli import formatacao as fmt
from ia.generativa.conversa import sintomas_para_perguntar
from ia.inferencia import ModeloIndisponivel, carregar_preditor


def _gerar_seed():
    return secrets.randbelow(10 ** 9)


@dataclass
class Dependencias:
    """Fábricas e funções usadas pela CLI; os testes trocam por versões falsas."""
    criar_sorteador: Callable = SorteadorTeste
    carregar_preditor: Callable = carregar_preditor          # só aqui o TensorFlow é importado
    obter_kb: Callable = obter_kb
    gerar_seed: Callable = _gerar_seed
    diagnosticar: Callable = diagnosticar_amostra
    salvar_figuras: Callable = salvar_figuras
    pasta_base: Optional[Path] = None                        # None: rotina_demo/cli/
    raiz: Path = RAIZ_REPOSITORIO                            # só para mostrar caminhos relativos
    # (diagnostico, kb) -> Laudo. None: sem IA generativa (padrão, para os testes não usarem
    # a API). O ponto de entrada diagnostico_cli.py liga o laudo do Gemini.
    gerar_laudo: Optional[Callable] = None
    # (texto_usuario, sintomas) -> Interpretacao. None: sem conversa (padrão, para os testes
    # não usarem a API). O ponto de entrada liga a conversa junto com o laudo.
    conversar: Optional[Callable] = None


class Sessao:
    """Estado de UMA execução da CLI: cria sob demanda e reaproveita o que é caro."""

    def __init__(self, deps, saida):
        self.deps, self.saida = deps, saida
        self._sorteador = None
        self._kb = None
        self._preditor = None
        self._modelo_tentado = False

    def sorteador(self):
        if self._sorteador is None:
            self._sorteador = self.deps.criar_sorteador()
        return self._sorteador

    def kb(self):
        if self._kb is None:
            self._kb = self.deps.obter_kb()
        return self._kb

    def preditor(self):
        """O modelo (ou None, sem CNN). Só tenta carregar uma vez por sessão."""
        if not self._modelo_tentado:
            self._modelo_tentado = True
            self.saida(fmt.MSG_CARREGANDO_MODELO)
            try:
                self._preditor = self.deps.carregar_preditor()
                self.saida(fmt.MSG_MODELO_CARREGADO)
            except ModeloIndisponivel as erro:
                self.saida(fmt.formatar_aviso_modelo(str(erro)))
            except Exception as erro:                        # TensorFlow ausente ou quebrado
                self.saida(fmt.formatar_aviso_modelo(type(erro).__name__))
        return self._preditor


def _perguntar_outra(entrada, saida):
    while True:
        resposta = entrada(fmt.PROMPT_OUTRA).strip().lower()
        if resposta in ("s", "sim"):
            return True
        if resposta in ("n", "nao", "não"):
            return False
        saida(fmt.MSG_RESPOSTA_INVALIDA)


def _perguntar_s_n(entrada, saida, prompt):
    while True:
        resposta = entrada(prompt).strip().lower()
        if resposta in ("s", "sim"):
            return True
        if resposta in ("n", "nao", "não"):
            return False
        saida(fmt.MSG_RESPOSTA_INVALIDA)


def _conversar(sessao, entrada, saida, amostra, diag, kb, preditor, gravacao):
    """Conversa DEPOIS da revelação: o usuário responde em texto livre, o LLM traduz em
    sintomas, o usuário confirma e o SISTEMA ESPECIALISTA recalcula."""
    deps = sessao.deps
    if deps.conversar is None or diag.status != "ok" or diag.conclusao.conclusivo:
        return
    sintomas = sintomas_para_perguntar(diag, kb)
    if not sintomas:
        return
    pasta = gravacao.pasta if gravacao is not None and gravacao.ok else None
    saida("")
    saida(fmt.formatar_perguntas_conversa(sintomas, pasta, deps.raiz))
    texto = entrada(fmt.PROMPT_RESPOSTA)
    if not texto.strip():
        saida(fmt.MSG_CONVERSA_PULADA)
        return
    saida(fmt.MSG_INTERPRETANDO)
    interp = deps.conversar(texto, sintomas)
    saida(fmt.formatar_interpretacao(interp, kb))
    if interp.aviso:
        return
    if not interp.sintomas:
        saida(fmt.MSG_NADA_A_RECALCULAR)
        return
    if not _perguntar_s_n(entrada, saida, fmt.PROMPT_CONFIRMA):
        saida(fmt.MSG_NAO_RECALCULADO)
        return
    saida(fmt.MSG_RECALCULANDO)
    novo = deps.diagnosticar(amostra.segmentada, amostra.mascara, amostra.imagem.split,
                             preditor=preditor, kb=kb, sintomas_usuario=dict(interp.sintomas))
    saida("")
    saida(fmt.formatar_antes_depois(diag, novo, kb))
    if deps.gerar_laudo is not None and novo.status == "ok":
        saida("")
        saida(fmt.MSG_GERANDO_LAUDO)
        try:
            saida(fmt.formatar_laudo(deps.gerar_laudo(novo, kb)))
        except Exception as erro:                            # o laudo nunca derruba a análise
            saida(fmt.formatar_aviso_laudo(type(erro).__name__))


def _gravar(deps, amostra, diag):
    try:
        return deps.salvar_figuras(amostra.original, amostra.segmentada, amostra.mascara, diag,
                                   pasta_base=deps.pasta_base)
    except Exception as erro:                                # nunca impede a comparação
        return ResultadoGravacao(False, None, (), f"falha inesperada ({type(erro).__name__})")


def analisar_uma(sessao, entrada, saida):
    """Uma análise completa. Devolve True se o usuário quer analisar outra imagem.

    EOFError e KeyboardInterrupt (da entrada) propagam para executar(), que encerra.
    """
    deps = sessao.deps
    seed = deps.gerar_seed()

    try:
        amostra = sessao.sorteador().sortear(seed=seed)
    except SorteioInvalido:
        saida(fmt.MSG_SORTEIO_INVALIDO)                      # genérica: sem o texto da exceção
        return False
    except Exception as erro:
        saida(fmt.formatar_erro_inesperado(erro))
        return False
    saida(fmt.formatar_sorteio(amostra.imagem.split))

    try:
        kb = sessao.kb()
        preditor = sessao.preditor()
        saida(fmt.MSG_ANALISANDO)
        diag = deps.diagnosticar(amostra.segmentada, amostra.mascara, amostra.imagem.split,
                                 preditor=preditor, kb=kb)
        tela = fmt.formatar_diagnostico(diag, kb)
    except Exception as erro:
        saida(fmt.formatar_erro_inesperado(erro))
        return False
    saida("")
    saida(tela)

    if deps.gerar_laudo is not None and diag.status == "ok":
        saida("")
        saida(fmt.MSG_GERANDO_LAUDO)
        try:
            saida(fmt.formatar_laudo(deps.gerar_laudo(diag, kb)))
        except Exception as erro:                            # o laudo nunca derruba a análise
            saida(fmt.formatar_aviso_laudo(type(erro).__name__))

    # --- revelação: nada abaixo é mostrado antes do Enter ---
    saida("")
    saida(fmt.separador())
    entrada(fmt.PROMPT_REVELAR)

    comparacao = None
    if diag.status == "ok":
        try:
            comparacao = comparar_com_classe_real(diag, amostra.imagem)
        except Exception:
            comparacao = None
    gravacao = _gravar(deps, amostra, diag)
    saida("")
    saida(fmt.formatar_revelacao(amostra.imagem, diag, comparacao, gravacao, seed, kb, deps.raiz))
    saida(fmt.separador())

    try:
        _conversar(sessao, entrada, saida, amostra, diag, kb, preditor, gravacao)
    except (EOFError, KeyboardInterrupt):
        raise                                    # encerram a CLI (EOFError é uma Exception)
    except Exception as erro:                    # a conversa nunca derruba a análise
        saida(fmt.formatar_aviso_conversa(type(erro).__name__))
    return _perguntar_outra(entrada, saida)


def executar(entrada=input, saida=print, deps=None):
    """Menu principal. Devolve 0 ao sair (inclusive por Ctrl+C ou fim da entrada)."""
    sessao = Sessao(deps if deps is not None else Dependencias(), saida)
    try:
        while True:
            saida("")
            saida(fmt.formatar_menu())
            opcao = entrada(fmt.PROMPT_MENU).strip()
            if opcao == "0":
                saida(fmt.MSG_ATE_LOGO)
                return 0
            if opcao == "1":
                saida("")
                while analisar_uma(sessao, entrada, saida):
                    saida("")
            else:
                saida(fmt.MSG_OPCAO_INVALIDA)
    except (KeyboardInterrupt, EOFError):
        saida("")
        saida(fmt.MSG_ENCERRANDO)
        return 0

"""Pipeline de diagnóstico: da imagem segmentada ao resultado completo para a CLI.

    resultado = diagnosticar_amostra(segmentada, mascara, "test", preditor=preditor)
    ... a CLI mostra o diagnóstico ...
    comparacao = comparar_com_classe_real(resultado, imagem)   # só DEPOIS do diagnóstico

Duas funções separadas, de propósito:
    diagnosticar_amostra    recebe só arrays e a divisão. Não existe parâmetro para caminho,
                            nome de arquivo nem classe real, e ela não grava nenhum arquivo
                            (as figuras ficam para aplicacao/exportacao.py, chamado pela CLI).
    comparar_com_classe_real  é a única que toca a classe real, e só aceita um diagnóstico
                            já pronto.

Fluxo: avaliar_mascara -> medir_lesoes (só se a máscara for confiável) -> CNN (chamável
injetado) -> montar_fatos (contrato) -> diagnosticar -> explicar. Se a CNN faltar ou falhar
(sem preditor, ModeloIndisponivel, erro de execução ou saída inválida), o diagnóstico segue
SEM CNN, com aviso explícito: nunca se inventam probabilidades, e "inconclusivo" nunca é
tratado como saudável. Erros do contrato (ex.: sintoma do usuário inválido) viram
status "erro" com mensagem, sem levantar exceção.

A KB é injetada (kb=...); se for None, é carregada uma vez e guardada (obter_kb).
O CF do sistema especialista NÃO é probabilidade calibrada.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aplicacao.contrato import (
    ContratoInvalido, carregar_class_names, montar_cnn_prior, montar_fatos,
)
from ia.expert_system import carregar_kb, chave_canonica, diagnosticar, explicar, proxima_pergunta
from ia.inferencia import AVISO_PESOS_LOCAIS, ModeloIndisponivel
from processamento_imagens.medidas_lesao import MascaraInvalida, avaliar_mascara, medir_lesoes

AVISO_CF = ("O CF (fator de certeza) do sistema especialista não é uma probabilidade "
            "calibrada: serve para ordenar hipóteses, não para medir chance de acerto.")

_KB_PADRAO = None


def obter_kb():
    """KB padrão, carregada uma única vez por sessão."""
    global _KB_PADRAO
    if _KB_PADRAO is None:
        _KB_PADRAO = carregar_kb()
    return _KB_PADRAO


@dataclass(frozen=True)
class Conclusao:
    hipotese_id: Optional[str]    # None: nenhuma hipótese atingiu a evidência mínima
    nome: Optional[str]
    cf: Optional[float]
    conclusivo: bool
    texto: str


@dataclass(frozen=True)
class RegraDisparada:
    regra_id: str
    hipoteses: tuple
    grau: float
    cf: float                     # CF da regra na KB (o efetivo é cf * grau)
    porque: str


@dataclass(frozen=True, eq=False)
class ResultadoDiagnostico:
    """Tudo que a CLI mostra do diagnóstico. NÃO contém caminho nem classe real."""
    status: str                   # "ok" | "erro"
    mensagem_erro: Optional[str]
    divisao: str
    modo: str                     # "com_cnn" | "sem_cnn"
    top3_cnn: tuple               # ((nome_dataset, nome_legivel, p), ...); vazio sem CNN
    qualidade: object             # QualidadeMascara
    medidas: object               # MedidasLesao ou None
    avisos: tuple
    regras_disparadas: tuple
    pendencias: tuple
    conclusao: Optional[Conclusao]
    explicacao: str
    resultado: object             # Resultado do motor, ou None quando status == "erro"


@dataclass(frozen=True, eq=False)
class Comparacao:
    """Comparação com a classe real, criada só depois do diagnóstico."""
    caminho: Path = field(repr=False, compare=False)
    classe_real: str = field(repr=False, compare=False)
    classe_real_canonica: Optional[str] = field(repr=False, compare=False)
    hipotese_especialista: Optional[str] = field(repr=False, compare=False)
    cnn_top1_coincide: Optional[bool]          # None: sem CNN ou classe sem tradução
    especialista_coincide: Optional[bool]      # None: nenhuma hipótese acima do mínimo
    especialista_conclusivo: bool


def _erro(mensagem, divisao, qualidade, medidas, avisos):
    return ResultadoDiagnostico(
        status="erro", mensagem_erro=mensagem, divisao=divisao, modo="sem_cnn", top3_cnn=(),
        qualidade=qualidade, medidas=medidas, avisos=tuple(avisos), regras_disparadas=(),
        pendencias=(), conclusao=None, explicacao="", resultado=None)


def _probabilidades_da_cnn(preditor, segmentada, class_names):
    """(probabilidades, nomes, aviso). Qualquer falha devolve (None, None, aviso)."""
    if preditor is None:
        return None, None, None
    try:
        nomes = getattr(preditor, "class_names", None) or class_names or carregar_class_names()
        bruto = preditor(segmentada)
        montar_cnn_prior(bruto, nomes)                       # só valida (tamanho, NaN, soma...)
        return bruto, nomes, None
    except ModeloIndisponivel as erro:
        return None, None, f"CNN indisponível ({erro}): diagnóstico só com medidas e regras."
    except ContratoInvalido as erro:
        return None, None, f"CNN ignorada, saída inválida ({erro}): diagnóstico só com medidas e regras."
    except Exception as erro:  # erro de execução do TensorFlow ou do preditor
        return None, None, (f"CNN ignorada, falhou ao executar ({type(erro).__name__}: {erro}): "
                            "diagnóstico só com medidas e regras.")


def _top3(probabilidades, nomes, kb):
    ordem = sorted(range(len(nomes)), key=lambda i: -float(probabilidades[i]))[:3]
    saida = []
    for i in ordem:
        try:
            legivel = kb.doencas[chave_canonica(nomes[i])].nome
        except KeyError:
            legivel = nomes[i]
        saida.append((nomes[i], legivel, float(probabilidades[i])))
    return tuple(saida)


def _regras_disparadas(resultado, kb):
    por_id = {r.id: r for r in kb.regras}
    achadas = {}
    for h in resultado.hipoteses:
        for e in h.evidencias:
            if e.fonte == "regra":
                achadas.setdefault(e.regra_id, (e, set()))[1].add(h.id)
    return tuple(
        RegraDisparada(rid, tuple(sorted(hips)), e.grau, por_id[rid].cf, e.porque)
        for rid, (e, hips) in sorted(achadas.items()))


def _pendencias(resultado, kb):
    if not resultado.hipoteses:
        return ()
    linhas = []
    sugerida = proxima_pergunta(resultado, kb)
    if sugerida is not None:
        linhas.append(f"Pergunta sugerida: {sugerida.pergunta}")
    for sid in sorted(resultado.melhor.pendentes):
        if sugerida is None or sid != sugerida.id:
            linhas.append(f"{sid}: {kb.sintomas[sid].pergunta}")
    return tuple(linhas)


def _conclusao(resultado, kb):
    cfg = kb.config
    if not resultado.hipoteses or resultado.melhor.cf_final < cfg["limiar_exibicao"]:
        return Conclusao(None, None, None, False,
                         "INCONCLUSIVO: nenhuma hipótese atingiu a evidência mínima. "
                         "Recomenda-se avaliação por um agrônomo. " + AVISO_CF)
    melhor = resultado.melhor
    nome = kb.doencas[melhor.id].nome
    status = "conclusivo" if resultado.conclusivo else "INCONCLUSIVO"
    return Conclusao(melhor.id, nome, melhor.cf_final, resultado.conclusivo,
                     f"Hipótese principal: {nome} (CF {melhor.cf_final:.2f}) [{status}]. {AVISO_CF}")


def diagnosticar_amostra(segmentada, mascara, divisao, preditor=None, kb=None,
                         class_names=None, sintomas_usuario=None):
    """Diagnostica uma imagem já segmentada. Não grava arquivos e não recebe caminho nem classe."""
    kb = kb if kb is not None else obter_kb()

    qualidade = avaliar_mascara(mascara)
    medidas = None
    if qualidade.confiavel:
        try:
            medidas = medir_lesoes(segmentada, mascara)
        except MascaraInvalida:
            medidas = None
        except ValueError as erro:
            return _erro(f"não foi possível medir a imagem: {erro}", divisao, qualidade, None, ())

    probabilidades, nomes, aviso_cnn = _probabilidades_da_cnn(preditor, segmentada, class_names)

    try:
        fatos = montar_fatos(probabilidades=probabilidades, class_names=nomes, medidas=medidas,
                             qualidade=qualidade, sintomas_usuario=sintomas_usuario, kb=kb)
    except ContratoInvalido as erro:
        return _erro(f"dados inválidos para o sistema especialista: {erro}", divisao, qualidade,
                     medidas, [aviso_cnn] if aviso_cnn else [])

    resultado = diagnosticar(fatos.fatos, kb)

    avisos = list(fatos.avisos)
    if aviso_cnn:
        avisos.append(aviso_cnn)
    if probabilidades is not None:
        avisos.append(AVISO_PESOS_LOCAIS)
    if resultado.cultura is None:
        avisos.append("cultura desconhecida: o diagnóstico considera as 13 classes.")
    elif fatos.cultura_origem == "a_inferir_pela_cnn":
        avisos.append(f"cultura inferida pela CNN: {resultado.cultura}.")

    return ResultadoDiagnostico(
        status="ok", mensagem_erro=None, divisao=divisao,
        modo="com_cnn" if probabilidades is not None else "sem_cnn",
        top3_cnn=_top3(probabilidades, nomes, kb) if probabilidades is not None else (),
        qualidade=qualidade, medidas=medidas, avisos=tuple(avisos),
        regras_disparadas=_regras_disparadas(resultado, kb), pendencias=_pendencias(resultado, kb),
        conclusao=_conclusao(resultado, kb), explicacao=explicar(resultado, kb),
        resultado=resultado)


def comparar_com_classe_real(diagnostico, imagem):
    """Compara o diagnóstico (já pronto) com a classe real de `imagem` (caminho e classe_real)."""
    if not isinstance(diagnostico, ResultadoDiagnostico) or diagnostico.status != "ok":
        raise ValueError("a comparação exige um diagnóstico concluído (status 'ok')")
    try:
        canonica = chave_canonica(imagem.classe_real)
    except KeyError:
        canonica = None

    cnn = None
    if diagnostico.top3_cnn and canonica is not None:
        cnn = chave_canonica(diagnostico.top3_cnn[0][0]) == canonica
    hipotese = diagnostico.conclusao.hipotese_id
    especialista = None if hipotese is None or canonica is None else hipotese == canonica
    return Comparacao(
        caminho=imagem.caminho, classe_real=imagem.classe_real, classe_real_canonica=canonica,
        hipotese_especialista=hipotese, cnn_top1_coincide=cnn,
        especialista_coincide=especialista,
        especialista_conclusivo=diagnostico.conclusao.conclusivo)

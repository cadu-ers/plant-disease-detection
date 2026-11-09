"""
CLI de demonstração do diagnóstico de doenças em folhas de tomate e batata.

Sorteia uma imagem do split de TESTE (data_split/split_manifest.csv), mostra o diagnóstico
(CNN + medidas da folha + sistema especialista) SEM revelar o arquivo nem a classe real, e só
depois de um Enter mostra a comparação com a classe real.

Uso (na raiz do repositório, com o Python 3.12 do PATH):
    python src/diagnostico_cli.py              # com o laudo da IA generativa (Gemini)
    python src/diagnostico_cli.py --sem-llm    # só o sistema especialista, sem API
    python src/diagnostico_cli.py --seed N     # repete o sorteio da seed N (mostrada na revelação)
    python src/diagnostico_cli.py --sem-cache  # o laudo sempre chama a API (rodadas de avaliação)

Com --seed, TODAS as análises da sessão usam a mesma seed: serve para reproduzir um caso,
por exemplo para comparar versões do prompt do laudo. A mesma seed só repete a mesma imagem
com o mesmo split_manifest.csv e a mesma pasta data/ (o sorteio usa random.Random(seed)).

Laudo: depois do diagnóstico, o Gemini reescreve o resultado em linguagem simples (requer a
variável GEMINI_API_KEY; sem ela, ou sem internet, a CLI avisa e segue sem o laudo). Cada
laudo passa pelo guardrail e vira uma linha em rotina_demo/guardrail_log.jsonl. Sem
--sem-cache, uma resposta já recebida é reaproveitada do cache (rotina_demo/llm_cache/): o
log marca "do_cache" e essas linhas não devem entrar nas contagens da avaliação.

Menu: 1) Analisar imagem aleatória   0) Sair
Gera (em rotina_demo/cli/AAAAMMDD_HHMMSS_<hex>/, ignorada pelo Git, sem apagar nada):
    original.png, mascara.png, segmentada.png e painel.png da imagem analisada
(requer models/ gerado por src/run_ia.py; sem modelo a CLI roda só com medidas e regras)
"""

import os

# Antes de qualquer import pesado: o TensorFlow e o OpenCV leem estas variáveis ao ser importados.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Sem isto, cv2.imread de um arquivo que não abre escreve o CAMINHO no stderr (com a pasta da
# classe) antes da revelação. "ERROR" ainda mostra erros graves, mas não esse aviso.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import sys  # noqa: E402
from typing import NamedTuple, Optional  # noqa: E402


def _entrada(prompt=""):
    """input(); com a entrada redirecionada (pipe/arquivo), repete o texto lido na tela."""
    linha = input(prompt)
    if not sys.stdin.isatty():
        print(linha)
    return linha


USO = "Uso: python src/diagnostico_cli.py [--sem-llm] [--sem-cache] [--seed N]"


class Opcoes(NamedTuple):
    sem_llm: bool = False
    seed: Optional[int] = None
    sem_cache: bool = False


def ler_opcoes(argv):
    """argv -> Opcoes(sem_llm, seed, sem_cache). Aceita --sem-llm, --sem-cache, --seed N e
    --seed=N; o resto é recusado. --sem-cache com --sem-llm não tem efeito (não há LLM).

    ValueError com a mensagem para o usuário (opção desconhecida, seed ausente ou inválida).
    A seed não é mostrada antes da revelação: quem a digitou já a conhece, mas a tela segue
    o mesmo contrato do sorteio normal."""
    sem_llm, seed, sem_cache, desconhecidos = False, None, False, []
    argv = list(argv)
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--sem-llm":
            sem_llm = True
        elif arg == "--sem-cache":
            sem_cache = True
        elif arg == "--seed" or arg.startswith("--seed="):
            if arg == "--seed":
                if i + 1 >= len(argv):
                    raise ValueError("Falta o valor de --seed (um número inteiro, ex.: --seed 594525008).")
                i += 1
                valor = argv[i]
            else:
                valor = arg[len("--seed="):]
            try:
                seed = int(valor)
            except ValueError:
                raise ValueError(f"Seed inválida: {valor!r}. Use um número inteiro "
                                 "(ex.: --seed 594525008).") from None
            if seed < 0:
                raise ValueError(f"Seed inválida: {seed}. Use um número inteiro maior ou igual a 0.")
        else:
            desconhecidos.append(arg)
        i += 1
    if desconhecidos:
        raise ValueError(f"Opção desconhecida: {' '.join(desconhecidos)}.")
    return Opcoes(sem_llm, seed, sem_cache)


def principal(argv=()):
    # Nunca quebrar com UnicodeEncodeError num console antigo: troca o que não couber por "?".
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        opcoes = ler_opcoes(argv)
    except ValueError as erro:
        print(f"{erro}\n{USO}", file=sys.stderr)
        return 2
    try:
        from cli.menu import Dependencias, executar
        from ia.generativa.laudo import criar_gerador_de_laudo
    except ImportError as erro:
        print(f"Não foi possível importar '{erro.name or erro}'. Use o Python do projeto "
              "(3.12 do PATH, com tensorflow e pyyaml); nada será instalado automaticamente.",
              file=sys.stderr)
        return 1
    extras = {} if opcoes.seed is None else {"gerar_seed": lambda: opcoes.seed}
    gerar_laudo = (None if opcoes.sem_llm
                   else criar_gerador_de_laudo(usar_cache=not opcoes.sem_cache))
    deps = Dependencias(gerar_laudo=gerar_laudo, **extras)
    return executar(entrada=_entrada, saida=print, deps=deps)


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))

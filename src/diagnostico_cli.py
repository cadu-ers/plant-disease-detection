"""
CLI de demonstração do diagnóstico de doenças em folhas de tomate e batata.

Sorteia uma imagem do split de TESTE (data_split/split_manifest.csv), mostra o diagnóstico
(CNN + medidas da folha + sistema especialista) SEM revelar o arquivo nem a classe real, e só
depois de um Enter mostra a comparação com a classe real.

Uso (na raiz do repositório, com o Python 3.12 do PATH):
    python src/diagnostico_cli.py

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


def _entrada(prompt=""):
    """input(); com a entrada redirecionada (pipe/arquivo), repete o texto lido na tela."""
    linha = input(prompt)
    if not sys.stdin.isatty():
        print(linha)
    return linha


def principal():
    # Nunca quebrar com UnicodeEncodeError num console antigo: troca o que não couber por "?".
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        from cli.menu import executar
    except ImportError as erro:
        print(f"Não foi possível importar '{erro.name or erro}'. Use o Python do projeto "
              "(3.12 do PATH, com tensorflow e pyyaml); nada será instalado automaticamente.",
              file=sys.stderr)
        return 1
    return executar(entrada=_entrada, saida=print)


if __name__ == "__main__":
    sys.exit(principal())

"""
Roda o projeto inteiro: o bloco de Processamento de Imagens (run_pi.py) e
depois o bloco de Inteligência Artificial (run_ia.py). Se o bloco de PI
falhar, o de IA não roda.

Uso:
    python src/run_all.py           -> roda o que falta nos dois blocos
    python src/run_all.py --force   -> repassa --force para run_pi.py e run_ia.py
                                       (a CNN só é treinada de novo com run_ia.py --treinar)

Gera:
    tudo o que run_pi.py e run_ia.py geram
"""

import argparse
import sys

from orchestration import Step, go_to_project_root, run_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="repassa --force para run_pi.py e run_ia.py")
    args = parser.parse_args()
    go_to_project_root()

    extra = ["--force"] if args.force else []
    steps = [
        Step("bloco 1: Processamento de Imagens", "src/run_pi.py", extra),
        Step("bloco 2: Inteligência Artificial", "src/run_ia.py", extra),
    ]
    sys.exit(run_steps("Projeto completo", steps))


if __name__ == "__main__":
    main()

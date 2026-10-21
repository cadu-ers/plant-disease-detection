"""Linha de comando do sistema especialista.

Uso (na raiz do repositório, com src/ no PYTHONPATH: no Linux "export PYTHONPATH=src",
no PowerShell '$env:PYTHONPATH = "src"'):
    python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/late_blight_tomate.json
    python -m ia.expert_system.diagnose fatos.json --sem-cnn        # só regras (ablação)
    python -m ia.expert_system.diagnose fatos.json --interativo     # faz perguntas

O JSON de entrada segue o contrato descrito em engine.py.
"""

import argparse
import json

from ia.expert_system import (carregar_kb, diagnosticar, diagnostico_interativo,
                              explicar, proxima_pergunta)


def perguntar_no_terminal(sintoma):
    while True:
        resp = input(f"{sintoma.pergunta} [s/n/0-1/?=não sei] ").strip().lower()
        if resp in ("?", ""):
            return None
        if resp in ("s", "sim"):
            return 1.0
        if resp in ("n", "nao", "não"):
            return 0.0
        try:
            valor = float(resp.replace(",", "."))
            if 0.0 <= valor <= 1.0:
                return valor
        except ValueError:
            pass
        print("Responda s, n, um número entre 0 e 1, ou ? para 'não sei'.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fatos", help="arquivo JSON com os fatos")
    ap.add_argument("--kb", default=None, help="caminho de outra base de conhecimento")
    ap.add_argument("--sem-cnn", action="store_true", help="ignora o prior da CNN")
    ap.add_argument("--interativo", action="store_true", help="faz perguntas até concluir")
    args = ap.parse_args()

    with open(args.fatos, encoding="utf-8") as f:
        fatos = json.load(f)
    kb = carregar_kb(args.kb)
    usar_cnn = not args.sem_cnn

    if args.interativo:
        resultado, _ = diagnostico_interativo(fatos, kb, perguntar_no_terminal, usar_cnn=usar_cnn)
    else:
        resultado = diagnosticar(fatos, kb, usar_cnn=usar_cnn)

    print(explicar(resultado, kb))

    if not args.interativo and not resultado.conclusivo:
        sugestao = proxima_pergunta(resultado, kb)
        if sugestao:
            print(f"\nPergunta sugerida: {sugestao.pergunta}")


if __name__ == "__main__":
    main()

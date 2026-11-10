"""
Testa a conexão com o Gemini (chave, modelo, texto e JSON). Não usa o cache.

Uso (na raiz do repositório, com src/ no PYTHONPATH):
    Git Bash / Linux:  export PYTHONPATH=src
    PowerShell:        $env:PYTHONPATH = "src"
    python -m ia.generativa.testar_conexao
    python -m ia.generativa.testar_conexao --modelo gemini-3.8-flash

Requer a variável de ambiente GEMINI_API_KEY (criada em aistudio.google.com).
Testa só o modelo pedido (sem o modelo reserva), repetindo erros temporários.
Gasta ao menos 2 requisições da cota. Não grava nenhum arquivo.
"""

import argparse
import sys
import time

from ia.generativa.cliente import ClienteLLM, LLMIndisponivel, RespostaInvalida, ler_chave

SCHEMA_TESTE = {
    "type": "object",
    "properties": {
        "cultura": {"type": "string", "enum": ["tomate", "batata"]},
        "parte_da_planta": {"type": "string"},
    },
    "required": ["cultura", "parte_da_planta"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=None, help="outro modelo do Gemini (padrão: config/GEMINI_MODEL)")
    args = ap.parse_args()

    print("1/3 Chave da API:", "definida" if ler_chave() else "NÃO definida")
    if not ler_chave():
        print("    Crie a chave em aistudio.google.com e defina, por exemplo:")
        print('      Git Bash:   export GEMINI_API_KEY="sua-chave"')
        print('      PowerShell: $env:GEMINI_API_KEY = "sua-chave"')
        print('      Permanente (Windows): setx GEMINI_API_KEY "sua-chave"  e abra um terminal novo')
        return 1

    cliente = ClienteLLM(modelo=args.modelo, modelo_reserva=None, usar_cache=False)
    print(f"    Modelo: {cliente.modelo}")

    try:
        inicio = time.perf_counter()
        texto = cliente.gerar_texto("Responda apenas com a palavra: ok", temperatura=0.0)
        print(f"2/3 Texto: {texto!r} ({time.perf_counter() - inicio:.1f} s)")

        inicio = time.perf_counter()
        dados = cliente.gerar_json(
            "Na frase 'mancha marrom na folha do tomateiro', qual é a cultura e a parte da planta?",
            schema=SCHEMA_TESTE)
        print(f"3/3 JSON: {dados} ({time.perf_counter() - inicio:.1f} s)")
    except (LLMIndisponivel, RespostaInvalida) as erro:
        print(f"FALHOU: {erro}")
        return 1

    print("\nConexão OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

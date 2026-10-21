"""
Funções comuns dos orquestradores (run_pi.py, run_ia.py, run_all.py): cada
etapa roda em um subprocesso (sys.executable), com o plano mostrado no início,
parada na primeira etapa com erro (dizendo o comando para reproduzi-lo) e uma
tabela-resumo no fim (etapa | status | tempo).

Uso:
    from common.orchestration import Step, run_steps, go_to_project_root

Gera:
    nada (módulo de funções auxiliares)
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

# Raiz do projeto: os scripts usam caminhos relativos a ela (data/, reports/, ...)
# (este arquivo está em src/common/, então sobe 3 níveis)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Base dos imports absolutos (common.config, processamento_imagens.preprocessing, ...)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


@dataclass
class Step:
    """Uma etapa: um script com argumentos.

    script: caminho de um script da raiz de src/ (ex.: "src/run_pi.py") ou nome de
        módulo de uma subpasta, rodado com python -m (ex.: "dados.split_dataset").
    skip_reason: se não for None, a etapa é pulada e o motivo é mostrado.
    note: aviso mostrado antes de rodar (ex.: etapa demorada).
    """
    name: str
    script: str
    args: list = field(default_factory=list)
    skip_reason: str = None
    note: str = None

    def is_module(self):
        return not self.script.endswith(".py")

    def command(self):
        target = ["-m", self.script] if self.is_module() else [self.script]
        return [sys.executable, *target, *self.args]

    def display(self):
        target = ["-m", self.script] if self.is_module() else [self.script]
        return "python " + " ".join([*target, *self.args])


def go_to_project_root():
    """Muda o diretório atual para a raiz do projeto (os caminhos do config são relativos)."""
    os.chdir(PROJECT_ROOT)


def _child_env():
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"   # plt.show() dos scripts não abre janela nem trava a execução
    env["PYTHONUTF8"] = "1"     # acentos nas saídas dos scripts, também no Windows
    # src/ no PYTHONPATH para os módulos das subpastas (python -m dados.split_dataset)
    # acharem os imports absolutos; os.pathsep é ";" no Windows e ":" no Linux
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC_DIR, env.get("PYTHONPATH")) if p)
    return env


def _format_time(seconds):
    if seconds is None:
        return "-"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s" if minutes else f"{seconds:.1f} s"


def run_steps(title, steps):
    """Roda as etapas em ordem; para na primeira com erro. Retorna o código de saída (0/1)."""
    # Sem isso, com a saída redirecionada (log/pipe), as mensagens do orquestrador
    # apareceriam depois das saídas dos subprocessos
    sys.stdout.reconfigure(line_buffering=True)
    print(f"=== {title} ===")
    print("Etapas:")
    for i, step in enumerate(steps, start=1):
        plan = f"pular ({step.skip_reason})" if step.skip_reason else step.display()
        print(f"  {i}. {step.name:<28} {plan}")
    print()

    results = []   # (nome, status, tempo)
    failed = None
    for step in steps:
        if failed is not None:
            results.append((step.name, "não executado", None))
            continue
        if step.skip_reason:
            print(f"[pulado] {step.name}: {step.skip_reason}")
            results.append((step.name, "pulado", None))
            continue

        print(f"\n>>> {step.name}: {step.display()}")
        if step.note:
            print(f"    Aviso: {step.note}")
        t0 = time.time()
        returncode = subprocess.run(step.command(), cwd=PROJECT_ROOT, env=_child_env()).returncode
        elapsed = time.time() - t0
        if returncode == 0:
            results.append((step.name, "ok", elapsed))
        else:
            results.append((step.name, "erro", elapsed))
            failed = step

    width = max(len(name) for name, _, _ in results)
    print(f"\n=== Resumo: {title} ===")
    print(f"{'etapa':<{width}} | {'status':<13} | tempo")
    print(f"{'-' * width}-+-{'-' * 13}-+-{'-' * 12}")
    for name, status, elapsed in results:
        print(f"{name:<{width}} | {status:<13} | {_format_time(elapsed)}")

    if failed is not None:
        print(f"\nErro na etapa '{failed.name}'. Para reproduzir o erro, rode na raiz do projeto:")
        print(f"    {failed.display()}")
        if failed.is_module():
            print('    (com src/ no PYTHONPATH: no Linux "export PYTHONPATH=src", '
                  'no PowerShell \'$env:PYTHONPATH = "src"\')')
        return 1
    return 0

import os
import sys
from pathlib import Path

import pytest

# Permite os imports absolutos a partir de src/ ("import ia.expert_system")
# rodando pytest a partir da raiz do repositório (src/tests/ -> sobe 2 níveis = src/)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from common.config import CLI_DEMO_DIR  # noqa: E402  (depende do sys.path acima)

RAIZ_REPOSITORIO = Path(SRC_DIR).parent


def listar_pastas(base):
    """Nomes das pastas diretamente dentro de `base` (vazio se `base` não existe)."""
    base = Path(base)
    return frozenset(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else frozenset()


def pastas_novas(base, antes):
    """Pastas de `base` que não estavam no instantâneo `antes` (ordenadas).

    As que já existiam são permitidas (execuções reais da CLI); qualquer pasta nova criada
    durante os testes é um vazamento de gravação e deve falhar.
    """
    return sorted(listar_pastas(base) - frozenset(antes))


@pytest.fixture(scope="session", autouse=True)
def pastas_cli_iniciais():
    """Instantâneo de rotina_demo/cli ANTES da suíte; confere de novo DEPOIS que ela termina."""
    base = RAIZ_REPOSITORIO / CLI_DEMO_DIR
    antes = listar_pastas(base)
    yield antes
    novas = pastas_novas(base, antes)
    assert not novas, f"a suíte criou pastas em {CLI_DEMO_DIR}: {novas}"

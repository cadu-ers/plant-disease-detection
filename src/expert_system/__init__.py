"""Sistema especialista para diagnóstico de doenças em folhas de tomate e batata."""

from .busca import diagnostico_interativo, proxima_pergunta
from .classes import chave_canonica
from .engine import diagnosticar
from .explicacao import explicar
from .kb import carregar_kb

__all__ = ["carregar_kb", "diagnosticar", "explicar", "diagnostico_interativo",
           "proxima_pergunta", "chave_canonica"]

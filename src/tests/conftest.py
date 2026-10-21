import os
import sys

# Permite os imports absolutos a partir de src/ ("import ia.expert_system")
# rodando pytest a partir da raiz do repositório (src/tests/ -> sobe 2 níveis = src/)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

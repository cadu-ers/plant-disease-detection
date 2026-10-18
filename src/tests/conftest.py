import os
import sys

# Permite "import expert_system" rodando pytest a partir da raiz do repositório
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

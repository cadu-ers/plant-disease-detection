"""
Constantes compartilhadas do projeto: caminhos, tamanhos de imagem, seed e
limiares de segmentação. Todos os scripts de src/ importam daqui.

Uso:
    from config import DATA_DIR, SEG_IMG_SIZE, ...

Gera:
    nada (módulo só de configuração)
"""

import os

import numpy as np

# ---------- Pastas e arquivos ----------
DATA_DIR = "data"                    # dataset original: data/<NomeDaClasse>/*.jpg
PROCESSED_DIR = "data_processed"     # imagens segmentadas (entrada da CNN)
SPLIT_DIR = "data_split"             # divisão treino/validação/teste
MODELS_DIR = "models"                # modelos treinados
REPORTS_DIR = "reports"              # gráficos e relatórios
SEGMENTATION_REPORTS_DIR = os.path.join(REPORTS_DIR, "segmentation")  # comparações de segmentação
CNN_REPORTS_DIR = os.path.join(REPORTS_DIR, "cnn")                    # saídas do train_model.py
DEMO_DIR = "rotina_demo"             # saídas da rotina de demonstração

MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
# Ordem das classes usada no treino (índice de saída do modelo -> nome da classe)
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# ---------- Divisão treino/validação/teste ----------
# Proporções (treino, validação, teste), estratificadas por classe.
# O teste é fixo e NUNCA entra no treino nem na seleção de modelo; o
# classificador híbrido reutiliza exatamente esta divisão.
SPLIT_RATIOS = (0.70, 0.15, 0.15)
# Uma linha por imagem: class, filename, split, processed_path, original_path
SPLIT_MANIFEST = os.path.join(SPLIT_DIR, "split_manifest.csv")

# Extensões de arquivo consideradas imagem
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------- Tamanhos de imagem ----------
SEG_IMG_SIZE = (224, 224)     # tamanho usado na segmentação (data_processed/)
MODEL_IMG_SIZE = (160, 160)   # tamanho de entrada da MobileNetV2

# Seed fixa para amostragens e divisões reprodutíveis
SEED = 42

# ---------- Treino da CNN ----------
BATCH_SIZE = 32
INITIAL_EPOCHS = 8       # etapa 1: só a cabeça de classificação
FINE_TUNE_EPOCHS = 5     # etapa 2: fine-tuning
FINE_TUNE_LAYERS = 30    # últimas camadas da MobileNetV2 descongeladas no fine-tuning

# ---------- Segmentação (limiares em HSV do OpenCV: H 0–179, S e V 0–255) ----------
# Faixa de verde, usada para separar a folha do fundo.
# Ajuste esses valores se o fundo do seu dataset não for removido corretamente.
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])

# Faixa de tons de lesão (manchas marrons/amarelas).
# As lesões das doenças têm matiz abaixo do verde (H ~8–35) e ficavam fora da
# máscara de verde, virando "buracos" pretos.
# S >= 40 segura o fundo bege/cinza do PlantVillage (baixa saturação, mas matiz
# na faixa do marrom); com S menor o fundo passa e o contorno preenchido o engole.
# V >= 20 recupera as lesões escuras na borda da folha, que falhavam com V >= 30.
LOWER_LESION = np.array([8, 40, 20])
UPPER_LESION = np.array([35, 255, 255])

# Área mínima (fração da imagem) para aceitar o maior contorno como a folha.
MIN_LEAF_AREA_RATIO = 0.05

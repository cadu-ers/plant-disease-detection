"""
Constantes compartilhadas do projeto: caminhos, tamanhos de imagem, seed e
limiares de segmentação. Todos os scripts de src/ importam daqui.

Uso:
    from common.config import DATA_DIR, SEG_IMG_SIZE, ...

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
REPORTS_DIR = "reports"              # gráficos e relatórios, separados por bloco

# reports/pi/: bloco de Processamento de Imagens
PI_REPORTS_DIR = os.path.join(REPORTS_DIR, "pi")
DATA_REPORTS_DIR = os.path.join(PI_REPORTS_DIR, "dados")                 # data_overview.py, split_dataset.py
SPLIT_SUMMARY_PATH = os.path.join(DATA_REPORTS_DIR, "split_summary.csv")
SEGMENTATION_REPORTS_DIR = os.path.join(PI_REPORTS_DIR, "segmentacao")   # segmentation_report.py
SEGMENTATION_QUALITY_PATH = os.path.join(SEGMENTATION_REPORTS_DIR, "qualidade_segmentacao.csv")
THRESHOLD_STUDY_DIR = os.path.join(SEGMENTATION_REPORTS_DIR, "estudo_limiares")  # comparações --s/--v
FFT_REPORTS_DIR = os.path.join(PI_REPORTS_DIR, "fft")                    # fft_analysis.py
P1_REPORTS_DIR = os.path.join(PI_REPORTS_DIR, "p1_histograma")           # histogram_analysis.py (P1)

# reports/ia/: bloco de Inteligência Artificial
IA_REPORTS_DIR = os.path.join(REPORTS_DIR, "ia")
CNN_REPORTS_DIR = os.path.join(IA_REPORTS_DIR, "cnn")                    # train_model.py
HYBRID_REPORTS_DIR = os.path.join(IA_REPORTS_DIR, "hibrido")             # hybrid_classifier.py
HYBRID_CONTROL_REPORTS_DIR = os.path.join(HYBRID_REPORTS_DIR, "controle_refit_treino")
HISTORY_REPORTS_DIR = os.path.join(IA_REPORTS_DIR, "historico")          # rodadas antigas (registro)
DEMO_DIR = "rotina_demo"             # saídas das rotinas de demonstração (não versionado)
PI_DEMO_DIR = os.path.join(DEMO_DIR, "pi")                  # demo_pi.py
PI_DEMO_LAST_PATH = os.path.join(PI_DEMO_DIR, "ultima.json")  # último resultado do demo_pi (lido pelo demo_ia --ultima-pi)
IA_DEMO_DIR = os.path.join(DEMO_DIR, "ia")                  # demo_ia.py
CLI_DEMO_DIR = os.path.join(DEMO_DIR, "cli")                # figuras da CLI de diagnóstico (aplicacao/exportacao.py)
FEATURES_DIR = "features_cache"      # features extraídas (handcrafted + embeddings da CNN)

MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
# Ordem das classes usada no treino (índice de saída do modelo -> nome da classe)
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
# Melhor pipeline do classificador híbrido (CNN + features clássicas -> SVM)
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "svm_hybrid.joblib")

# Arquivos gerados por extract_features.py (na mesma ordem do SPLIT_MANIFEST)
HANDCRAFTED_PATH = os.path.join(FEATURES_DIR, "handcrafted.csv")
CNN_EMBEDDINGS_PATH = os.path.join(FEATURES_DIR, "cnn_embeddings.npz")
CNN_TEST_PRED_PATH = os.path.join(FEATURES_DIR, "cnn_test_pred.npy")
# Gerado por fft_analysis.py (descritores de textura no domínio da frequência)
FFT_FEATURES_PATH = os.path.join(FEATURES_DIR, "fft_features.csv")

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
INITIAL_EPOCHS = 10      # etapa 1: só a cabeça de classificação
FINE_TUNE_EPOCHS = 20    # etapa 2: fine-tuning (máximo; o EarlyStopping pode parar antes)
FINE_TUNE_LAYERS = 50    # últimas camadas da MobileNetV2 descongeladas no fine-tuning
                         # (as BatchNormalization dentro delas continuam congeladas)
HEAD_LR = 1e-3           # learning rate da etapa 1
FINE_TUNE_LR = 1e-4      # learning rate inicial da etapa 2 (ReduceLROnPlateau reduz)

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

# ---------- Medidas de lesão (sistema especialista) ----------
# Usadas por processamento_imagens/medidas_lesao.py. São proporções de pixels por
# FAIXA DE COR, não uma segmentação de lesão validada (não há máscaras de
# referência). Derivam das faixas de segmentação acima, para existir uma única
# fonte de verdade: o matiz H é dividido em três partes que não se sobrepõem.
#   marrom : H 8–24   (de LOWER_LESION até antes do verde), S/V mínimos da lesão
#   amarelo: H 25–35  (faixa em que as máscaras de verde e de lesão se sobrepõem),
#            S/V mínimos do verde
#   verde  : H 36–90  (depois da lesão até o fim do verde), S/V mínimos do verde
# Pixels fora das três (pálidos, S < 40, ou escuros) contam só na lesão total.
LOWER_BROWN = np.array([LOWER_LESION[0], LOWER_LESION[1], LOWER_LESION[2]])
UPPER_BROWN = np.array([LOWER_GREEN[0] - 1, 255, 255])
LOWER_YELLOW = np.array([LOWER_GREEN[0], LOWER_GREEN[1], LOWER_GREEN[2]])
UPPER_YELLOW = np.array([UPPER_LESION[0], 255, 255])
LOWER_HEALTHY_GREEN = np.array([UPPER_LESION[0] + 1, LOWER_GREEN[1], LOWER_GREEN[2]])
UPPER_HEALTHY_GREEN = np.array([UPPER_GREEN[0], 255, 255])

# Componente marrom com menos pixels que isto é tratado como ruído, não como lesão.
MIN_LESION_AREA_PX = 6
# Lesão "pequena": área até esta fração da área da folha (valor provisório).
SMALL_LESION_MAX_RATIO = 0.005
# Fração da imagem ocupada pela folha fora desta faixa torna a máscara suspeita e
# as medidas não são calculadas. Mesmos limites de SUSPECT_LOW/SUSPECT_HIGH em
# segmentation_report.py (copiados, não importados: aquele módulo fixa o backend
# do matplotlib ao ser importado).
SUSPECT_LEAF_FRACTION = (0.10, 0.95)

# Fração dos pixels das 4 bordas da imagem que estão dentro da máscara. Igual ou
# acima disto a máscara é suspeita (provável fundo bege/marrom vazado) e as medidas
# não são calculadas. PROVISÓRIO: escolhido em 65 imagens de tomate (e conferido em
# 300, também quase só tomate); batata não foi avaliada.
MASK_EDGE_CONTACT_MAX = 0.20

# ---------- IA generativa (src/ia/generativa/) ----------
# Modelo do Gemini usado pela aplicação. A variável de ambiente GEMINI_MODEL tem
# prioridade (útil se o Google renomear ou aposentar o modelo). Em 09/11/2026 o
# 3.5 Flash-Lite respondeu em ~1,7 s, enquanto o 3.8 Flash devolveu 503/504.
LLM_MODEL = "gemini-3.5-flash-lite"
# Modelo tentado quando o principal está fora do ar (503/504/429 mesmo depois das
# novas tentativas) ou não existe (404). None desliga a reserva. A variável de
# ambiente GEMINI_MODEL_RESERVA tem prioridade.
LLM_MODEL_RESERVA = "gemini-3.8-flash"
# Tempo máximo de espera por uma resposta da API, em segundos.
LLM_TIMEOUT_S = 60
# Erros temporários (429, 5xx, rede) são repetidos até este número de tentativas por
# modelo, esperando LLM_ESPERA_INICIAL_S, depois o dobro, e assim por diante.
LLM_TENTATIVAS = 3
LLM_ESPERA_INICIAL_S = 1.0
# Respostas já recebidas (não versionado, dentro de rotina_demo/): a mesma pergunta
# não gasta cota de novo e a demo funciona sem internet para entradas já vistas.
LLM_CACHE_DIR = os.path.join(DEMO_DIR, "llm_cache")
# Registro do guardrail do laudo (não versionado): uma linha JSON por laudo, com status
# (aprovado / corrigido / reprovado), tentativas e violações. Base da visão crítica.
GUARDRAIL_LOG = os.path.join(DEMO_DIR, "guardrail_log.jsonl")
# Registro da conversa (não versionado): uma linha JSON por resposta interpretada, com os
# sintomas aceitos e os descartados pelo guardrail da conversa. Nunca o texto do usuário.
CONVERSA_LOG = os.path.join(DEMO_DIR, "conversa_log.jsonl")

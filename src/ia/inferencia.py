"""Inferência da CNN para o pipeline de diagnóstico.

Recebe a imagem JÁ SEGMENTADA (a segmentação é do bloco de Processamento de Imagens)
e devolve as 13 probabilidades softmax, na ordem de models/class_names.json. Não
recebe caminho, nome de arquivo nem classe real: não há parâmetro para isso.

    preditor = carregar_preditor()                   # importa o TensorFlow aqui
    probs = preditor(segmentada_bgr)                 # float32 (13,)
    # a validação (tamanho, NaN, soma ~1) é do contrato: aplicacao/contrato.py

O pipeline só exige um chamável (segmentada_bgr, extensao) -> vetor de 13 valores,
então nos testes qualquer função com essa assinatura substitui o PreditorCNN.

O TensorFlow é importado só dentro de carregar_preditor e preparar_entrada: importar
este módulo é barato e não exige TensorFlow. Se o modelo não puder ser usado, levanta
ModeloIndisponivel (nunca sys.exit), para o chamador decidir (ex.: seguir sem CNN).

Pré-processamento igual ao do demo_ia.py e do treino: a imagem segmentada é codificada
como o preprocessing.py a grava em data_processed/ (cv2.imencode; extensão fixa
".jpg", pois todo o dataset é JPEG e a extensão do arquivo original não entra na
inferência), depois decodificada em RGB e redimensionada (bilinear) para 160x160, em
float 0-255. O preprocess_input da MobileNetV2 já está dentro do modelo.

Uma imagem na CLI é só uma demonstração, não uma avaliação da CNN: ver
NOTA_CNN_DEMONSTRACAO.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from common.config import CLASS_NAMES_PATH, MODEL_IMG_SIZE, MODEL_PATH

N_CLASSES = 13
EXTENSAO_PADRAO = ".jpg"
EXTENSOES_SUPORTADAS = (".jpg", ".jpeg", ".png")
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]

# Nota (não é aviso de problema): mostrada na revelação da CLI quando a CNN foi usada.
NOTA_CNN_DEMONSTRACAO = (
    "Nota sobre a CNN: os pesos foram treinados depois da divisão 70/15/15 e com a "
    "segmentação atual. Esta comparação usa UMA imagem e é só uma demonstração, não uma "
    "avaliação da CNN: a avaliação completa no conjunto de teste está nos relatórios do "
    "projeto (reports/ia/LEIAME.md)."
)


class ModeloIndisponivel(RuntimeError):
    """A CNN não pode ser usada (arquivo ausente ou inválido, ou TensorFlow indisponível)."""


def _caminho(caminho):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else RAIZ_REPOSITORIO / caminho


def _ler_class_names(caminho):
    if not caminho.is_file():
        raise ModeloIndisponivel(f"lista de classes não encontrada: {caminho}")
    try:
        with open(caminho, encoding="utf-8") as f:
            nomes = json.load(f)
    except (OSError, ValueError) as erro:
        raise ModeloIndisponivel(f"não foi possível ler {caminho}: {erro}") from None
    if (not isinstance(nomes, list) or len(nomes) != N_CLASSES
            or not all(isinstance(n, str) for n in nomes) or len(set(nomes)) != N_CLASSES):
        raise ModeloIndisponivel(f"{caminho} deve ter {N_CLASSES} nomes de classe distintos")
    return tuple(nomes)


def preparar_entrada(segmentada_bgr, extensao=EXTENSAO_PADRAO):
    """Imagem segmentada (BGR uint8) -> tensor (1, 160, 160, 3) float 0-255, como no treino."""
    if (not isinstance(segmentada_bgr, np.ndarray) or segmentada_bgr.dtype != np.uint8
            or segmentada_bgr.ndim != 3 or segmentada_bgr.shape[2] != 3
            or segmentada_bgr.size == 0):
        raise ValueError("a imagem deve ser um array uint8 BGR com formato (altura, largura, 3)")
    extensao = str(extensao).lower()
    if extensao not in EXTENSOES_SUPORTADAS:
        raise ValueError(f"extensão não suportada: {extensao!r} (use {EXTENSOES_SUPORTADAS})")

    import tensorflow as tf
    from ia.hibrido.extract_features import decode_and_resize

    ok, buffer = cv2.imencode(extensao, segmentada_bgr)
    if not ok:
        raise ValueError(f"não foi possível codificar a imagem como {extensao}")
    imagem = decode_and_resize(tf.constant(buffer.tobytes()))
    return tf.expand_dims(imagem, 0)


class PreditorCNN:
    """CNN carregada. Chamar com a imagem segmentada devolve as 13 probabilidades."""

    def __init__(self, modelo, class_names):
        self._modelo = modelo
        self.class_names = tuple(class_names)

    def __call__(self, segmentada_bgr, extensao=EXTENSAO_PADRAO):
        entrada = preparar_entrada(segmentada_bgr, extensao)
        saida = self._modelo(entrada, training=False)
        return np.asarray(saida.numpy()[0], dtype=np.float32)


def carregar_preditor(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH, verbose=False):
    """Carrega a CNN (só leitura). Levanta ModeloIndisponivel se não for possível."""
    model_path, class_names_path = _caminho(model_path), _caminho(class_names_path)
    if not model_path.is_file():
        raise ModeloIndisponivel(f"modelo da CNN não encontrado: {model_path}")
    class_names = _ler_class_names(class_names_path)

    if not verbose:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        import tensorflow as tf
    except ImportError as erro:
        raise ModeloIndisponivel(f"TensorFlow não está instalado: {erro}") from None
    if not verbose:
        tf.get_logger().setLevel("ERROR")

    try:
        modelo = tf.keras.models.load_model(model_path)
    except Exception as erro:  # arquivo corrompido, versão incompatível etc.
        raise ModeloIndisponivel(f"não foi possível carregar {model_path}: {erro}") from None

    forma_entrada, forma_saida = modelo.input_shape, modelo.output_shape
    if (not isinstance(forma_entrada, tuple) or tuple(forma_entrada[1:]) != (*MODEL_IMG_SIZE, 3)
            or not isinstance(forma_saida, tuple) or forma_saida[-1] != len(class_names)):
        raise ModeloIndisponivel(
            f"o modelo não bate com o esperado: entrada {forma_entrada}, saída {forma_saida} "
            f"(esperado (None, {MODEL_IMG_SIZE[0]}, {MODEL_IMG_SIZE[1]}, 3) e {len(class_names)} classes)")
    return PreditorCNN(modelo, class_names)

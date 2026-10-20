"""
Definição da arquitetura do modelo (data augmentation + MobileNetV2 + cabeça
de classificação), usada pelo train_model.py.

O modelo treinado é salvo inteiro em .keras (model.save) e carregado com
tf.keras.models.load_model no extract_features.py e no demo_ia.py, que não
precisam reconstruir a arquitetura. As camadas de augmentation ficam dentro do modelo e só atuam
no treino.

Uso:
    from model_utils import build_model
    model, base_model = build_model(num_classes)

Gera:
    nada (módulo só de definição do modelo)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from config import MODEL_IMG_SIZE


def build_model(num_classes, weights="imagenet"):
    base_model = MobileNetV2(
        input_shape=MODEL_IMG_SIZE + (3,),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    # Data augmentation: essas camadas só atuam no treino (na inferência passam direto)
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=MODEL_IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    return model, base_model

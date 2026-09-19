"""
Definição da arquitetura do modelo, compartilhada entre train_model.py e demo.py.

Mantida separada para permitir reconstruir a mesma arquitetura e carregar
apenas os pesos salvos (model.load_weights), contornando um problema de
compatibilidade do Keras ao desserializar o modelo completo salvo em .h5.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (160, 160)


def build_model(num_classes, weights="imagenet"):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    return model, base_model

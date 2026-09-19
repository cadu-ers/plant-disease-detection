"""
Treino do modelo de classificação de doenças em folhas via transfer learning.

Pipeline de IA:
    data_processed/ (imagens segmentadas)
        -> tf.data.Dataset (train/val split automático)
        -> MobileNetV2 pré-treinada (ImageNet), congelada
        -> camada de classificação customizada
        -> treino inicial (só a cabeça)
        -> fine-tuning (descongela últimas camadas da base)
        -> avaliação + salva o modelo

Uso:
    python src/train_model.py

Gera:
    models/plant_disease_model.h5
    reports/training_history.png
    reports/classification_report.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Configurações ----------
DATA_DIR = "data_processed"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
SEED = 42

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names

    # Otimização de I/O
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # congela a base pré-treinada

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def plot_history(history_initial, history_fine, path="reports/training_history.png"):
    acc = history_initial.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history_initial.history["val_accuracy"] + history_fine.history["val_accuracy"]

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="Treino")
    plt.plot(val_acc, label="Validação")
    plt.axvline(x=len(history_initial.history["accuracy"]) - 1,
                color="gray", linestyle="--", label="Início do fine-tuning")
    plt.title("Acurácia durante o treino")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    print(f"Gráfico salvo em: {path}")


def evaluate(model, val_ds, class_names):
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Relatório salvo em: reports/classification_report.txt")


def main():
    print("Carregando dataset...")
    train_ds, val_ds, class_names = load_datasets()
    print(f"Classes encontradas ({len(class_names)}): {class_names}")

    print("\nConstruindo modelo (MobileNetV2 + head customizada)...")
    model, base_model = build_model(num_classes=len(class_names))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    print("\nEtapa 1: treinando apenas a camada de classificação...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[early_stop],
    )

    print("\nEtapa 2: fine-tuning das últimas camadas da base...")
    base_model.trainable = True
    # Congela tudo exceto as últimas ~30 camadas
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[early_stop],
    )

    print("\nSalvando modelo...")
    model.save("models/plant_disease_model.h5")

    plot_history(history_initial, history_fine)

    print("\nAvaliando no conjunto de validação...")
    evaluate(model, val_ds, class_names)

    print("\nConcluído!")


if __name__ == "__main__":
    main()
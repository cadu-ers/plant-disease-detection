"""
Treino do modelo de classificação de doenças em folhas via transfer learning.

Pipeline de IA:
    data_split/{train,val,test} (imagens segmentadas, divisão fixa 70/15/15)
        -> tf.data.Dataset por split
        -> data augmentation (só no treino) + MobileNetV2 pré-treinada (ImageNet)
        -> etapa 1: treina só a cabeça de classificação (base congelada)
        -> etapa 2: fine-tuning das últimas FINE_TUNE_LAYERS camadas da base
           (BatchNormalization sempre congeladas)
        -> class_weight balanceado nas duas etapas; EarlyStopping + ReduceLROnPlateau
           pela val_loss; cada etapa termina com os pesos da época de menor val_loss
        -> avaliação final no TESTE (nunca usado no treino nem na seleção de modelo)

Uso:
    python src/train_model.py
    (requer data_split/ gerado por src/split_dataset.py)

Gera:
    models/plant_disease_model.keras
    models/class_names.json
    reports/ia/cnn/training_history.png
    reports/ia/cnn/classification_report.txt
    reports/ia/cnn/confusion_matrix.png
    reports/ia/cnn/metrics_cnn.json
    reports/ia/cnn/history.json     -> métricas por época das duas etapas (com learning rate)
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE, CLASS_NAMES_PATH, CNN_REPORTS_DIR, FINE_TUNE_EPOCHS, FINE_TUNE_LAYERS,
    FINE_TUNE_LR, HEAD_LR, INITIAL_EPOCHS, MODEL_IMG_SIZE, MODEL_PATH, MODELS_DIR, SEED,
    SPLIT_DIR,
)
from model_utils import build_model
from utils import list_images, plot_confusion_matrix

HISTORY_PATH = os.path.join(CNN_REPORTS_DIR, "training_history.png")
REPORT_PATH = os.path.join(CNN_REPORTS_DIR, "classification_report.txt")
CONFUSION_PATH = os.path.join(CNN_REPORTS_DIR, "confusion_matrix.png")
METRICS_PATH = os.path.join(CNN_REPORTS_DIR, "metrics_cnn.json")
HISTORY_JSON_PATH = os.path.join(CNN_REPORTS_DIR, "history.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CNN_REPORTS_DIR, exist_ok=True)


def load_split(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, split),
        shuffle=shuffle,
        seed=SEED,
        image_size=MODEL_IMG_SIZE,
        batch_size=BATCH_SIZE,
    )


def load_datasets():
    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)
    test_ds = load_split("test", shuffle=False)

    class_names = train_ds.class_names
    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError("As classes de train/val/test não coincidem. Rode split_dataset.py de novo.")

    # Otimização de I/O
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_weights(class_names):
    """class_weight "balanced" a partir dos rótulos do treino (contagem de arquivos por classe)."""
    counts = [len(list_images(os.path.join(SPLIT_DIR, "train", c))) for c in class_names]
    y_train = np.repeat(np.arange(len(class_names)), counts)
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_train)

    print("\nPesos por classe (class_weight balanced):")
    for name, n, w in zip(class_names, counts, weights):
        print(f"  {name:<45} {n:>5} imagens  peso {w:.3f}")
    return {i: float(w) for i, w in enumerate(weights)}


def stage_callbacks():
    """Callbacks de uma etapa; instâncias novas a cada etapa, para não herdar estado.

    No Keras 3 instalado, EarlyStopping(restore_best_weights=True) restaura os
    pesos da época de menor val_loss em on_train_end MESMO QUE o early stopping
    não dispare (a etapa chega ao número máximo de épocas). Por isso não é
    preciso ModelCheckpoint para garantir os melhores pesos ao fim da etapa.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1,
        ),
    ]


def configure_fine_tuning(base_model):
    """Destrava as últimas FINE_TUNE_LAYERS camadas da base, mantendo TODAS as
    BatchNormalization congeladas (senão elas atualizam as estatísticas com os
    batches do treino e o modelo desestabiliza no início do fine-tuning).

    Retorna (camadas treináveis, BN congeladas entre as destravadas, total de BN).
    """
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    bn_layers = [l for l in base_model.layers
                 if isinstance(l, tf.keras.layers.BatchNormalization)]
    bn_refrozen = sum(l.trainable for l in bn_layers)
    for layer in bn_layers:
        layer.trainable = False

    n_trainable = sum(l.trainable for l in base_model.layers)
    return n_trainable, bn_refrozen, len(bn_layers)


def save_history_json(history_initial, history_fine, path=HISTORY_JSON_PATH):
    """Salva as métricas por época das duas etapas (inclui learning_rate)."""
    stages = {}
    for name, history in [("head", history_initial), ("fine_tune", history_fine)]:
        metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        stages[name] = {
            "epochs": len(metrics["loss"]),
            "best_epoch": int(np.argmin(metrics["val_loss"])) + 1,  # pesos restaurados
            "metrics": metrics,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stages, f, indent=2)
    print(f"Histórico salvo em: {path}")


def plot_history(history_initial, history_fine, path=HISTORY_PATH):
    # Épocas realmente executadas na etapa 1 (pode ser < INITIAL_EPOCHS com early stopping)
    n_initial = len(history_initial.history["accuracy"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title in [(axes[0], "accuracy", "Acurácia"), (axes[1], "loss", "Loss")]:
        train = history_initial.history[metric] + history_fine.history[metric]
        val = history_initial.history[f"val_{metric}"] + history_fine.history[f"val_{metric}"]
        epochs = range(1, len(train) + 1)
        ax.plot(epochs, train, label="Treino")
        ax.plot(epochs, val, label="Validação")
        ax.axvline(x=n_initial + 0.5, color="gray", linestyle="--", label="Início do fine-tuning")
        ax.set_title(f"{title} durante o treino")
        ax.set_xlabel("Época")
        ax.set_ylabel(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Gráfico salvo em: {path}")


def evaluate(model, test_ds, class_names, train_time_seconds):
    """Avaliação final no conjunto de TESTE."""
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
    print(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Relatório salvo em: {REPORT_PATH}")

    plot_confusion_matrix(y_true, y_pred, class_names, path=CONFUSION_PATH)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "n_test": int(len(y_true)),
        "train_time_seconds": round(train_time_seconds, 1),
    }
    metrics = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in metrics.items()}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


def main():
    print("Carregando dataset...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print(f"Classes encontradas ({len(class_names)}): {class_names}")

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Ordem das classes salva em: {CLASS_NAMES_PATH}")

    class_weight = compute_weights(class_names)

    print("\nConstruindo modelo (augmentation + MobileNetV2 + head customizada)...")
    model, base_model = build_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    start = time.time()

    print("\nEtapa 1: treinando apenas a camada de classificação...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weight,
        callbacks=stage_callbacks(),
    )

    print("\nEtapa 2: fine-tuning das últimas camadas da base...")
    n_trainable, bn_refrozen, n_bn = configure_fine_tuning(base_model)
    print(f"Camadas treináveis na base: {n_trainable} de {len(base_model.layers)} "
          f"(últimas {FINE_TUNE_LAYERS}, sem as BatchNormalization)")
    print(f"BatchNormalization congeladas entre as destravadas: {bn_refrozen} "
          f"(total de BN na base, todas congeladas: {n_bn})")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=stage_callbacks(),
    )

    train_time = time.time() - start

    print("\nSalvando modelo...")
    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")

    plot_history(history_initial, history_fine)
    save_history_json(history_initial, history_fine)

    print("\nAvaliando no conjunto de TESTE...")
    evaluate(model, test_ds, class_names, train_time)

    minutes, seconds = divmod(int(train_time), 60)
    print(f"\nTempo total de treino: {minutes} min {seconds} s")
    print("Concluído!")


if __name__ == "__main__":
    main()

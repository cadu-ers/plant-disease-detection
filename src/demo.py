"""
Rotina de demonstração: pega uma imagem (aleatória ou informada), segmenta a
folha, roda a predição do modelo treinado e salva o resultado.

Uso:
    python src/demo.py                          -> imagem aleatória do conjunto de TESTE
    python src/demo.py caminho/da/imagem.jpg     -> imagem específica
    (requer models/ gerado por src/train_model.py)

Sem argumento, sorteia uma imagem com split == "test" do manifest de
src/split_dataset.py (imagens nunca vistas no treino) e segmenta a imagem
original dela. Se o manifest não existir, sorteia de data/ com um aviso.

Gera:
    rotina_demo/resultado_<timestamp>.png   -> imagem original + folha segmentada
    rotina_demo/resultado_<timestamp>.txt   -> classe predita, confiança e top 3
"""

import csv
import json
import os
import random
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    CLASS_NAMES_PATH, DATA_DIR, DEMO_DIR, MODEL_IMG_SIZE, MODEL_PATH, SPLIT_MANIFEST,
)
from preprocessing import segment_image
from utils import pick_random_image


def load_class_names():
    """Ordem das classes usada no treino (índice de saída do modelo -> nome)."""
    if not os.path.exists(CLASS_NAMES_PATH):
        sys.exit(f"Erro: {CLASS_NAMES_PATH} não encontrado. Rode train_model.py primeiro.")
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        return json.load(f)


def pick_test_image():
    """Sorteia uma imagem do conjunto de teste; retorna (caminho_original, classe)."""
    if not os.path.exists(SPLIT_MANIFEST):
        print(f"Aviso: {SPLIT_MANIFEST} não encontrado; sorteando de {DATA_DIR}/ "
              "(a imagem pode ter sido usada no treino).")
        return pick_random_image(DATA_DIR)
    with open(SPLIT_MANIFEST, encoding="utf-8", newline="") as f:
        test_rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
    row = random.choice(test_rows)
    return row["original_path"], row["class"]


def predict(model, segmented, class_names):
    img = cv2.resize(segmented, MODEL_IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(class_names[i], preds[i]) for i in top3_idx]

    return top3[0][0], top3[0][1], top3


def save_result(original, segmented, true_class, predicted_class, confidence, top3):
    os.makedirs(DEMO_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Folha segmentada")
    axes[1].axis("off")

    fig.suptitle(
        f"Real: {true_class}\nPredito: {predicted_class} ({confidence:.1%})",
        fontsize=12,
    )
    plt.tight_layout()

    img_path = os.path.join(DEMO_DIR, f"resultado_{timestamp}.png")
    plt.savefig(img_path)

    txt_path = os.path.join(DEMO_DIR, f"resultado_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classe real: {true_class}\n")
        f.write(f"Classe predita: {predicted_class}\n")
        f.write(f"Confiança: {confidence:.1%}\n\n")
        f.write("Top 3 predições:\n")
        for name, prob in top3:
            f.write(f"  {name}: {prob:.1%}\n")

    print(f"Resultado salvo em: {img_path}")
    print(f"Detalhes salvos em: {txt_path}")
    plt.show()


def main():
    class_names = load_class_names()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        true_class = "(imagem informada manualmente)"
    else:
        img_path, true_class = pick_test_image()

    print(f"Imagem: {img_path} (classe real: {true_class})")

    model = tf.keras.models.load_model(MODEL_PATH)

    result = segment_image(img_path)
    if result is None:
        print(f"Não foi possível ler a imagem: {img_path}")
        sys.exit(1)
    original, segmented, _ = result
    predicted_class, confidence, top3 = predict(model, segmented, class_names)

    print(f"Predição: {predicted_class} ({confidence:.1%})")
    save_result(original, segmented, true_class, predicted_class, confidence, top3)


if __name__ == "__main__":
    main()
"""
Pega uma imagem (aleatória ou informada), roda o pipeline de
processamento de imagem + predição do modelo, e salva o resultado
(imagem + texto) na pasta rotina_demo/.

Uso:
    python src/demo.py                          -> imagem aleatória de data/
    python src/demo.py caminho/da/imagem.jpg     -> imagem específica
"""

import sys
import random
import os
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Permite importar módulos de src/ ao rodar "python src/demo.py" a partir da raiz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import segment_image  # noqa: E402

DATA_DIR = "data"
MODEL_PATH = "models/plant_disease_model.keras"
OUTPUT_DIR = "rotina_demo"
IMG_SIZE = (160, 160)


def pick_random_image():
    classes = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))]
    class_name = random.choice(classes)
    class_dir = os.path.join(DATA_DIR, class_name)
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_name = random.choice(images)
    return os.path.join(class_dir, img_name), class_name


def predict(model, segmented, class_names):
    img = cv2.resize(segmented, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(class_names[i], preds[i]) for i in top3_idx]

    return top3[0][0], top3[0][1], top3


def save_result(original, segmented, true_class, predicted_class, confidence, top3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    img_path = os.path.join(OUTPUT_DIR, f"resultado_{timestamp}.png")
    plt.savefig(img_path)

    txt_path = os.path.join(OUTPUT_DIR, f"resultado_{timestamp}.txt")
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
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        true_class = "(imagem informada manualmente)"
    else:
        img_path, true_class = pick_random_image()

    print(f"Imagem: {img_path} (classe real: {true_class})")

    class_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

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
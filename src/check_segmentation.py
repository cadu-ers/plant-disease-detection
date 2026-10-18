"""
Compara a segmentação antiga (máscara só de verde) com a nova
(verde + lesão, contorno externo preenchido) em imagens de exemplo.

Pega 2 imagens aleatórias de 6 classes diferentes de data/ (seed fixa)
e salva uma grade em reports/segmentation_comparison.png com as colunas:
    Original | Máscara antiga (só verde) | Máscara nova | Folha segmentada (nova)

Uso:
    python src/check_segmentation.py
    python src/check_segmentation.py --s 40 --v 20 --out reports/segmentation_S40_V20.png
        -> testa outros limites mínimos de S e V da máscara de lesão

Gera:
    reports/segmentation_comparison.png   (ou o arquivo passado em --out)
"""

import argparse
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import (  # noqa: E402
    DATA_DIR, REPORTS_DIR, SEED, LOWER_GREEN, UPPER_GREEN, LOWER_LESION,
)
from preprocessing import segment_image, denoise, to_hsv  # noqa: E402
from utils import list_classes, list_images  # noqa: E402

OUTPUT_PATH = os.path.join(REPORTS_DIR, "segmentation_comparison.png")
N_CLASSES = 6
N_PER_CLASS = 2


def old_green_mask(original):
    """Máscara antiga (só verde), mantida aqui apenas para comparação."""
    img_hsv = to_hsv(denoise(original))
    mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def pick_samples(rng):
    classes = list_classes(DATA_DIR)
    chosen = rng.sample(classes, N_CLASSES)
    samples = []
    for class_name in chosen:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = list_images(class_dir)
        for fname in rng.sample(images, N_PER_CLASS):
            samples.append((class_name, os.path.join(class_dir, fname)))
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s", type=int, default=None,
                        help="S mínimo da máscara de lesão (default: valor de config.py)")
    parser.add_argument("--v", type=int, default=None,
                        help="V mínimo da máscara de lesão (default: valor de config.py)")
    parser.add_argument("--out", default=OUTPUT_PATH,
                        help=f"arquivo de saída (default: {OUTPUT_PATH})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Limites da máscara de lesão desta execução (default: config.LOWER_LESION)
    lower = LOWER_LESION.copy()
    if args.s is not None:
        lower[1] = args.s
    if args.v is not None:
        lower[2] = args.v
    print(f"LOWER_LESION = {lower.tolist()}")

    rng = random.Random(SEED)
    samples = pick_samples(rng)

    titles = ["Original", "Máscara antiga (só verde)", "Máscara nova",
              "Folha segmentada (nova)"]
    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))

    for row, (class_name, path) in enumerate(samples):
        result = segment_image(path, lower_lesion=lower)
        if result is None:
            print(f"Não foi possível ler: {path}")
            continue
        original, segmented, new_mask = result
        old_mask = old_green_mask(original)

        panels = [
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            old_mask,
            new_mask,
            cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB),
        ]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
        axes[row, 0].set_ylabel(class_name, fontsize=8, rotation=0,
                                ha="right", va="center")

    plt.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=100)
    print(f"Comparação salva em: {args.out}")


if __name__ == "__main__":
    main()

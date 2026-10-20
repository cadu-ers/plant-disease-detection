"""
Visão geral do dataset: uma imagem de exemplo por classe e um resumo de
quantidade, resolução, formato e modo de cor de TODAS as imagens de data/.

Uso:
    python src/data_overview.py

Gera (em reports/pi/dados/):
    exemplos_por_classe.png   -> 1 imagem original por classe (título: classe e nº de imagens)
    resumo_dados.csv          -> classe, nº de imagens, resoluções, extensões, formato real e modo de cor
"""

import os
import random
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from config import DATA_DIR, DATA_REPORTS_DIR, SEED  # noqa: E402
from utils import list_classes, list_images, slide_style  # noqa: E402


def describe(counter):
    """Counter -> "valor (n); valor (n)", do mais comum para o menos comum."""
    return "; ".join(f"{k} ({n})" for k, n in counter.most_common())


def wrap_class_name(name, width=26):
    """Quebra o nome da classe para caber no título, sem alterar o nome:
    as quebras só acontecem logo depois de um "_" (os "_" repetidos são mantidos)."""
    pieces = re.findall(r"[^_]*_+|[^_]+$", name)
    lines, current = [], ""
    for piece in pieces:
        if current and len(current + piece) > width:
            lines.append(current)
            current = piece
        else:
            current += piece
    lines.append(current)
    return "\n".join(lines)


def summarize():
    rows, all_sizes, ignored = [], Counter(), 0
    for class_name in list_classes(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        images = list_images(class_dir)
        ignored += len(os.listdir(class_dir)) - len(images)
        sizes, formats, real_formats, modes = Counter(), Counter(), Counter(), Counter()
        for fname in images:
            with Image.open(os.path.join(class_dir, fname)) as img:  # lê só o cabeçalho
                sizes[f"{img.width}x{img.height}"] += 1
                modes[img.mode] += 1
                real_formats[img.format] += 1   # formato lido do cabeçalho, não da extensão
            formats[os.path.splitext(fname)[1]] += 1
        all_sizes.update(sizes)
        rows.append({"classe": class_name, "n_imagens": len(images),
                     "resolucoes": describe(sizes), "extensoes": describe(formats),
                     "formato_real": describe(real_formats),
                     "modos_de_cor": describe(modes)})
    return pd.DataFrame(rows), all_sizes, ignored


def plot_examples(summary, path):
    rng = random.Random(SEED)
    n_cols = 5
    n_rows = -(-len(summary) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 4.8 * n_rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), summary.itertuples()):
        class_dir = os.path.join(DATA_DIR, row.classe)
        with Image.open(os.path.join(class_dir, rng.choice(list_images(class_dir)))) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{wrap_class_name(row.classe)}\n{row.n_imagens} imagens", fontsize=12)
    fig.suptitle(f"PlantVillage (tomate e batata): uma imagem original por classe — "
                 f"{summary['n_imagens'].sum()} imagens, {len(summary)} classes")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


def main():
    slide_style()
    os.makedirs(DATA_REPORTS_DIR, exist_ok=True)
    summary, all_sizes, ignored = summarize()
    csv_path = os.path.join(DATA_REPORTS_DIR, "resumo_dados.csv")
    summary.to_csv(csv_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nTotal: {summary['n_imagens'].sum()} imagens em {len(summary)} classes "
          f"({ignored} arquivo(s) que não são imagem ignorado(s))")
    if len(all_sizes) == 1:
        print(f"Todas as imagens têm a mesma resolução: {next(iter(all_sizes))}")
    else:
        print(f"As imagens NÃO têm todas a mesma resolução: {describe(all_sizes)}")
    print(f"Resumo salvo em: {csv_path}")
    plot_examples(summary, os.path.join(DATA_REPORTS_DIR, "exemplos_por_classe.png"))


if __name__ == "__main__":
    main()

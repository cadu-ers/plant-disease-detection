"""
Material de segmentação para a apresentação: as etapas da segmentação em
sequência e uma verificação objetiva da qualidade em TODAS as imagens.

As constantes (faixas HSV, área mínima) e as funções de leitura, blur, HSV e
segmentação são as reais do pipeline (config.py e preprocessing.py). Os passos
intermediários são reimplementados aqui só para poder mostrá-los; o script
confere que a máscara final reimplementada é idêntica à de segment_leaf.

Uso:
    python src/segmentation_report.py
    (requer data_split/split_manifest.csv e features_cache/handcrafted.csv)

Gera (em reports/segmentation/):
    etapas_segmentacao.png        -> original até segmentada, para 2 folhas com lesões grandes
    qualidade_segmentacao.csv     -> por imagem: fração de folha, fallback, buracos (antiga x nova)
    qualidade_segmentacao.png     -> fração de folha por classe + buracos antiga x nova
    casos_suspeitos.png           -> as 12 segmentações mais suspeitas
"""

import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from config import (  # noqa: E402
    HANDCRAFTED_PATH, LOWER_GREEN, LOWER_LESION, MIN_LEAF_AREA_RATIO, SEGMENTATION_REPORTS_DIR,
    SPLIT_MANIFEST, UPPER_GREEN, UPPER_LESION,
)
from preprocessing import denoise, load_and_resize, segment_leaf, to_hsv  # noqa: E402
from utils import short_name, slide_style  # noqa: E402

STEP_CLASSES = ["Potato___Late_blight", "Tomato_Early_blight"]
LESION_QUANTILE = 0.90          # lesão grande, mas não o caso mais extremo
SUSPECT_LOW, SUSPECT_HIGH = 0.10, 0.95
N_SUSPECTS_SHOWN = 12

COLOR_OLD, COLOR_NEW = "#eb6834", "#2a78d6"   # antiga = laranja, nova = azul
INK_SECONDARY, GRID_COLOR = "#52514e", "#e4e3df"


# ---------- Etapas (reimplementação para visualização) ----------
def segmentation_steps(path):
    """Todas as etapas intermediárias de segment_leaf, com as constantes reais."""
    original = load_and_resize(path)
    blurred = denoise(original)
    hsv = to_hsv(blurred)

    green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    lesion = cv2.inRange(hsv, LOWER_LESION, UPPER_LESION)
    union = cv2.bitwise_or(green, lesion)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(union, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    fallback = largest_area < MIN_LEAF_AREA_RATIO * morph.size
    if fallback:
        filled = morph
    else:
        filled = np.zeros_like(morph)
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)

    # Resultado REAL do pipeline, para conferir a reimplementação
    segmented, mask = segment_leaf(blurred, hsv)
    return {
        "original": original, "blurred": blurred, "hsv": hsv, "green": green,
        "lesion": lesion, "union": union, "morph": morph, "filled": filled,
        "segmented": segmented, "mask": mask, "fallback": fallback,
    }


def old_green_mask(hsv):
    """Máscara antiga: só verde + abertura e fechamento (sem contorno preenchido)."""
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def count_holes(mask):
    """Número de buracos internos (contornos internos na hierarquia de 2 níveis)."""
    _, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int((hierarchy[0][:, 3] != -1).sum())


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def figure_steps(examples, path):
    titles = ["original", "blur gaussiano 5x5", "canal H (matiz)", "máscara verde",
              "máscara de lesão", "união (verde OU lesão)", "abertura + fechamento",
              "contorno externo\npreenchido", "folha segmentada"]
    fig, axes = plt.subplots(len(examples), len(titles), figsize=(28, 4.2 * len(examples)))
    for row, (class_name, path_img) in enumerate(examples):
        s = segmentation_steps(path_img)
        assert np.array_equal(s["filled"], s["mask"]), "reimplementação diferente do pipeline"
        panels = [
            (rgb(s["original"]), {}), (rgb(s["blurred"]), {}),
            (s["hsv"][..., 0], {"cmap": "hsv", "vmin": 0, "vmax": 179}),
            (s["green"], {"cmap": "gray"}), (s["lesion"], {"cmap": "gray"}),
            (s["union"], {"cmap": "gray"}), (s["morph"], {"cmap": "gray"}),
            (s["filled"], {"cmap": "gray"}), (rgb(s["segmented"]), {}),
        ]
        for col, (img, kwargs) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=13)
        axes[row, 0].set_ylabel(short_name(class_name), fontsize=13)

    fig.suptitle("Etapas da segmentação: a máscara verde sozinha deixa as lesões de fora; "
                 "a união com a máscara de lesão e o contorno preenchido as recuperam")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


# ---------- Qualidade em todas as imagens ----------
def quality_one(path):
    s = segmentation_steps(path)
    if not np.array_equal(s["filled"], s["mask"]):
        raise AssertionError(f"reimplementação diferente do pipeline em {path}")
    return {
        "leaf_fraction": float((s["mask"] > 0).mean()),
        "fallback": bool(s["fallback"]),
        "holes_old": count_holes(old_green_mask(s["hsv"])),
        "holes_new": count_holes(s["mask"]),
    }


def figure_quality(quality, path):
    order = quality.groupby("class")["leaf_fraction"].median().sort_values().index
    n_fallback = int(quality["fallback"].sum())
    n_suspect = int(quality["suspect"].sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8.5), gridspec_kw={"width_ratios": [1.2, 1]})
    data = [quality.loc[quality["class"] == c, "leaf_fraction"] for c in order]
    box = ax1.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6,
                      flierprops={"markersize": 3, "alpha": 0.5})
    for patch in box["boxes"]:
        patch.set_facecolor(COLOR_NEW)
        patch.set_alpha(0.6)
    for median in box["medians"]:
        median.set_color("black")
    for x, label in [(SUSPECT_LOW, f"{SUSPECT_LOW:.0%}"), (SUSPECT_HIGH, f"{SUSPECT_HIGH:.0%}")]:
        ax1.axvline(x, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
        ax1.text(x, len(order) + 0.6, label, ha="center", color=INK_SECONDARY)
    ax1.set_yticks(range(1, len(order) + 1))
    ax1.set_yticklabels([short_name(c) for c in order])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("fração da imagem ocupada pela folha (máscara nova)")
    ax1.set_title("Fração de folha por classe\n(linhas tracejadas: limites de suspeita)")
    ax1.grid(True, axis="x", color=GRID_COLOR)

    means = quality.groupby("class")[["holes_old", "holes_new"]].mean().loc[order]
    y = np.arange(len(order))
    h = 0.38
    ax2.barh(y + h / 2, means["holes_old"], h, color=COLOR_OLD, label="antiga (só verde)")
    ax2.barh(y - h / 2, means["holes_new"], h, color=COLOR_NEW, label="nova (verde + lesão, preenchida)")
    ax2.set_yticks(y)
    ax2.set_yticklabels([short_name(c) for c in order])
    ax2.set_xlabel("nº médio de buracos internos na máscara, por imagem")
    ax2.set_title("Buracos na máscara: antiga x nova")
    ax2.legend(frameon=False, loc="lower right")
    n_new = int((quality["holes_new"] > 0).sum())
    ax2.text(0.98, 0.16, f"máscara nova: buracos em só {n_new} de {len(quality)} imagens\n"
             f"(média {quality['holes_new'].mean():.4f}; barras azuis ≈ 0)",
             transform=ax2.transAxes, ha="right", va="bottom", color=INK_SECONDARY)
    ax2.grid(True, axis="x", color=GRID_COLOR)

    fig.suptitle(f"Verificação objetiva da segmentação em {len(quality)} imagens: "
                 f"{n_fallback} fallback(s), {n_suspect} imagem(ns) suspeita(s) "
                 f"(folha < {SUSPECT_LOW:.0%} ou > {SUSPECT_HIGH:.0%})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


def figure_suspects(suspects, path):
    n_cols = 4
    n_rows = -(-len(suspects) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 5.6 * n_rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, row in zip(np.atleast_1d(axes).ravel(), suspects.itertuples()):
        s = segmentation_steps(row.original_path)
        overlay = rgb(s["original"]).copy()
        contours, _ = cv2.findContours(s["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 255), 2)
        ax.imshow(overlay)
        flags = []
        if row.fallback:
            flags.append("fallback")
        if row.suspect:
            flags.append("suspeita")
        ax.set_title(f"{short_name(row._1)}\nfolha = {row.leaf_fraction:.0%}"
                     + (f" ({', '.join(flags)})" if flags else ""), fontsize=12)
    fig.suptitle("Segmentações mais suspeitas (contorno da máscara em magenta): fallback com folha "
                 "esbranquiçada (pouca saturação)\ne fundo bege que passa na faixa de lesão "
                 "(a máscara vira a imagem inteira)")
    plt.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


def main():
    slide_style()
    os.makedirs(SEGMENTATION_REPORTS_DIR, exist_ok=True)
    manifest = pd.read_csv(SPLIT_MANIFEST)

    # f. Etapas: imagem de treino no percentil 90 da proporção de lesão
    hand = pd.read_csv(HANDCRAFTED_PATH, usecols=["class", "filename", "split", "lesion_ratio"])
    hand = hand.merge(manifest[["class", "filename", "original_path"]], on=["class", "filename"])
    examples = []
    for c in STEP_CLASSES:
        rows = hand[(hand["class"] == c) & (hand["split"] == "train")]
        target = rows["lesion_ratio"].quantile(LESION_QUANTILE)
        pick = rows.iloc[(rows["lesion_ratio"] - target).abs().argsort().iloc[0]]
        examples.append((c, pick["original_path"]))
        print(f"Etapas: {c} -> {pick['original_path']} (lesion_ratio = {pick['lesion_ratio']:.2f})")
    figure_steps(examples, os.path.join(SEGMENTATION_REPORTS_DIR, "etapas_segmentacao.png"))

    # g. Qualidade em todas as imagens
    t0 = time.time()
    results = Parallel(n_jobs=-1)(delayed(quality_one)(p) for p in manifest["original_path"])
    quality = pd.concat([manifest[["class", "filename", "split", "original_path"]],
                         pd.DataFrame(results)], axis=1)
    quality["suspect"] = ((quality["leaf_fraction"] < SUSPECT_LOW)
                          | (quality["leaf_fraction"] > SUSPECT_HIGH))
    print(f"Qualidade calculada em {len(quality)} imagens em {time.time() - t0:.1f} s "
          "(máscara reimplementada idêntica à do pipeline em todas)")
    csv_path = os.path.join(SEGMENTATION_REPORTS_DIR, "qualidade_segmentacao.csv")
    quality.drop(columns="original_path").to_csv(csv_path, index=False)
    print(f"Tabela salva em: {csv_path}")

    n_low = int((quality["leaf_fraction"] < SUSPECT_LOW).sum())
    n_high = int((quality["leaf_fraction"] > SUSPECT_HIGH).sum())
    print(f"\nFallbacks (maior contorno < {MIN_LEAF_AREA_RATIO:.0%} da imagem): {int(quality['fallback'].sum())}")
    print(f"Folha < {SUSPECT_LOW:.0%} da imagem: {n_low} | folha > {SUSPECT_HIGH:.0%}: {n_high} "
          f"| suspeitas: {n_low + n_high}")
    print(f"Buracos internos (média por imagem): antiga {quality['holes_old'].mean():.2f} "
          f"| nova {quality['holes_new'].mean():.2f}; imagens com algum buraco: "
          f"antiga {(quality['holes_old'] > 0).sum()} | nova {(quality['holes_new'] > 0).sum()}")
    print("\nFração de folha por classe (mediana, mín, máx):")
    print(quality.groupby("class")["leaf_fraction"].agg(["median", "min", "max"]).round(3).to_string())

    figure_quality(quality, os.path.join(SEGMENTATION_REPORTS_DIR, "qualidade_segmentacao.png"))

    # 12 mais suspeitas: fallbacks primeiro, depois a distância para fora da faixa [10%, 95%]
    quality["suspicion"] = (quality["fallback"].astype(float) * 10
                            + np.maximum(SUSPECT_LOW - quality["leaf_fraction"],
                                         quality["leaf_fraction"] - SUSPECT_HIGH))
    suspects = quality.sort_values("suspicion", ascending=False).head(N_SUSPECTS_SHOWN)
    suspects = suspects[["class", "leaf_fraction", "fallback", "suspect", "original_path"]]
    figure_suspects(suspects, os.path.join(SEGMENTATION_REPORTS_DIR, "casos_suspeitos.png"))


if __name__ == "__main__":
    main()

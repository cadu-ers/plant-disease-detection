"""
Figuras e verificação da segmentação (bloco de Processamento de Imagens):
as etapas da segmentação em sequência, uma verificação objetiva da qualidade
em TODAS as imagens e a comparação da máscara antiga (só verde) com a nova.

As constantes (faixas HSV, área mínima) e as funções de leitura, blur, HSV e
segmentação são as reais do pipeline (config.py e preprocessing.py). Os passos
intermediários são reimplementados aqui só para poder mostrá-los; o script
confere que a máscara final reimplementada é idêntica à de segment_leaf.
Não depende de nada do bloco de IA (features_cache/, models/).

Uso:
    python src/segmentation_report.py                -> etapas + qualidade + suspeitos
    python src/segmentation_report.py --comparacao   -> máscara antiga x nova em 12 imagens
    python src/segmentation_report.py --comparacao --s 40 --v 20 --out reports/pi/segmentacao/estudo_limiares/segmentation_S40_V20.png
        -> comparação com outros limites mínimos de S e V da máscara de lesão
    (requer data/ e data_split/split_manifest.csv)

Gera (em reports/pi/segmentacao/):
    etapas_segmentacao.png        -> original até segmentada, para 2 folhas com lesões grandes
    qualidade_segmentacao.csv     -> por imagem: fração de folha, proporção de lesão, fallback,
                                     buracos (antiga x nova)
    qualidade_segmentacao.png     -> fração de folha por classe + buracos antiga x nova
    casos_suspeitos.png           -> as 12 segmentações mais suspeitas
    segmentation_comparison.png   -> (--comparacao) Original | Máscara antiga | Máscara nova |
                                     Folha segmentada, 2 imagens de 6 classes (ou o arquivo de --out)
"""

import argparse
import os
import random
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from config import (  # noqa: E402
    DATA_DIR, LOWER_GREEN, LOWER_LESION, MIN_LEAF_AREA_RATIO, SEED, SEGMENTATION_QUALITY_PATH,
    SEGMENTATION_REPORTS_DIR,
    SPLIT_MANIFEST, UPPER_GREEN, UPPER_LESION,
)
from preprocessing import (  # noqa: E402
    denoise, load_and_resize, segment_image, segment_leaf, to_hsv,
)
from utils import list_classes, list_images, short_name, slide_style  # noqa: E402

COMPARISON_PATH = os.path.join(SEGMENTATION_REPORTS_DIR, "segmentation_comparison.png")
COMPARISON_N_CLASSES = 6
COMPARISON_N_PER_CLASS = 2

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
    """Máscara antiga: só verde + abertura e fechamento (sem contorno preenchido).

    Mantida aqui só para comparação (figura de comparação e contagem de buracos).
    """
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
def lesion_ratio_from_steps(s):
    """Fração dos pixels da folha (máscara final) na faixa LOWER_LESION/UPPER_LESION.

    Mesmo cálculo da feature lesion_ratio de features.py, feito aqui direto da
    máscara para o bloco de PI não depender de features_cache/.
    """
    leaf = s["mask"] > 0
    if not leaf.any():
        return 0.0
    return float(((s["lesion"] > 0) & leaf).sum() / leaf.sum())


def quality_one(path):
    s = segmentation_steps(path)
    if not np.array_equal(s["filled"], s["mask"]):
        raise AssertionError(f"reimplementação diferente do pipeline em {path}")
    return {
        "leaf_fraction": float((s["mask"] > 0).mean()),
        "lesion_ratio": lesion_ratio_from_steps(s),
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


# ---------- Comparação máscara antiga x nova (--comparacao) ----------
def pick_comparison_samples(rng):
    """2 imagens aleatórias de 6 classes de data/ (seed fixa)."""
    chosen = rng.sample(list_classes(DATA_DIR), COMPARISON_N_CLASSES)
    samples = []
    for class_name in chosen:
        class_dir = os.path.join(DATA_DIR, class_name)
        for fname in rng.sample(list_images(class_dir), COMPARISON_N_PER_CLASS):
            samples.append((class_name, os.path.join(class_dir, fname)))
    return samples


def run_comparison(s_min, v_min, out_path):
    """Original | Máscara antiga (só verde) | Máscara nova | Folha segmentada (nova)."""
    # Limites da máscara de lesão desta execução (default: config.LOWER_LESION)
    lower = LOWER_LESION.copy()
    if s_min is not None:
        lower[1] = s_min
    if v_min is not None:
        lower[2] = v_min
    print(f"LOWER_LESION = {lower.tolist()}")

    samples = pick_comparison_samples(random.Random(SEED))
    titles = ["Original", "Máscara antiga (só verde)", "Máscara nova",
              "Folha segmentada (nova)"]
    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))

    for row, (class_name, path) in enumerate(samples):
        result = segment_image(path, lower_lesion=lower)
        if result is None:
            print(f"Não foi possível ler: {path}")
            continue
        original, segmented, new_mask = result
        old_mask = old_green_mask(to_hsv(denoise(original)))

        panels = [rgb(original), old_mask, new_mask, rgb(segmented)]
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
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Comparação salva em: {out_path}")


# ---------- Main ----------
def lesion_ratio_one(path):
    """Proporção de lesão de uma imagem (ver lesion_ratio_from_steps)."""
    return lesion_ratio_from_steps(segmentation_steps(path))


def pick_step_examples(manifest):
    """Por classe, a imagem de treino no percentil 90 da proporção de lesão."""
    examples = []
    for c in STEP_CLASSES:
        rows = manifest[(manifest["class"] == c) & (manifest["split"] == "train")].copy()
        rows["lesion_ratio"] = Parallel(n_jobs=-1)(
            delayed(lesion_ratio_one)(p) for p in rows["original_path"])
        target = rows["lesion_ratio"].quantile(LESION_QUANTILE)
        pick = rows.iloc[(rows["lesion_ratio"] - target).abs().argsort().iloc[0]]
        examples.append((c, pick["original_path"]))
        print(f"Etapas: {c} -> {pick['original_path']} (lesion_ratio = {pick['lesion_ratio']:.2f})")
    return examples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparacao", action="store_true",
                        help="gera só a comparação máscara antiga x nova")
    parser.add_argument("--s", type=int, default=None,
                        help="(--comparacao) S mínimo da máscara de lesão (default: config.py)")
    parser.add_argument("--v", type=int, default=None,
                        help="(--comparacao) V mínimo da máscara de lesão (default: config.py)")
    parser.add_argument("--out", default=COMPARISON_PATH,
                        help=f"(--comparacao) arquivo de saída (default: {COMPARISON_PATH})")
    args = parser.parse_args()
    if not args.comparacao and (args.s is not None or args.v is not None
                                or args.out != COMPARISON_PATH):
        parser.error("--s, --v e --out só valem junto com --comparacao")
    return args


def main():
    args = parse_args()
    if args.comparacao:
        run_comparison(args.s, args.v, args.out)
        return

    slide_style()
    os.makedirs(SEGMENTATION_REPORTS_DIR, exist_ok=True)
    manifest = pd.read_csv(SPLIT_MANIFEST)

    # f. Etapas: imagem de treino no percentil 90 da proporção de lesão
    figure_steps(pick_step_examples(manifest),
                 os.path.join(SEGMENTATION_REPORTS_DIR, "etapas_segmentacao.png"))

    # g. Qualidade em todas as imagens
    t0 = time.time()
    results = Parallel(n_jobs=-1)(delayed(quality_one)(p) for p in manifest["original_path"])
    quality = pd.concat([manifest[["class", "filename", "split", "original_path"]],
                         pd.DataFrame(results)], axis=1)
    quality["suspect"] = ((quality["leaf_fraction"] < SUSPECT_LOW)
                          | (quality["leaf_fraction"] > SUSPECT_HIGH))
    print(f"Qualidade calculada em {len(quality)} imagens em {time.time() - t0:.1f} s "
          "(máscara reimplementada idêntica à do pipeline em todas)")
    csv_path = SEGMENTATION_QUALITY_PATH
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

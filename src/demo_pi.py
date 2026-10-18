"""
Demonstração do BLOCO DE PROCESSAMENTO DE IMAGENS em uma imagem: etapas da
segmentação, FFT e histogramas HSV com as faixas e limiares reais do pipeline,
mais as medidas da imagem comparadas com a mediana da classe.

Não usa nada do bloco de IA: não importa tensorflow e não lê models/ nem
features_cache/ (exceto o fft_features.csv gerado pelo fft_analysis.py, do
próprio bloco de PI, para a mediana da classe).

Uso:
    python src/demo_pi.py                              -> imagem aleatória do TESTE (manifest)
    python src/demo_pi.py --classe Tomato_Early_blight -> aleatória do teste dessa classe
    python src/demo_pi.py caminho/da/imagem.jpg        -> imagem específica
    python src/demo_pi.py --seed 7                     -> sorteio reprodutível
    python src/demo_pi.py --sem-janela                 -> só salva, não abre janela

Gera (em rotina_demo/pi/):
    resultado_<timestamp>.png   -> figura com 3 linhas: segmentação, FFT e histogramas HSV
    resultado_<timestamp>.json  -> caminho, classe real, split e medidas
    ultima.json                 -> cópia do último resultado (usada pelo demo_ia --ultima-pi)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib

# segmentation_report e fft_analysis fixam o backend "Agg" ao serem importados;
# guarda o backend interativo para poder abrir a janela no fim
INTERACTIVE_BACKEND = matplotlib.get_backend()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    FFT_FEATURES_PATH, LOWER_GREEN, LOWER_LESION, PI_DEMO_DIR, PI_DEMO_LAST_PATH,
    SEGMENTATION_QUALITY_PATH,
    SPLIT_MANIFEST, UPPER_GREEN, UPPER_LESION,
)
from fft_analysis import (  # noqa: E402
    HIGH_LOW_CUT, LOWPASS_D0, fft_descriptors, gaussian_lowpass, normalize01, prepare_image,
    reconstruct, spectrum,
)
from preprocessing import segment_image  # noqa: E402
from segmentation_report import lesion_ratio_from_steps, rgb, segmentation_steps  # noqa: E402
from utils import describe_image, load_manifest, pick_test_image, slide_style  # noqa: E402


COLOR_LEAF, COLOR_OTHER = "#2a78d6", "#52514e"          # folha = azul; imagem inteira/fundo = cinza
COLOR_GREEN, COLOR_LESION, COLOR_BOTH = "#1baf7a", "#eda100", "#e87ba4"
INK_SECONDARY = "#52514e"


# ---------- Escolha da imagem ----------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imagem", nargs="?", help="caminho de uma imagem específica")
    parser.add_argument("--classe", help="sorteia uma imagem do teste dessa classe")
    parser.add_argument("--seed", type=int, help="seed do sorteio (reprodutível)")
    parser.add_argument("--sem-janela", action="store_true", help="só salva, não abre janela")
    args = parser.parse_args()
    if args.imagem and args.classe:
        parser.error("use um caminho de imagem OU --classe, não os dois")
    return args


# ---------- Medidas ----------
def class_medians(class_name):
    """Medianas da classe a partir dos arquivos do bloco de PI (ou None se faltarem)."""
    medians = {"leaf_fraction": None, "lesion_ratio": None, "fft_high_low_ratio": None}
    if class_name is None:
        return medians
    if os.path.exists(SEGMENTATION_QUALITY_PATH):
        quality = pd.read_csv(SEGMENTATION_QUALITY_PATH)
        rows = quality[quality["class"] == class_name]
        if len(rows):
            medians["leaf_fraction"] = float(rows["leaf_fraction"].median())
            if "lesion_ratio" in rows:
                medians["lesion_ratio"] = float(rows["lesion_ratio"].median())
    if os.path.exists(FFT_FEATURES_PATH):
        fft = pd.read_csv(FFT_FEATURES_PATH, usecols=["class", "fft_high_low_ratio"])
        rows = fft[fft["class"] == class_name]
        if len(rows):
            medians["fft_high_low_ratio"] = float(rows["fft_high_low_ratio"].median())
    return medians


def compute_measures(steps, F):
    leaf = steps["mask"] > 0
    n_leaf = max(int(leaf.sum()), 1)
    green = (steps["green"] > 0) & leaf
    lesion = (steps["lesion"] > 0) & leaf
    return {
        "leaf_fraction": float(leaf.mean()),
        "lesion_ratio": lesion_ratio_from_steps(steps),
        "leaf_pct_green_range": float(100 * green.sum() / n_leaf),
        "leaf_pct_lesion_range": float(100 * lesion.sum() / n_leaf),
        "leaf_pct_both_ranges": float(100 * (green & lesion).sum() / n_leaf),
        "leaf_pct_no_range": float(100 * (leaf & ~green & ~lesion).sum() / n_leaf),
        "fft_high_low_ratio": float(fft_descriptors(F)["fft_high_low_ratio"]),
    }


def measures_text(m, medians, class_name):
    def with_median(value, key, fmt):
        median = medians.get(key)
        text = format(value, fmt)
        return text + (f"  (mediana da classe: {format(median, fmt)})" if median is not None else "")

    lines = [
        f"Fração da imagem ocupada pela folha: {with_median(m['leaf_fraction'], 'leaf_fraction', '.1%')}",
        f"Proporção de lesão na folha: {with_median(m['lesion_ratio'], 'lesion_ratio', '.1%')}",
        f"Pixels da folha na faixa verde: {m['leaf_pct_green_range']:.1f}%",
        f"Pixels da folha na faixa de lesão: {m['leaf_pct_lesion_range']:.1f}%",
        f"   (nas duas faixas, H 25–35: {m['leaf_pct_both_ranges']:.1f}%; "
        f"em nenhuma: {m['leaf_pct_no_range']:.1f}%)",
        f"Razão de energia alta/baixa freq. (corte {HIGH_LOW_CUT} de Nyquist): "
        f"{with_median(m['fft_high_low_ratio'], 'fft_high_low_ratio', '.3f')}",
    ]
    if class_name is None:
        lines.append("Classe real desconhecida: sem mediana para comparar.")
    elif all(v is None for v in medians.values()):
        lines.append("Arquivos do bloco de PI não encontrados: sem mediana da classe.")
    return lines


# ---------- Figura ----------
def _image(ax, img, title, **kwargs):
    ax.imshow(img, **kwargs)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _hist(values, bins, value_range):
    if values.size == 0:
        return None
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    return edges[:-1], hist / hist.sum()


def _plot_hist(ax, values, bins, value_range, label, color, style="-"):
    result = _hist(values, bins, value_range)
    if result is not None:
        x, y = result
        ax.step(x, y, where="post", color=color, linestyle=style, linewidth=1.8, label=label)


def row_segmentation(subfig, s):
    axes = subfig.subplots(1, 7)
    _image(axes[0], rgb(s["original"]), "original")
    _image(axes[1], rgb(s["blurred"]), "blur gaussiano 5x5")
    _image(axes[2], s["hsv"][..., 0], "canal H (matiz)", cmap="hsv", vmin=0, vmax=179)
    _image(axes[3], s["green"], "máscara verde", cmap="gray")
    _image(axes[4], s["lesion"], "máscara de lesão", cmap="gray")
    _image(axes[5], s["mask"], "máscara final\n(contorno preenchido)", cmap="gray")
    _image(axes[6], rgb(s["segmented"]), "folha segmentada")
    subfig.suptitle("Segmentação", fontweight="bold")


def row_fft(subfig, prepared, F, lines):
    axes = subfig.subplots(1, 5, gridspec_kw={"width_ratios": [1, 1, 1, 1, 2.3]})
    lowpass = gaussian_lowpass()
    _image(axes[0], prepared, "imagem preparada\n(fundo = média, sem DC, Hann)", cmap="gray")
    _image(axes[1], np.log1p(np.abs(F)), "log(1+|F|)\n(centro = baixas freq.)", cmap="magma")
    _image(axes[2], reconstruct(F * lowpass), f"passa-baixa\n(D0 = {LOWPASS_D0})", cmap="gray")
    _image(axes[3], normalize01(reconstruct(F * (1 - lowpass))), "passa-alta\n(normalizado)",
           cmap="gray")
    axes[4].axis("off")
    axes[4].text(0.0, 0.95, "Medidas da imagem", fontsize=14, fontweight="bold",
                 va="top", transform=axes[4].transAxes)
    axes[4].text(0.0, 0.80, "\n".join(lines), fontsize=12, va="top", linespacing=1.7,
                 transform=axes[4].transAxes)
    subfig.suptitle("FFT", fontweight="bold")


def row_histograms(subfig, s):
    axes = subfig.subplots(1, 3)
    hsv, leaf = s["hsv"], s["mask"] > 0
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # a. Canal H: imagem inteira x só a folha, com as faixas de matiz reais
    ax = axes[0]
    ax.axvspan(LOWER_LESION[0], LOWER_GREEN[0], color=COLOR_LESION, alpha=0.25,
               label=f"só lesão (H {LOWER_LESION[0]}–{LOWER_GREEN[0] - 1})")
    ax.axvspan(LOWER_GREEN[0], UPPER_LESION[0] + 1, color=COLOR_BOTH, alpha=0.3,
               label=f"verde E lesão (H {LOWER_GREEN[0]}–{UPPER_LESION[0]}):\nentra nas duas máscaras")
    ax.axvspan(UPPER_LESION[0] + 1, UPPER_GREEN[0] + 1, color=COLOR_GREEN, alpha=0.2,
               label=f"só verde (H {UPPER_LESION[0] + 1}–{UPPER_GREEN[0]})")
    _plot_hist(ax, h.ravel(), 180, (0, 180), "imagem inteira (com fundo)", COLOR_OTHER)
    _plot_hist(ax, h[leaf], 180, (0, 180), "só a folha (após segmentação)", COLOR_LEAF)
    ax.set_xlim(0, 179)
    ax.set_xlabel("H (matiz, 0–179)")
    ax.set_ylabel("fração dos pixels")
    ax.set_title("Canal H com as faixas de matiz\n(S e V também precisam passar nos limiares)")
    ax.legend(fontsize=11, loc="upper right", frameon=False)

    # b/c. Canais S e V: folha x fundo, com os limiares mínimos da máscara de lesão
    # A legenda fica do lado oposto ao da curva do fundo (S: fundo à esquerda; V: à direita)
    for ax, channel, name, threshold, label, legend_loc in [
        (axes[1], sat, "S (saturação, 0–255)", LOWER_LESION[1], "limiar S da lesão", "upper right"),
        (axes[2], val, "V (brilho, 0–255)", LOWER_LESION[2], "limiar V da lesão", "upper left"),
    ]:
        _plot_hist(ax, channel[leaf], 64, (0, 256), "folha", COLOR_LEAF)
        _plot_hist(ax, channel[~leaf], 64, (0, 256), "fundo (mask == 0)", COLOR_OTHER)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.4,
                   label=f"{label} ({name[0]} ≥ {threshold})")
        ax.set_xlim(0, 255)
        ax.set_xlabel(name)
        ax.set_ylabel("fração dos pixels")
        ax.set_title(f"Canal {name[0]}: folha x fundo")
        ax.legend(fontsize=11, loc=legend_loc, bbox_to_anchor=(0.1, 1.0) if legend_loc == "upper left"
                  else None, frameon=False)
    subfig.suptitle("Histogramas (espaço HSV da imagem suavizada)", fontweight="bold")


def build_figure(s, prepared, F, lines, title):
    fig = plt.figure(figsize=(30, 17), layout="constrained")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[1, 1.05, 1.15])
    row_segmentation(top, s)
    row_fft(middle, prepared, F, lines)
    row_histograms(bottom, s)
    fig.suptitle(title, fontsize=16)
    return fig


# ---------- Main ----------
def main():
    t0 = time.time()
    args = parse_args()
    slide_style()
    plt.switch_backend("Agg" if args.sem_janela else INTERACTIVE_BACKEND)

    manifest = load_manifest()
    try:
        if args.imagem:
            path = args.imagem
            class_name, split = describe_image(path, manifest)
        else:
            path, class_name, split = pick_test_image(manifest, args.classe, args.seed)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Imagem: {path}")
    print(f"Classe real: {class_name or 'desconhecida'} | split: {split or 'fora do manifest'}")

    s = segmentation_steps(path)
    real = segment_image(path)
    if real is None:
        sys.exit(f"Não foi possível ler a imagem: {path}")
    if not np.array_equal(s["mask"], real[2]) or not np.array_equal(s["filled"], real[2]):
        sys.exit("A máscara reimplementada difere da de segment_image (não deveria acontecer).")

    prepared = prepare_image(s["original"], s["mask"])
    F = spectrum(prepared)
    measures = compute_measures(s, F)
    medians = class_medians(class_name)
    lines = measures_text(measures, medians, class_name)
    print("\nMedidas:")
    for line in lines:
        print("  " + line)

    title = (f"Processamento de Imagens — classe real: {class_name or 'desconhecida'} — "
             f"{os.path.basename(path)}")
    fig = build_figure(s, prepared, F, lines, title)

    os.makedirs(PI_DEMO_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(PI_DEMO_DIR, f"resultado_{stamp}.png")
    fig.savefig(png_path)

    result = {
        "image_path": path.replace("\\", "/"),
        "class": class_name,
        "split": split,
        "measures": measures,
        "class_medians": medians,
        "figure": png_path.replace("\\", "/"),
    }
    json_path = os.path.join(PI_DEMO_DIR, f"resultado_{stamp}.json")
    for out in (json_path, PI_DEMO_LAST_PATH):
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nFigura salva em: {png_path}")
    print(f"Medidas salvas em: {json_path} (e {PI_DEMO_LAST_PATH})")
    print(f"Tempo: {time.time() - t0:.1f} s")
    if not args.sem_janela:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

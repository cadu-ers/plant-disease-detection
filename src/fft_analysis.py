"""
Análise das folhas no domínio da frequência (FFT 2D) e uso da FFT como
descritor de textura.

Preparação de cada imagem (prepare_image):
    1. Cinza da imagem ORIGINAL 224x224, antes do blur: o blur é um passa-baixa
       e apagaria justamente as altas frequências que queremos analisar.
    2. Pixels de fundo (mask == 0) preenchidos com a média de cinza da folha: o
       fundo preto da segmentação criaria um degrau forte na borda da folha, que
       aparece no espectro como um "+" enorme e esconde a textura da folha.
    3. Subtração da média: remove o termo DC (brilho médio). Sem isso, o DC,
       espalhado pela janela, domina as primeiras frequências.
    4. Multiplicação por uma janela de Hann 2D: a FFT trata a imagem como
       periódica, e as bordas que "não se encaixam" geram vazamento espectral
       (linhas horizontais/verticais no espectro). A janela leva a imagem
       suavemente a zero nas bordas.
    5. np.fft.fft2 + np.fft.fftshift (frequência zero no centro do espectro).

Uso:
    python src/fft_analysis.py
    (requer data_split/split_manifest.csv e features_cache/handcrafted.csv)

Gera (em reports/fft/):
    fft_saudavel_vs_doente.png       -> imagem, magnitude, fase, passa-baixa e passa-alta
    fft_magnitude_vs_fase.png        -> reconstrução só com magnitude x só com fase
    blur_espaco_vs_frequencia.png    -> teorema da convolução com o blur do pipeline
    espectro_radial_por_classe.png   -> espectro radial médio e razão alta/baixa por classe
    fft_svm.csv, fft_svm.png         -> SVM com FFT, clássicas e clássicas + FFT (teste)
e features_cache/fft_features.csv (descritores FFT de todas as imagens do manifest).
"""

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
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from config import (  # noqa: E402
    FEATURES_DIR, FFT_FEATURES_PATH, FFT_REPORTS_DIR, HANDCRAFTED_PATH, SEED, SEG_IMG_SIZE,
    SEGMENTATION_REPORTS_DIR, SPLIT_MANIFEST,
)
from preprocessing import segment_image  # noqa: E402
from utils import short_name, slide_style  # noqa: E402

N = SEG_IMG_SIZE[0]                 # imagens quadradas 224x224
NYQUIST_RADIUS = N // 2             # raio (em pixels do espectro) da frequência de Nyquist
LOWPASS_D0 = 15                     # raio de corte do passa-baixa gaussiano (figura a)
HIGH_LOW_CUT = 0.25                 # corte alta/baixa, em fração da frequência de Nyquist
N_BANDS = 8                         # faixas radiais (espaçadas em log) dos descritores
N_PER_CLASS_RADIAL = 150            # imagens de treino por classe no espectro radial
EXAMPLE_CLASSES = ["Tomato_healthy", "Tomato_Septoria_leaf_spot"]

# Cores: azul e laranja (slots 1 e 2 da paleta categórica validada) e tinta de texto
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
INK_PRIMARY, INK_SECONDARY, GRID_COLOR = "#0b0b0b", "#52514e", "#e4e3df"

# Distância de cada ponto do espectro (já com fftshift) ao centro, normalizada
# pela frequência de Nyquist (1.0 = Nyquist; os cantos passam de 1)
_yy, _xx = np.indices((N, N))
RADIUS_PX = np.hypot(_yy - N // 2, _xx - N // 2)
RADIUS = RADIUS_PX / NYQUIST_RADIUS
HANN_2D = np.outer(np.hanning(N), np.hanning(N))
BAND_EDGES = np.logspace(np.log10(1 / NYQUIST_RADIUS), 0, N_BANDS + 1)


# ---------- Preparação e espectro ----------
def prepare_image(original_bgr, mask):
    """Cinza, fundo = média da folha, sem DC, janela de Hann (ver docstring do módulo)."""
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    leaf = mask > 0
    filled = gray.copy()
    if leaf.any():
        filled[~leaf] = gray[leaf].mean()
    return (filled - filled.mean()) * HANN_2D


def spectrum(img):
    """FFT 2D com a frequência zero no centro."""
    return np.fft.fftshift(np.fft.fft2(img))


def load_prepared(path):
    original, _, mask = segment_image(path)
    return prepare_image(original, mask), original, mask


def fft_descriptors(F):
    """Descritores de textura a partir do espectro de potência (sem o DC, até Nyquist).

    band_XX: fração da energia em cada uma das 8 faixas radiais (espaçadas em log);
    high_low_ratio: energia acima / abaixo de 0.25 da frequência de Nyquist;
    centroid: frequência média ponderada pela energia (0 = DC, 1 = Nyquist);
    entropy: entropia espectral normalizada (0 = energia concentrada, 1 = espalhada).
    """
    power = np.abs(F) ** 2
    valid = (RADIUS > 0) & (RADIUS <= 1)
    p, r = power[valid], RADIUS[valid]
    total = p.sum()
    if total == 0:
        return dict.fromkeys(fft_feature_names(), 0.0)

    feats = {}
    for i in range(N_BANDS):
        upper = r <= BAND_EDGES[i + 1] if i == N_BANDS - 1 else r < BAND_EDGES[i + 1]
        feats[f"fft_band_{i}"] = p[(r >= BAND_EDGES[i]) & upper].sum() / total
    low = p[r <= HIGH_LOW_CUT].sum()
    feats["fft_high_low_ratio"] = p[r > HIGH_LOW_CUT].sum() / max(low, 1e-12)
    feats["fft_centroid"] = (p * r).sum() / total
    prob = p / total
    feats["fft_entropy"] = -(prob * np.log(prob + 1e-20)).sum() / np.log(prob.size)
    return feats


def fft_feature_names():
    return ([f"fft_band_{i}" for i in range(N_BANDS)]
            + ["fft_high_low_ratio", "fft_centroid", "fft_entropy"])


def features_one(path):
    prepared, _, _ = load_prepared(path)
    return fft_descriptors(spectrum(prepared))


def radial_profile_one(path):
    """Média radial do espectro de potência, anel a anel, até Nyquist."""
    prepared, _, _ = load_prepared(path)
    power = np.abs(spectrum(prepared)) ** 2
    rings = RADIUS_PX.astype(int).ravel()
    sums = np.bincount(rings, weights=power.ravel())
    counts = np.bincount(rings)
    return (sums / np.maximum(counts, 1))[:NYQUIST_RADIUS + 1]


# ---------- Figuras ----------
def _show(ax, img, title, cmap="gray", **kwargs):
    ax.imshow(img, cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def gaussian_lowpass():
    return np.exp(-(RADIUS_PX ** 2) / (2 * LOWPASS_D0 ** 2))


def reconstruct(F_shifted):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F_shifted)))


def normalize01(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)


def pick_median_leaf_examples(train):
    """Exemplos da figura (a): em cada classe, a imagem de treino com fração de folha
    mais próxima da mediana da classe (evita folhas mal segmentadas); seed 42 desempata."""
    quality = pd.read_csv(os.path.join(SEGMENTATION_REPORTS_DIR, "qualidade_segmentacao.csv"))
    rng = random.Random(SEED)
    examples = []
    for c in EXAMPLE_CLASSES:
        median = quality.loc[quality["class"] == c, "leaf_fraction"].median()
        rows = train[train["class"] == c].merge(
            quality[["class", "filename", "leaf_fraction"]], on=["class", "filename"])
        distance = (rows["leaf_fraction"] - median).abs()
        closest = rows[distance == distance.min()].sort_values("filename")
        pick = closest.iloc[rng.randrange(len(closest))]
        examples.append((c, pick["original_path"]))
        print(f"Figura (a): {c} -> {pick['original_path']} "
              f"(folha = {pick['leaf_fraction']:.3f}; mediana da classe = {median:.3f}; "
              f"{len(closest)} empatada(s))")
    return examples


def figure_healthy_vs_diseased(examples, path):
    """a. Imagem | magnitude | fase | passa-baixa | passa-alta, para saudável e doente."""
    lowpass = gaussian_lowpass()
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.6))
    for row, (class_name, img_path) in enumerate(examples):
        prepared, _, _ = load_prepared(img_path)
        F = spectrum(prepared)
        ax = axes[row]
        _show(ax[0], prepared, f"{short_name(class_name)}\nimagem preparada")
        _show(ax[1], np.log1p(np.abs(F)), "magnitude log(1+|F|)", cmap="magma")
        _show(ax[2], np.angle(F), "fase (parece ruído, mas guarda a estrutura;\n"
              "ver fft_magnitude_vs_fase.png)", cmap="twilight")
        _show(ax[3], reconstruct(F * lowpass), f"passa-baixa gaussiano\n(D0 = {LOWPASS_D0})")
        _show(ax[4], normalize01(reconstruct(F * (1 - lowpass))),
              "passa-alta (1 − passa-baixa)\nnormalizado")

        # Anotações no espectro de magnitude
        c = N // 2
        ax[1].annotate("baixas freq.", xy=(c, c), xytext=(c, c - 40), ha="center",
                       color="white", fontsize=12, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="white"))
        ax[1].text(8, N - 10, "altas freq.", color="white", fontsize=12, fontweight="bold")
        ax[1].text(N - 8, 18, "altas freq.", color="white", fontsize=12, fontweight="bold",
                   ha="right")

    fig.suptitle("Folha saudável x folha com lesões (Septoria) no domínio da frequência\n"
                 "centro do espectro = baixas frequências (forma geral, manchas grandes); "
                 "periferia = altas (bordas, nervuras, textura fina)")
    plt.tight_layout(h_pad=2.5)
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


def figure_magnitude_vs_phase(img_path, class_name, path):
    """b. Reconstrução só com magnitude (fase = 0) x só com fase (|F| = 1)."""
    prepared, _, _ = load_prepared(img_path)
    F = np.fft.fft2(prepared)
    only_magnitude = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F))))
    only_phase = np.real(np.fft.ifft2(np.exp(1j * np.angle(F))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    _show(axes[0], prepared, f"imagem preparada\n({short_name(class_name)})")
    _show(axes[1], normalize01(np.log1p(np.abs(only_magnitude))),
          "só magnitude (fase zerada)")
    _show(axes[2], normalize01(only_phase), "só fase (magnitude = 1)")
    fig.suptitle("A fase guarda a estrutura (posição das bordas e das lesões);\n"
                 "a magnitude guarda \"quanto\" de cada frequência existe, mas não onde")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


def figure_convolution_theorem(img_path, class_name, path):
    """c. Blur gaussiano 5x5 no espaço x multiplicação pela resposta em frequência."""
    original, _, _ = segment_image(img_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Espaço: o mesmo GaussianBlur 5x5 do preprocessing, com borda circular.
    # Esta versão do OpenCV não aceita BORDER_WRAP no GaussianBlur; o padding
    # circular de 2 pixels (metade do kernel) + recorte é equivalente.
    pad = 2
    wrapped = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_WRAP)
    blurred = cv2.GaussianBlur(wrapped, (5, 5), 0)[pad:-pad, pad:-pad]

    # Frequência: kernel 5x5 com zero-padding para 224x224, centrado na origem
    k1d = cv2.getGaussianKernel(5, 0)
    kernel = k1d @ k1d.T
    kernel_padded = np.zeros((N, N))
    kernel_padded[:5, :5] = kernel
    kernel_padded = np.roll(kernel_padded, (-pad, -pad), axis=(0, 1))
    H = np.fft.fft2(kernel_padded)
    filtered = np.real(np.fft.ifft2(np.fft.fft2(gray) * H))

    diff = np.abs(blurred - filtered)
    max_err = diff.max()

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.4))
    _show(axes[0], gray, f"original\n({short_name(class_name)})", vmin=0, vmax=255)
    _show(axes[1], blurred, "blur no espaço\n(GaussianBlur 5x5)", vmin=0, vmax=255)
    _show(axes[2], filtered, "filtragem na frequência\nIFFT(FFT(img) · H)", vmin=0, vmax=255)
    im = axes[3].imshow(diff, cmap="magma")
    axes[3].set_title(f"|diferença|\nmáx = {max_err:.1e}")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    im = axes[4].imshow(np.abs(np.fft.fftshift(H)), cmap="magma", vmin=0, vmax=1)
    axes[4].set_title("espectro do kernel |H|\n(passa-baixa)")
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    fig.suptitle("Teorema da convolução: convoluir no espaço = multiplicar na frequência "
                 "(blur do pipeline)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")
    return max_err


def figure_radial_spectrum(profiles, fft_train, path):
    """d. Espectro radial médio por classe + boxplot da razão alta/baixa."""
    classes = sorted(profiles)
    freq = np.arange(NYQUIST_RADIUS + 1) / NYQUIST_RADIUS
    palette = plt.get_cmap("tab20").colors
    colors = {c: palette[(2 * i) % 20 if i < 10 else (2 * (i - 10) + 1) % 20]
              for i, c in enumerate(classes)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 8.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    for c in classes:
        mean_log = np.log10(profiles[c] + 1e-12).mean(axis=0)
        healthy = c.endswith("healthy")
        ax1.plot(freq[1:], mean_log[1:], color=colors[c], linewidth=2.6 if healthy else 1.6,
                 linestyle="--" if healthy else "-", label=short_name(c))
    ax1.axvline(HIGH_LOW_CUT, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
    ax1.text(HIGH_LOW_CUT + 0.01, ax1.get_ylim()[1], "corte alta/baixa (0.25)",
             va="top", color=INK_SECONDARY)
    ax1.set_xlabel("frequência normalizada (1 = Nyquist)")
    ax1.set_ylabel("log10 da potência média (anel)")
    ax1.set_title(f"Espectro radial médio (até {N_PER_CLASS_RADIAL} imagens de treino por classe)\n"
                  "tracejado = classes saudáveis")
    ax1.grid(True, color=GRID_COLOR)
    ax1.legend(ncol=1, fontsize=11, loc="lower left", frameon=False)

    order = fft_train.groupby("class")["fft_high_low_ratio"].median().sort_values().index
    data = [fft_train.loc[fft_train["class"] == c, "fft_high_low_ratio"] for c in order]
    box = ax2.boxplot(data, orientation="horizontal", patch_artist=True, showfliers=False,
                      widths=0.6)
    for patch, c in zip(box["boxes"], order):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color(INK_PRIMARY)
    ax2.set_yticks(range(1, len(order) + 1))
    ax2.set_yticklabels([short_name(c) for c in order])
    ax2.set_xscale("log")
    ax2.set_xlabel("energia alta / baixa frequência (escala log)")
    ax2.set_title(f"Razão alta/baixa (corte em {HIGH_LOW_CUT} de Nyquist)\n"
                  f"todas as imagens de treino; sem outliers")
    ax2.grid(True, axis="x", color=GRID_COLOR)

    # Título gerado a partir dos dados (medianas), para não afirmar nada que o gráfico não mostre
    top = ", ".join(short_name(c) for c in order[::-1][:2])
    fig.suptitle(f"Conteúdo de frequência por classe: maior razão alta/baixa em {top}; "
                 f"menor em {short_name(order[0])}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


def figure_svm(results, path):
    """Barras de acurácia e F1-macro no teste, com zoom no eixo y."""
    x = np.arange(len(results))
    width, gap = 0.36, 0.02
    lowest = results[["accuracy", "f1_macro"]].to_numpy().min()
    y_low = np.floor((lowest - 0.03) * 20) / 20

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, metric, label, color in [(-(width + gap) / 2, "accuracy", "Acurácia", COLOR_A),
                                         ((width + gap) / 2, "f1_macro", "F1-macro", COLOR_B)]:
        bars = ax.bar(x + offset, results[metric], width, label=label, color=color, zorder=2)
        for bar, v in zip(bars, results[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=12, color=INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(results["modelo"])
    ax.set_ylim(y_low, 1.0)
    ax.set_ylabel(f"valor no teste (eixo começa em {y_low:.2f}, não em 0)")
    ax.set_title("SVM no teste: FFT x features clássicas x clássicas + FFT\n"
                 "(zoom: o eixo y NÃO começa em 0)")
    ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Figura salva em: {path}")


# ---------- Descritores e SVM ----------
def extract_all(manifest):
    paths = manifest["original_path"].tolist()
    t0 = time.time()
    feats = Parallel(n_jobs=-1)(delayed(features_one)(p) for p in paths)
    df = pd.concat([manifest[["class", "filename", "split"]].reset_index(drop=True),
                    pd.DataFrame(feats, columns=fft_feature_names())], axis=1)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(FFT_FEATURES_PATH, index=False)
    print(f"Descritores FFT de {len(df)} imagens em {time.time() - t0:.1f} s -> {FFT_FEATURES_PATH}")
    return df


def evaluate_svm(fft_df):
    """SVM (C=10, gamma=scale, balanced) treinado SÓ no treino e avaliado no TESTE."""
    hand = pd.read_csv(HANDCRAFTED_PATH)
    for col in ("class", "filename", "split"):
        if not (hand[col].to_numpy() == fft_df[col].to_numpy()).all():
            raise ValueError("handcrafted.csv e fft_features.csv fora de ordem.")
    hand_cols = [c for c in hand.columns if c not in ("class", "filename", "split")]
    data = pd.concat([hand, fft_df[fft_feature_names()]], axis=1)
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]

    configs = [("só FFT", fft_feature_names()),
               ("só clássicas", hand_cols),
               ("clássicas + FFT", hand_cols + fft_feature_names())]
    rows = []
    for name, cols in configs:
        model = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
        model.fit(train[cols], train["class"])
        pred = model.predict(test[cols])
        rows.append({"modelo": name, "n_features": len(cols),
                     "accuracy": accuracy_score(test["class"], pred),
                     "f1_macro": f1_score(test["class"], pred, average="macro")})
    return pd.DataFrame(rows)


def main():
    slide_style()
    os.makedirs(FFT_REPORTS_DIR, exist_ok=True)
    manifest = pd.read_csv(SPLIT_MANIFEST)
    train = manifest[manifest["split"] == "train"]

    # Exemplos fixos (seed 42) do treino
    rng = random.Random(SEED)
    examples = []
    for c in EXAMPLE_CLASSES:
        rows = train[train["class"] == c].sort_values("filename")
        examples.append((c, rng.choice(rows["original_path"].tolist())))
    print("Exemplos:", examples)

    figure_healthy_vs_diseased(pick_median_leaf_examples(train),
                               os.path.join(FFT_REPORTS_DIR, "fft_saudavel_vs_doente.png"))
    figure_magnitude_vs_phase(examples[1][1], examples[1][0],
                              os.path.join(FFT_REPORTS_DIR, "fft_magnitude_vs_fase.png"))
    max_err = figure_convolution_theorem(
        examples[1][1], examples[1][0], os.path.join(FFT_REPORTS_DIR, "blur_espaco_vs_frequencia.png"))
    print(f"Teorema da convolução: erro máximo absoluto = {max_err:.3e} (em níveis de cinza 0-255)")

    # Descritores FFT de todas as imagens
    fft_df = extract_all(manifest)

    # Espectro radial: até 150 imagens de treino por classe (Potato___healthy tem só 106)
    sample = pd.concat([g.sample(min(N_PER_CLASS_RADIAL, len(g)), random_state=SEED)
                        for _, g in train.groupby("class")])
    radial = Parallel(n_jobs=-1)(delayed(radial_profile_one)(p) for p in sample["original_path"])
    profiles = {c: np.array([r for r, cls in zip(radial, sample["class"]) if cls == c])
                for c in sorted(sample["class"].unique())}
    figure_radial_spectrum(profiles, fft_df[fft_df["split"] == "train"],
                           os.path.join(FFT_REPORTS_DIR, "espectro_radial_por_classe.png"))

    # SVM no teste
    results = evaluate_svm(fft_df)
    results.to_csv(os.path.join(FFT_REPORTS_DIR, "fft_svm.csv"), index=False)
    print("\n" + results.round(4).to_string(index=False))
    figure_svm(results, os.path.join(FFT_REPORTS_DIR, "fft_svm.png"))


if __name__ == "__main__":
    main()

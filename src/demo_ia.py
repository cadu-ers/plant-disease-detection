"""
Demonstração do BLOCO DE INTELIGÊNCIA ARTIFICIAL em uma imagem: segmenta a folha
(bloco de Processamento de Imagens), classifica com a CNN (top 3) e com o SVM
híbrido (embedding da CNN + features clássicas) e compara as duas predições
com a classe real.

Uso:
    python src/demo_ia.py                              -> imagem aleatória do TESTE (manifest)
    python src/demo_ia.py --classe Tomato_Early_blight -> aleatória do teste dessa classe
    python src/demo_ia.py caminho/da/imagem.jpg        -> imagem específica
    python src/demo_ia.py --ultima-pi                  -> mesma imagem da última execução do
                                                          demo_pi (rotina_demo/pi/ultima.json)
    python src/demo_ia.py --seed 7                     -> sorteio reprodutível
    python src/demo_ia.py --sem-janela                 -> só salva, não abre janela
    python src/demo_ia.py --verbose                    -> mostra os logs do TensorFlow
    (requer models/ gerado por src/run_ia.py; sem models/svm_hybrid.joblib roda só a CNN)

Gera (em rotina_demo/ia/):
    resultado_<timestamp>.png   -> original | segmentada | top 3 da CNN | quadro com as predições
    resultado_<timestamp>.txt   -> os mesmos dados em texto
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# --verbose mantém os logs informativos do TensorFlow (depuração). Tem que ser
# lido antes do "import tensorflow", que é quando essas variáveis são aplicadas.
VERBOSE = "--verbose" in sys.argv
if not VERBOSE:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

if not VERBOSE:
    tf.get_logger().setLevel("ERROR")

from config import (  # noqa: E402
    CLASS_NAMES_PATH, HYBRID_MODEL_PATH, IA_DEMO_DIR, MODEL_PATH, PI_DEMO_LAST_PATH,
)
from extract_features import (  # noqa: E402
    build_embedding_extractor, decode_and_resize, handcrafted_one,
)
from features import feature_names  # noqa: E402
from preprocessing import segment_image  # noqa: E402
from utils import (  # noqa: E402
    describe_image, load_manifest, pick_test_image, short_name, slide_style,
)

N_EMBEDDING = 1280                    # dimensão da GlobalAveragePooling2D da MobileNetV2
CNN_COLUMNS = [f"cnn_{i:04d}" for i in range(N_EMBEDDING)]   # mesmos nomes do hybrid_classifier.py

COLOR_BAR, COLOR_BAR_TOP = "#86b6ef", "#2a78d6"
INK_SECONDARY = "#52514e"


# ---------- Modelos ----------
def load_models():
    """CNN (extrator com embedding + softmax), nomes das classes e SVM híbrido (ou None)."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)):
        sys.exit(f"Modelo da CNN não encontrado ({MODEL_PATH} / {CLASS_NAMES_PATH}): "
                 "rode python src/run_ia.py")
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    extractor = build_embedding_extractor(tf.keras.models.load_model(MODEL_PATH))

    svm = None
    if os.path.exists(HYBRID_MODEL_PATH):
        svm = joblib.load(HYBRID_MODEL_PATH)
        expected = feature_names() + CNN_COLUMNS
        if list(svm.feature_names_in_) != expected:
            sys.exit(f"As colunas de {HYBRID_MODEL_PATH} não batem com features.py + "
                     "embedding da CNN; rode python src/run_ia.py --force")
    else:
        print(f"Aviso: {HYBRID_MODEL_PATH} não encontrado; rodando só a CNN "
              "(o SVM híbrido é gerado por python src/run_ia.py).")
    return extractor, class_names, svm


# ---------- Predição ----------
def cnn_input(segmented, original_path):
    """Entrada da CNN igual à do treino: a imagem segmentada é codificada como o
    preprocessing.py a grava em data_processed/ (cv2.imwrite, mesma extensão) e
    depois decodificada e redimensionada como no extract_features.py."""
    ext = os.path.splitext(original_path)[1].lower()
    ok, buffer = cv2.imencode(ext, segmented)
    if not ok:
        raise ValueError(f"Não foi possível codificar a imagem como {ext}")
    image = decode_and_resize(tf.constant(buffer.tobytes()))
    return tf.expand_dims(image, 0)


def predict_cnn(extractor, segmented, original_path):
    """Retorna (embedding 1280-d, probabilidades softmax) da imagem segmentada."""
    embedding, probs = extractor(cnn_input(segmented, original_path), training=False)
    return embedding.numpy()[0].astype(np.float32), probs.numpy()[0].astype(np.float32)


def svm_input(original_path, embedding):
    """Linha de entrada do SVM híbrido, com as MESMAS colunas e ordem do treino:
    as 147 features clássicas (features.py, como em extract_features.py) seguidas
    das 1280 dimensões do embedding (cnn_0000 ... cnn_1279)."""
    features, warning = handcrafted_one(original_path)
    if warning:
        print(f"Aviso: {warning} em {original_path}")
    row = {**features, **dict(zip(CNN_COLUMNS, embedding))}
    return pd.DataFrame([row])[feature_names() + CNN_COLUMNS]


def predict_svm(svm, original_path, embedding):
    return svm.predict(svm_input(original_path, embedding))[0]


# ---------- Saídas ----------
def summary_lines(true_class, top3, svm_pred):
    cnn_pred, cnn_conf = top3[0]
    lines = [f"Classe real: {true_class or 'desconhecida'}",
             f"CNN: {cnn_pred} ({cnn_conf:.1%})"]
    lines.append(f"SVM híbrido: {svm_pred}" if svm_pred is not None
                 else "SVM híbrido: indisponível (rode python src/run_ia.py)")
    if svm_pred is not None:
        lines.append("CNN e SVM híbrido " + ("concordam" if svm_pred == cnn_pred else "discordam"))
    if true_class is not None:
        lines.append(f"CNN {'acertou' if cnn_pred == true_class else 'errou'}")
        if svm_pred is not None:
            lines.append(f"SVM híbrido {'acertou' if svm_pred == true_class else 'errou'}")
    return lines


def build_figure(original, segmented, top3, lines, title):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.3, 1.3]})
    for ax, img, name in [(axes[0], original, "original"), (axes[1], segmented, "folha segmentada")]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[2]
    names = [short_name(c) for c, _ in top3][::-1]
    probs = [p for _, p in top3][::-1]
    bars = ax.barh(names, probs, color=[COLOR_BAR, COLOR_BAR, COLOR_BAR_TOP], height=0.6)
    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.02, 0.98), bar.get_y() + bar.get_height() / 2, f"{p:.1%}",
                va="center", ha="left" if p < 0.85 else "right",
                color="black" if p < 0.85 else "white", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probabilidade (softmax)")
    ax.set_title("CNN: top 3")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    axes[3].axis("off")
    axes[3].text(0.0, 0.95, "Predições", fontsize=14, fontweight="bold", va="top",
                 transform=axes[3].transAxes)
    axes[3].text(0.0, 0.80, "\n".join(lines), fontsize=13, va="top", linespacing=1.8,
                 transform=axes[3].transAxes)
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig


# ---------- Main ----------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imagem", nargs="?", help="caminho de uma imagem específica")
    parser.add_argument("--classe", help="sorteia uma imagem do teste dessa classe")
    parser.add_argument("--ultima-pi", action="store_true",
                        help="usa a imagem da última execução do demo_pi.py")
    parser.add_argument("--seed", type=int, help="seed do sorteio (reprodutível)")
    parser.add_argument("--sem-janela", action="store_true", help="só salva, não abre janela")
    parser.add_argument("--verbose", action="store_true", help="mostra os logs do TensorFlow")
    args = parser.parse_args()
    if sum([bool(args.imagem), bool(args.classe), args.ultima_pi]) > 1:
        parser.error("use só um entre: caminho de imagem, --classe e --ultima-pi")
    return args


def choose_image(args):
    """Retorna (caminho, classe real ou None, split ou None)."""
    if args.ultima_pi:
        if not os.path.exists(PI_DEMO_LAST_PATH):
            sys.exit(f"{PI_DEMO_LAST_PATH} não encontrado: rode antes python src/demo_pi.py")
        with open(PI_DEMO_LAST_PATH, encoding="utf-8") as f:
            last = json.load(f)
        return last["image_path"], last.get("class"), last.get("split")
    manifest = load_manifest()
    try:
        if args.imagem:
            return (args.imagem, *describe_image(args.imagem, manifest))
        return pick_test_image(manifest, args.classe, args.seed)
    except ValueError as e:
        sys.exit(str(e))


def main():
    t0 = time.time()
    args = parse_args()
    slide_style()
    if args.sem_janela:
        plt.switch_backend("Agg")

    path, true_class, split = choose_image(args)
    print(f"Imagem: {path}")
    print(f"Classe real: {true_class or 'desconhecida'} | split: {split or 'fora do manifest'}")

    extractor, class_names, svm = load_models()

    # 1. Segmentação (bloco de Processamento de Imagens)
    result = segment_image(path)
    if result is None:
        sys.exit(f"Não foi possível ler a imagem: {path}")
    original, segmented, _ = result

    # 2. CNN: top 3
    embedding, probs = predict_cnn(extractor, segmented, path)
    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], float(probs[i])) for i in top_idx]

    # 3. SVM híbrido: features clássicas + embedding da CNN
    svm_pred = predict_svm(svm, path, embedding) if svm is not None else None

    # TODO (sistema especialista): quando existir o extrator de medidas de lesão
    # (tamanho, cor, halo, aspecto encharcado etc. a partir da máscara de lesão),
    # montar aqui os fatos da folha e chamar o motor de src/expert_system/
    # (ver src/expert_system/diagnose.py), passando as probabilidades da CNN como
    # evidência (campo cnn_prior), e acrescentar o diagnóstico e a explicação ao
    # quadro de texto e ao .txt.

    lines = summary_lines(true_class, top3, svm_pred)
    print("\nCNN top 3:")
    for name, p in top3:
        print(f"  {name}: {p:.1%}")
    print("\n" + "\n".join(lines))

    os.makedirs(IA_DEMO_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(IA_DEMO_DIR, f"resultado_{stamp}.png")
    txt_path = os.path.join(IA_DEMO_DIR, f"resultado_{stamp}.txt")
    title = (f"Inteligência Artificial — classe real: {true_class or 'desconhecida'} — "
             f"{os.path.basename(path)}")
    fig = build_figure(original, segmented, top3, lines, title)
    fig.savefig(png_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Imagem: {path}\nSplit: {split or 'fora do manifest'}\n")
        f.write("\n".join(lines) + "\n\nCNN top 3:\n")
        for name, p in top3:
            f.write(f"  {name}: {p:.1%}\n")

    print(f"\nFigura salva em: {png_path}")
    print(f"Detalhes salvos em: {txt_path}")
    print(f"Tempo: {time.time() - t0:.1f} s")
    if not args.sem_janela:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

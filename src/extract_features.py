"""
Extrai as features usadas pelo classificador híbrido, na ordem do split fixo
(data_split/split_manifest.csv):
    1. Features clássicas (features.py), calculadas só na folha: segmenta a
       imagem ORIGINAL de cada linha com o mesmo segment_image do pipeline.
       Paralelizado com joblib.
    2. Embeddings da CNN treinada (saída da GlobalAveragePooling2D, 1280-d),
       calculados nas imagens de data_processed/ (a mesma entrada do treino da
       CNN), sem augmentation; e as probabilidades softmax da CNN no teste.

Uso:
    python src/extract_features.py               -> todas as imagens do manifest
    python src/extract_features.py --limit 5     -> 5 imagens por classe por split (teste rápido)
    python src/extract_features.py --verbose     -> mostra os logs do TensorFlow
    (requer data_split/ de split_dataset.py e models/ de train_model.py)

Gera:
    features_cache/handcrafted.csv      -> class, filename, split + 147 features clássicas
    features_cache/cnn_embeddings.npz   -> X (n x 1280), filename, class, split
    features_cache/cnn_test_pred.npy    -> probabilidades softmax da CNN nas linhas de teste
"""

import argparse
import json
import os
import sys
import time
import warnings

# --verbose mantém os logs informativos do TensorFlow (mesmo esquema do demo.py).
# Tem que ser lido antes do "import tensorflow" (feito só em compute_embeddings).
VERBOSE = "--verbose" in sys.argv
if not VERBOSE:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE, CLASS_NAMES_PATH, CNN_EMBEDDINGS_PATH, CNN_TEST_PRED_PATH, FEATURES_DIR,
    HANDCRAFTED_PATH, MODEL_IMG_SIZE, MODEL_PATH, SPLIT_MANIFEST,
)
from features import extract_handcrafted, feature_names  # noqa: E402
from preprocessing import segment_image  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="N imagens por classe por split (teste rápido)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="processos do joblib (default: -1 = todos os núcleos)")
    parser.add_argument("--verbose", action="store_true",
                        help="mostra os logs do TensorFlow")
    return parser.parse_args()


def load_manifest(limit=None):
    df = pd.read_csv(SPLIT_MANIFEST)
    if limit is not None:
        # Mantém a ordem do manifest
        df = df.groupby(["split", "class"], sort=False).head(limit)
    return df.reset_index(drop=True)


def handcrafted_one(original_path):
    """Segmenta a imagem original e extrai as features. Retorna (features, aviso)."""
    result = segment_image(original_path)
    if result is None:
        return dict.fromkeys(feature_names(), 0.0), "imagem ilegível"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = extract_handcrafted(*result)
    return feats, ("máscara vazia" if caught else None)


def extract_all_handcrafted(df, n_jobs):
    paths = df["original_path"].tolist()
    n = len(paths)
    step = max(n // 100, 1)
    results = []
    # return_as="generator" devolve os resultados na ordem de entrada, à medida que ficam prontos
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    for i, res in enumerate(parallel(delayed(handcrafted_one)(p) for p in paths), start=1):
        results.append(res)
        if i % step == 0 or i == n:
            print(f"\rFeatures clássicas: {i}/{n} ({i / n:.0%})", end="", flush=True)
    print()

    problems = [(p, msg) for p, (_, msg) in zip(paths, results) if msg]
    for p, msg in problems:
        print(f"Aviso: {msg} em {p} (features = 0)")

    feats = pd.DataFrame([f for f, _ in results], columns=feature_names())
    return pd.concat([df[["class", "filename", "split"]], feats], axis=1)


def compute_embeddings(df):
    """Embeddings (GAP, 1280-d) e softmax da CNN, na ordem de df."""
    import tensorflow as tf
    if not VERBOSE:
        tf.get_logger().setLevel("ERROR")

    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    if sorted(df["class"].unique()) != sorted(class_names):
        raise ValueError(f"As classes do manifest não batem com {CLASS_NAMES_PATH}.")

    model = tf.keras.models.load_model(MODEL_PATH)
    gap = next(layer for layer in model.layers
               if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D))
    # Um único passe dá os embeddings (GAP) e as probabilidades (saída do modelo).
    # Na inferência as camadas de augmentation e o Dropout ficam inativos.
    extractor = tf.keras.Model(model.inputs, [gap.output, model.outputs[0]])

    def load(path):
        # Mesmo carregamento do image_dataset_from_directory usado no treino da CNN:
        # decode RGB + resize bilinear para MODEL_IMG_SIZE, float 0-255
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(img, MODEL_IMG_SIZE, method="bilinear")

    ds = (tf.data.Dataset.from_tensor_slices(df["processed_path"].tolist())
          .map(load, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))
    embeddings, probs = extractor.predict(ds, verbose=1)
    return embeddings.astype(np.float32), probs.astype(np.float32)


def main():
    args = parse_args()
    os.makedirs(FEATURES_DIR, exist_ok=True)

    df = load_manifest(args.limit)
    print(f"{len(df)} imagens do manifest"
          + (f" (--limit {args.limit} por classe por split)" if args.limit else ""))
    print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())

    # 1. Features clássicas
    t0 = time.time()
    handcrafted = extract_all_handcrafted(df, args.n_jobs)
    t_hand = time.time() - t0
    handcrafted.to_csv(HANDCRAFTED_PATH, index=False)
    print(f"{handcrafted.shape[1] - 3} features clássicas salvas em: {HANDCRAFTED_PATH}")

    # 2. Embeddings + softmax da CNN
    print("\nEmbeddings da CNN...")
    t0 = time.time()
    embeddings, probs = compute_embeddings(df)
    t_emb = time.time() - t0
    np.savez_compressed(
        CNN_EMBEDDINGS_PATH, X=embeddings,
        filename=df["filename"].to_numpy(str), **{"class": df["class"].to_numpy(str)},
        split=df["split"].to_numpy(str),
    )
    print(f"Embeddings {embeddings.shape} salvos em: {CNN_EMBEDDINGS_PATH}")

    test_probs = probs[(df["split"] == "test").to_numpy()]
    np.save(CNN_TEST_PRED_PATH, test_probs)
    print(f"Softmax da CNN no teste {test_probs.shape} salvo em: {CNN_TEST_PRED_PATH}")

    n = len(df)
    print(f"\nTempo: features clássicas {t_hand:.1f} s ({1000 * t_hand / n:.1f} ms/img), "
          f"embeddings {t_emb:.1f} s ({1000 * t_emb / n:.1f} ms/img)")


if __name__ == "__main__":
    main()

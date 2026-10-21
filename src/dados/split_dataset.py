"""
Divide as imagens segmentadas em treino/validação/teste (70/15/15),
estratificado por classe e com seed fixa.

O conjunto de teste é fixo e NUNCA pode ser usado no treino nem na seleção
de modelo. O classificador híbrido (próximo checkpoint) reutiliza exatamente
esta divisão, pelo manifest.

Uso (na raiz do repositório, com src/ no PYTHONPATH: no Linux "export PYTHONPATH=src",
no PowerShell '$env:PYTHONPATH = "src"'):
    python -m dados.split_dataset
    (requer data_processed/ gerado por src/processamento_imagens/preprocessing.py)

Gera:
    data_split/{train,val,test}/<NomeDaClasse>/*.jpg   -> cópias das imagens segmentadas
    data_split/split_manifest.csv                     -> class, filename, split, processed_path, original_path
    reports/pi/dados/split_summary.csv                -> contagem por classe x split
"""

import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from common.config import (
    DATA_DIR, PROCESSED_DIR, SEED, SPLIT_DIR, SPLIT_MANIFEST, SPLIT_RATIOS, SPLIT_SUMMARY_PATH,
)
from common.utils import list_classes, list_images

SPLITS = ("train", "val", "test")


def as_posix(*parts):
    """Caminho com "/" (o manifest funciona igual no Windows e no Docker)."""
    return "/".join(parts)


def list_processed_images():
    """Retorna um DataFrame com uma linha por imagem de PROCESSED_DIR."""
    rows = []
    for class_name in list_classes(PROCESSED_DIR):
        for fname in list_images(os.path.join(PROCESSED_DIR, class_name)):
            rows.append({
                "class": class_name,
                "filename": fname,
                "processed_path": as_posix(PROCESSED_DIR, class_name, fname),
                "original_path": as_posix(DATA_DIR, class_name, fname),
            })
    return pd.DataFrame(rows)


def assign_splits(df):
    """Split estratificado em duas etapas: treino vs resto, depois resto -> val/test."""
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    train_idx, rest_idx = train_test_split(
        df.index, test_size=1 - train_ratio, stratify=df["class"], random_state=SEED,
    )
    rest = df.loc[rest_idx]
    val_idx, test_idx = train_test_split(
        rest.index, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=rest["class"], random_state=SEED,
    )
    df = df.copy()
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def copy_images(df):
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    for split in SPLITS:
        for class_name in df["class"].unique():
            os.makedirs(os.path.join(SPLIT_DIR, split, class_name), exist_ok=True)
    for class_name, fname, split, src in zip(
            df["class"], df["filename"], df["split"], df["processed_path"]):
        shutil.copy2(src, os.path.join(SPLIT_DIR, split, class_name, fname))


def save_summary(df):
    summary = pd.crosstab(df["class"], df["split"])[list(SPLITS)]
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum()
    os.makedirs(os.path.dirname(SPLIT_SUMMARY_PATH), exist_ok=True)
    summary.to_csv(SPLIT_SUMMARY_PATH)
    return summary


def main():
    df = list_processed_images()
    if df.empty:
        raise SystemExit(f"Nenhuma imagem em {PROCESSED_DIR}/. Rode preprocessing.py primeiro.")

    missing = sum(not os.path.exists(p) for p in df["original_path"])
    if missing:
        print(f"Aviso: {missing} imagens sem original correspondente em {DATA_DIR}/")

    df = assign_splits(df)
    # Ordena por split (train, val, test), classe e arquivo, para um manifest estável
    df["split"] = pd.Categorical(df["split"], categories=SPLITS, ordered=True)
    df = df.sort_values(["split", "class", "filename"])
    df["split"] = df["split"].astype(str)
    df = df[["class", "filename", "split", "processed_path", "original_path"]]

    print(f"Copiando {len(df)} imagens para {SPLIT_DIR}/ ...")
    copy_images(df)
    df.to_csv(SPLIT_MANIFEST, index=False)

    summary = save_summary(df)
    print()
    print(summary.to_string())
    print(f"\nManifest salvo em: {SPLIT_MANIFEST}")
    print(f"Resumo salvo em: {SPLIT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

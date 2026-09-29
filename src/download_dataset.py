"""Baixa o PlantVillage do Kaggle e prepara as classes de tomate e batata."""

import os
import shutil

import kagglehub

DATASET_HANDLE = "emmarex/plantdisease"
DATA_DIR = "data"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CLASS_PREFIXES = ("tomato", "potato")

def download_dataset():
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
    class_dirs = []

    for current_dir, _, filenames in os.walk(dataset_path):
        has_images = any(
            filename.lower().endswith(IMAGE_EXTENSIONS)
            for filename in filenames
        )
        class_name = os.path.basename(current_dir)
        if has_images and class_name.lower().startswith(CLASS_PREFIXES):
            class_dirs.append(current_dir)

    if not class_dirs:
        raise FileNotFoundError(
            f"Nenhuma classe de tomate ou batata encontrada em {dataset_path}."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        destination = os.path.join(DATA_DIR, class_name)
        shutil.copytree(class_dir, destination, dirs_exist_ok=True)
        print(f"Classe pronta: {destination}")

    print(f"\nDataset baixado em: {dataset_path}")
    print(f"Imagens selecionadas copiadas para: {DATA_DIR}/")


if __name__ == "__main__":
    download_dataset()
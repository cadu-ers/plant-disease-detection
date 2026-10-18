"""
Funções utilitárias: listar classes e imagens do dataset e plotar a matriz de
confusão no padrão do projeto.

Uso:
    from utils import list_classes, list_images, pick_random_image
    from utils import short_name, plot_confusion_matrix

Gera:
    nada (módulo de funções auxiliares)
"""

import os
import random
import re

import numpy as np

from config import IMAGE_EXTENSIONS


def list_classes(data_dir):
    """Retorna a lista ordenada das subpastas (classes) de data_dir."""
    return sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)))


def list_images(class_dir):
    """Retorna a lista ordenada dos arquivos de imagem de class_dir."""
    return sorted(f for f in os.listdir(class_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS))


def pick_random_image(data_dir, rng=None):
    """Sorteia uma classe e uma imagem dela. Retorna (caminho, nome_da_classe).

    rng: instância de random.Random para sorteio reprodutível; se None, usa
    o gerador global do módulo random.
    """
    rng = rng or random
    class_name = rng.choice(list_classes(data_dir))
    class_dir = os.path.join(data_dir, class_name)
    img_name = rng.choice(list_images(class_dir))
    return os.path.join(class_dir, img_name), class_name


def short_name(class_name):
    """Abrevia o nome da classe para eixos de gráficos (ex: "Tom_Early_blight")."""
    name = re.sub(r"_+", "_", class_name)
    name = name.replace("Tomato_Tomato_", "Tomato_")
    name = name.replace("_Two_spotted_spider_mite", "")
    name = re.sub(r"^Tomato_", "Tom_", name)
    name = re.sub(r"^Potato_", "Pot_", name)
    return name


def plot_confusion_matrix(y_true, y_pred, class_names, path,
                          title="Matriz de confusão (teste, normalizada por linha)"):
    """Matriz de confusão normalizada por linha (cada linha soma 1 = recall da classe).

    y_true e y_pred são índices de classe (0..len(class_names)-1).
    """
    # Imports locais: utils é usado também por scripts que não plotam
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    cm = np.nan_to_num(cm)
    labels = [short_name(c) for c in class_names]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {path}")

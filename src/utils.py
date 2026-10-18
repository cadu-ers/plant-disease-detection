"""
Funções utilitárias para listar classes e imagens do dataset.

Uso:
    from utils import list_classes, list_images, pick_random_image

Gera:
    nada (módulo de funções auxiliares)
"""

import os
import random

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

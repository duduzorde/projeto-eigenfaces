# datasets.py
import numpy as np
import cv2
import os
from sklearn.datasets import fetch_olivetti_faces

def load_olivetti():
    data = fetch_olivetti_faces()
    images = data.images
    flat = data.data
    shape = images[0].shape
    return images, flat, shape


def load_from_folder(folder_path, img_size=(64, 64)):
    imgs = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        imgs.append(img)

    imgs = np.array(imgs)
    flat = imgs.reshape(len(imgs), -1)
    return imgs, flat, img_size


def carregar_dataset(origin="olivetti", folder_path=None, img_size=(64, 64)):
    """
    origin = "olivetti"  usa fetch_olivetti_faces()
    origin = "folder"    carrega imagens de uma pasta
    """
    if origin == "olivetti":
        return load_olivetti()

    elif origin == "folder":
        if folder_path is None:
            raise ValueError("Você precisa passar folder_path para origin='folder'.")
        return load_from_folder(folder_path, img_size)

    else:
        raise ValueError("origin deve ser 'olivetti' ou 'folder'.")

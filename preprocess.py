# preprocess.py
import numpy as np

def preprocessar(images):
    """
    Recebe images com shape (n, H, W)
    Retorna:
    - X_centralizado
    - media
    - shape original (H, W)
    """
    n, H, W = images.shape

    X = images.reshape(n, H * W)

    media = np.mean(X, axis=0)
    X_centralizado = X - media

    return X_centralizado, media, (H, W)

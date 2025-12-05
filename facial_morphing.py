import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from eigenfaces import projetar_imagem, reconstruir_imagem

import numpy as np

def gerar_landmarks_sinteticos(shape):
    """
    Gera ~32 pontos sintéticos representando landmarks faciais
    adaptados ao formato médio das imagens do Olivetti Faces (64x64).
    """
    h, w = shape
    return np.array([
        # Contorno da face (em formato oval aproximado)
        (int(w*0.50), int(h*0.08)),  # topo da cabeça
        (int(w*0.30), int(h*0.15)),
        (int(w*0.15), int(h*0.30)),
        (int(w*0.10), int(h*0.50)),
        (int(w*0.15), int(h*0.70)),
        (int(w*0.30), int(h*0.85)),
        (int(w*0.50), int(h*0.92)),  # queixo
        (int(w*0.70), int(h*0.85)),
        (int(w*0.85), int(h*0.70)),
        (int(w*0.90), int(h*0.50)),
        (int(w*0.85), int(h*0.30)),
        (int(w*0.70), int(h*0.15)),

        # Sobrancelhas
        (int(w*0.35), int(h*0.28)),
        (int(w*0.45), int(h*0.26)),
        (int(w*0.55), int(h*0.26)),
        (int(w*0.65), int(h*0.28)),

        # Olhos
        (int(w*0.35), int(h*0.38)),
        (int(w*0.45), int(h*0.40)),
        (int(w*0.55), int(h*0.40)),
        (int(w*0.65), int(h*0.38)),

        # Nariz
        (int(w*0.50), int(h*0.48)),
        (int(w*0.45), int(h*0.52)),
        (int(w*0.55), int(h*0.52)),
        (int(w*0.50), int(h*0.58)),

        # Boca
        (int(w*0.40), int(h*0.66)),
        (int(w*0.50), int(h*0.68)),
        (int(w*0.60), int(h*0.66)),
        (int(w*0.45), int(h*0.73)),
        (int(w*0.55), int(h*0.73)),
        (int(w*0.50), int(h*0.76)),
    ])


def interpolar_geom(A_pts, B_pts, t):
    return (1 - t) * A_pts + t * B_pts

from scipy.spatial import Delaunay

def triangulos(points):
    tri = Delaunay(points)
    return tri.simplices

def warp_triangulo(src, dst, tri_src, tri_dst):
    M = cv2.getAffineTransform(np.float32(tri_src), np.float32(tri_dst))
    warped = cv2.warpAffine(src, M, (dst.shape[1], dst.shape[0]))
    return warped

def morphing_facial(imgA, imgB, media, eigenfaces, k, n_passos=10):

    h, w = imgA.shape

    ptsA = gerar_landmarks_sinteticos((h, w))
    ptsB = gerar_landmarks_sinteticos((h, w))

    # triangulação fixa baseada nos pontos sintéticos
    tri = triangulos((ptsA + ptsB) / 2)

    frames = []

    for i in range(n_passos):
        t = i / (n_passos - 1)

        # interpolação geométrica dos pontos
        ptsT = ptsA*(1-t) + ptsB*t

        imgA_warp = np.zeros((h, w), np.float32)
        imgB_warp = np.zeros((h, w), np.float32)

        # warp triângulo a triângulo
        for simplex in tri:
            triA = ptsA[simplex]
            triB = ptsB[simplex]
            triT = ptsT[simplex]

            imgA_warp += warp_triangulo(imgA, np.zeros((h, w)), triA, triT)
            imgB_warp += warp_triangulo(imgB, np.zeros((h, w)), triB, triT)

        # blend
        img_t = (1 - t)*imgA_warp + t*imgB_warp

        # refinamento com PCA
        w_t = np.dot(eigenfaces[:, :k].T, (img_t.flatten() - media))
        img_final = np.dot(eigenfaces[:, :k], w_t) + media

        frames.append(img_final.reshape(h, w))

    return frames

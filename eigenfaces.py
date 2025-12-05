# eigenfaces.py
import numpy as np

def calcular_pca(X_centralizado):
    """
    Recebe X centralizado (n_imagens, n_pixels)
    Retorna:
    - autovetores (eigenfaces)
    - autovalores
    """

    # método mais eficiente: PCA via covariância reduzida
    # C = X X^T  (ao invés de X^T X)
    C = np.dot(X_centralizado, X_centralizado.T)

    print("Calculando autovalores/autovetores...")
    autovalores, autovetores_reduzidos = np.linalg.eigh(C)

    # ordenar em ordem decrescente
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovetores_reduzidos = autovetores_reduzidos[:, idx]

    # converter autovetores reduzidos para eigenfaces
    eigenfaces = np.dot(X_centralizado.T, autovetores_reduzidos)

    # normalização: cada eigenface deve ter norma 1
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    return eigenfaces, autovalores


def projetar_imagem(img_flat, media, eigenfaces, k):
    """
    Projeta uma imagem (vetor 1D) no subespaço das k eigenfaces.
    """
    img_centered = img_flat - media
    return np.dot(eigenfaces[:, :k].T, img_centered)


def reconstruir_imagem(projecao, media, eigenfaces, k):
    """
    Reconstrói imagem a partir da projeção nas eigenfaces.
    """
    rec = np.dot(eigenfaces[:, :k], projecao) + media
    return rec


def variancia_explicada(autovalores):
    """
    Retorna a variância explicada acumulada do PCA.
    Útil para escolher o número de componentes k.
    """
    total = np.sum(autovalores)
    return np.cumsum(autovalores) / total


def gerar_face_sintetica(media, eigenfaces, autovalores, k):
    """
    Gera uma nova face usando distribuição normal nas componentes principais.

    - Cada componente é amostrada como:
      alpha_i ~ Normal(0, sqrt(lambda_i))

    Isso produz rostos plausíveis no espaço PCA.
    """

    # variância para cada componente
    desvios = np.sqrt(autovalores[:k])

    # amostra aleatória nas k dimensões principais
    coeficientes = np.random.randn(k) * desvios

    # sintetiza a nova imagem
    face_flat = np.dot(eigenfaces[:, :k], coeficientes) + media

    return face_flat


def gerar_varias_faces(media, eigenfaces, autovalores, k, n=10):
    """
    Gera n rostos sintéticos e retorna matriz (n, H*W)
    """
    faces = []
    for _ in range(n):
        faces.append(gerar_face_sintetica(media, eigenfaces, autovalores, k))
    return np.array(faces)

def interpolar_entre_faces(imgA_flat, imgB_flat, media, eigenfaces, k, n_passos=10):
    """
    Gera n passos de interpolação entre duas faces, no espaço das eigenfaces.
    Retorna uma matriz (n_passos, H*W).
    """

    # projeções das imagens
    wA = projetar_imagem(imgA_flat, media, eigenfaces, k)
    wB = projetar_imagem(imgB_flat, media, eigenfaces, k)

    interpolacoes = []

    for i in range(n_passos):
        t = i / (n_passos - 1)
        w_t = (1 - t) * wA + t * wB
        img_t = reconstruir_imagem(w_t, media, eigenfaces, k)
        interpolacoes.append(img_t)

    return np.array(interpolacoes)
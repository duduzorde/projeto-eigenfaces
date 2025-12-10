# eigenfaces.py
import numpy as np
import matplotlib.pyplot as plt

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

def plot_erro_reconstrucao_por_k(X_flat, media, eigenfaces, shape, idx_img):
    """
    Plota o erro ||x - x_k|| para k = 1 até o máximo possível.
    X_flat: matriz (n_amostras, d)
    media: vetor (d,)
    eigenfaces: matriz (d, d) ou (d, n_comp)
    shape: tupla de reshape
    idx_img: índice da imagem que será reconstruída
    """

    x = X_flat[idx_img]        # imagem original (flatten)
    d = eigenfaces.shape[1]    # número máximo de componentes possíveis

    erros = []

    for k in range(1, d + 1):
        # projeção em k componentes
        w_k = projetar_imagem(x, media, eigenfaces, k)

        # reconstrução
        rec_k = reconstruir_imagem(w_k, media, eigenfaces, k)

        # erro na norma 2
        erro = np.linalg.norm(x - rec_k)
        erros.append(erro)

    # --- plot ---
    plt.figure(figsize=(8,4))
    plt.plot(range(1, d+1), erros, marker="o")
    plt.xlabel("Número de componentes principais (k)")
    plt.ylabel("Erro de reconstrução ||x − x_k||")
    plt.title(f"Erro de reconstrução para a imagem {idx_img}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return erros


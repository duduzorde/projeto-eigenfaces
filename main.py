from datasets import carregar_dataset
from preprocess import preprocessar
from eigenfaces import calcular_pca, projetar_imagem, reconstruir_imagem, gerar_face_sintetica, interpolar_entre_faces, plot_erro_reconstrucao_por_k
import matplotlib.pyplot as plt
import numpy as np

def plotar_eigenfaces(eigenfaces, shape, n=None):
    """
    Plota as eigenfaces do dataset.

    eigenfaces: matriz (d, m) sendo cada coluna uma eigenface.
    n: quantidade de eigenfaces a exibir (default = todas)
    """
    H, W = shape
    total = eigenfaces.shape[1]

    if n is None or n > total:
        n = total

    cols = 10
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 1.2 * rows))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        face = eigenfaces[:, i].reshape(H, W)
        plt.imshow(face, cmap="gray")
        plt.title(f"PC {i+1}")
        plt.axis("off")

    plt.suptitle("Eigenfaces", fontsize=16)
    plt.tight_layout()
    plt.show()

def plotar_face_media(media, shape):
    """
    Plota a face média (mean face) do dataset.
    """
    H, W = shape
    mean_img = media.reshape(H, W)

    plt.figure(figsize=(4, 4))
    plt.imshow(mean_img, cmap="gray")
    plt.title("Mean Face (Face Média)")
    plt.axis("off")
    plt.show()



# escolha do dataset
# images, X_flat, shape = carregar_dataset("olivetti")
images, X_flat, shape = carregar_dataset("folder", folder_path="dataset_padronizado\pessoa1")

# pré-processamento
Xc, media, shape = preprocessar(images)

# PCA / Eigenfaces
eigenfaces, autovalores = calcular_pca(Xc)

# escolher número de componentes
k = 20

plotar_eigenfaces(eigenfaces,shape)

"""
# projetar 1 imagem
i = 31
w = projetar_imagem(X_flat[i], media, eigenfaces, k)

# reconstruir
rec = reconstruir_imagem(w, media, eigenfaces, k)
rec_img = rec.reshape(shape)

# visualizar
plt.subplot(1,2,1)
plt.imshow(images[i], cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rec_img, cmap="gray")
plt.title(f"Reconstrução (k={k})")
plt.axis("off")

plt.show()

erros = plot_erro_reconstrucao_por_k(X_flat, media, eigenfaces, shape, idx_img=i)
#"""

"""
# gerar 1 rosto sintético
face_flat = gerar_face_sintetica(media, eigenfaces, autovalores, k)
face_img = face_flat.reshape(shape)

plt.imshow(face_img, cmap="gray")
plt.title("Face Sintética Gerada")
plt.axis("off")
plt.show()
"""

"""
# escolher duas imagens do dataset
A_id = 36
B_id = 49

A = X_flat[A_id]
B = X_flat[B_id]
A_img = images[A_id]
B_img = images[B_id]


# gerar interpolação
interps = interpolar_entre_faces(A, B, media, eigenfaces, k, n_passos=9)

# visualizar

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Imagem A (inicial)")
plt.imshow(A_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagem B (final)")
plt.imshow(B_img, cmap="gray")
plt.axis("off")

plt.figure(figsize=(12, 2))
for i in range(len(interps)):
    plt.subplot(1, len(interps), i+1)
    plt.imshow(interps[i].reshape(shape), cmap="gray")
    plt.axis("off")
    plt.title(f"t={i/(len(interps)-1):.2f}")
plt.tight_layout()
plt.show()
"""





"""

# interpolação diferenciada (somando a imagem A projetada nos PCA 1 até k/2 com imagem B nos PCA k/2 até k)

# projetar 1 imagem
A_id = 6
wA = projetar_imagem(X_flat[A_id], media, eigenfaces, round(k/2))
wA2 = np.pad(wA, (0,round(k/2)), mode='constant', constant_values=0)

B_id = 0
wB1 = projetar_imagem(X_flat[B_id], media, eigenfaces, k)
wB2 = projetar_imagem(X_flat[B_id], media, eigenfaces, round(k/2))
wB22 = np.pad(wB2, (0,round(k/2)), mode='constant', constant_values=0)
wB = wB1 - wB22

w = wA2 + wB

# reconstruir
rec = reconstruir_imagem(w, media, eigenfaces, k)
rec_img = rec.reshape(shape)


# visualizar
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Imagem A")
plt.imshow(images[A_id], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagem B")
plt.imshow(images[B_id], cmap="gray")
plt.axis("off")

plt.figure(figsize=(4, 3))
plt.imshow(rec_img, cmap="gray")
plt.title(f"Reconstrução (k={k})")
plt.axis("off")

plt.show()
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

IMG_PATH = Path("Img1.jpeg")  # formatos aceptados jpeg, png, jpg
OUTPUT_PATH = Path("outputs") / "actividad_5_2.png"
VARIANZAS = [0.70, 0.99]


def main() -> None:
    # 1. Cargar imagen
    img = Image.open(IMG_PATH)
    img_gray = img.convert("L")  # convertimos a escala de grises
    img_resized = img_gray.resize((512, 512), Image.LANCZOS)  # redimensionar la imagen
    img_array = np.array(img_resized, dtype=float)  # obtener la matriz numérica de la imagen

    # 2. Aplicar PCA para compresión
    # img_array tiene dimensión (512, 512): cada fila es una muestra, cada columna una feature
    reconstructions = []
    infos = []
    for explained_variance in VARIANZAS:
        pca = PCA(explained_variance)
        pca.fit(img_array)
        z = pca.transform(img_array)  # representación comprimida
        K = pca.n_components_

        # 3. Reconstrucción
        img_approx = pca.inverse_transform(z)
        reconstructions.append(img_approx)

        variance_retained = np.sum(pca.explained_variance_ratio_)
        infos.append((explained_variance, K, variance_retained))

        # 5. Mostrar información
        print(f"\nVarianza objetivo: {explained_variance * 100:.0f}%")
        print("Dimensión original:", img_array.shape)
        print("Dimensión comprimida (z):", z.shape)
        print("Número de componentes principales K:", K)
        print(f"Varianza retenida real: {variance_retained * 100:.2f}%")

    # 4. Visualización: original + reconstrucciones (70% y 99%)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Imagen original")
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")

    for i, (explained_variance, K, variance_retained) in enumerate(infos):
        plt.subplot(1, 3, i + 2)
        plt.title(f"{explained_variance * 100:.0f}% varianza (k={K})")
        plt.imshow(reconstructions[i], cmap="gray")
        plt.axis("off")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

OUTPUT_PATH = Path("outputs") / "actividad_2_1_pca_student_performance_numeric.png"


def main() -> None:
    # 1) Cargar dataset (id=320: Student Performance)
    student_performance = fetch_ucirepo(id=320)
    X = student_performance.data.features
    y = student_performance.data.targets  # G1, G2, G3 (notas 0-20)

    # 2) Quedarnos solo con variables numéricas (sin one-hot de categóricas)
    X_num = X.select_dtypes(include=[np.number])
    print("Variables numéricas usadas:", list(X_num.columns))
    print("Dimensión (solo numéricas):", X_num.shape)

    # 3) Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # 4) PCA con 2 componentes para graficar
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("Varianza explicada (PC1, PC2):", pca.explained_variance_ratio_)
    print(f"Varianza total explicada por PC1+PC2: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    # 5) Loadings
    loadings = pd.DataFrame(pca.components_.T, index=X_num.columns, columns=["PC1", "PC2"])
    print("\nLoadings (todas las variables, son pocas):")
    print(loadings)

    # 6) Cuántos componentes para retener 80% / 95% de la varianza (comparar contra activity_2.py)
    pca_full = PCA().fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    for VR in (0.80, 0.95):
        k = int(np.argmax(cumulative_variance_ratio >= VR) + 1)
        print(f"Componentes necesarios para retener {VR * 100:.0f}% de la varianza: {k}")

    # 7) Graficar coloreando por la nota final G3 (para buscar patrones de rendimiento)
    g3 = y["G3"].to_numpy()
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=g3, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(scatter, label="Nota final (G3)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (solo numéricas) - Student Performance, coloreado por G3")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

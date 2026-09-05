from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

OUTPUT_PATH = Path("outputs") / "actividad_2_pca_student_performance.png"


def main() -> None:
    # 1) Cargar dataset (id=320: Student Performance)
    student_performance = fetch_ucirepo(id=320)
    X = student_performance.data.features  # solo features (sin notas G1/G2/G3)
    print("Dimensión original (filas, columnas):", X.shape)

    # 2) Preprocesar: one-hot a categóricas + quitar inf/NaN
    X_proc = pd.get_dummies(X, drop_first=True)
    X_proc = X_proc.replace([np.inf, -np.inf], np.nan).dropna()

    # 3) Estandarizar (PCA necesita media 0 y varianza 1 en cada variable)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)

    # 4) PCA con 2 componentes para poder graficar
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("Dimensión tras one-hot + estandarizar:", X_scaled.shape)
    print("Varianza explicada (PC1, PC2):", pca.explained_variance_ratio_)
    print(f"Varianza total explicada por PC1+PC2: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    # 5) Loadings: qué variables definen cada componente
    loadings = pd.DataFrame(pca.components_.T, index=X_proc.columns, columns=["PC1", "PC2"])
    print("\nTop 10 variables por |PC1|:")
    print(loadings.reindex(loadings.PC1.abs().sort_values(ascending=False).index).head(10))
    print("\nTop 10 variables por |PC2|:")
    print(loadings.reindex(loadings.PC2.abs().sort_values(ascending=False).index).head(10))

    # 6) Cuántos componentes se necesitan para retener 80% / 95% de la varianza (estilo SVD manual)
    pca_full = PCA().fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    for VR in (0.80, 0.95):
        k = int(np.argmax(cumulative_variance_ratio >= VR) + 1)
        print(f"Componentes necesarios para retener {VR * 100:.0f}% de la varianza: {k}")

    # 7) Graficar (PCA puro, sin usar la variable objetivo)
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=15, c="steelblue")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - Student Performance (UCI id=320)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

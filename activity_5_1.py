from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data import get_dataset

OUTPUT_PATH = Path("outputs") / "varianza_vs_k.png"


def main() -> None:
    X, filenames, images = get_dataset()

    # --- Actividad 1: Varianza retenida vs Número de componentes (k) ---
    varianza = [0.99, 0.75, 0.3]
    num_compo_k = []

    for x in varianza:
        explained_variance = x  # varianza retenida = varianza explicada = "información" retenida
        pca = PCA(explained_variance)
        pca.fit(X)
        k = pca.n_components_
        num_compo_k.append(k)
        print(f"Varianza retenida={explained_variance} -> componentes principales k={k}")

    # aqui codigo para dibujar
    plt.figure(figsize=(10, 5))
    plt.title("Varianza vs k components")
    plt.xlabel("Número de componentes principales (k)")
    plt.ylabel("Varianza retenida")
    plt.plot(num_compo_k, varianza, marker="o")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada en: {OUTPUT_PATH}")
    plt.show()

    print("\nAnálisis:")
    print(
        "La relación entre la varianza retenida y el número de componentes k es "
        "no lineal y de rendimientos decrecientes: para retener el "
        f"{varianza[2] * 100:.0f}% de la varianza solo se necesitan {num_compo_k[2]} "
        f"componentes, mientras que para el {varianza[1] * 100:.0f}% se requieren "
        f"{num_compo_k[1]} y para el {varianza[0] * 100:.0f}% se necesitan "
        f"{num_compo_k[0]} componentes. Esto ocurre porque las primeras componentes "
        "principales capturan los patrones más generales (mayor varianza) de las "
        "caras, mientras que las componentes adicionales aportan cada vez menos "
        "información nueva (detalles finos). Por eso, elegir un k asociado a una "
        "varianza moderada-alta (por ejemplo 0.95-0.99) suele ser el mejor balance "
        "entre compresión (menos features) y calidad de reconstrucción."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

LFW_URL = "http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "lfwcrop_grey.zip"
DATASET_DIR = DATA_DIR / "lfwcrop_grey"
IMAGE_SIZE = 64


def download_dataset(url: str = LFW_URL, zip_path: Path = ZIP_PATH) -> None:
    if zip_path.exists():
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Descargando dataset"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_dataset(zip_path: Path = ZIP_PATH, dest_dir: Path = DATA_DIR) -> None:
    if DATASET_DIR.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def load_faces(dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, list[str], np.ndarray]:
    faces_dir = dataset_dir / "faces"

    filenames = []
    images = []
    for filename in os.listdir(faces_dir):
        filenames.append(filename)
        image = np.array(Image.open(os.path.join(faces_dir, filename)))
        images.append(image)

    images = np.array(images)
    print("Total Number of Faces: {}".format(len(images)))
    print(images.shape)

    n = IMAGE_SIZE * IMAGE_SIZE  # dimensión de mis datos (original) n = 4096 features
    X = images.reshape(len(images), n)  # m = ejemplos de entrenamiento
    print(X.shape)

    return X, filenames, images


def get_dataset() -> tuple[np.ndarray, list[str], np.ndarray]:
    download_dataset()
    extract_dataset()
    return load_faces()

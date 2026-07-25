"""Small, dependency-light MNIST loader for the baseline experiment."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(data_dir: Path, filename: str) -> Path:
    destination = data_dir / filename
    if destination.exists():
        return destination
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(BASE_URL + filename, destination)
    return destination


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic number in {path}: {magic}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * columns
    if values.size != expected:
        raise ValueError(f"Expected {expected} image bytes in {path}, found {values.size}")
    return values.reshape(count, rows * columns).astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic number in {path}: {magic}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"Expected {count} labels in {path}, found {values.size}")
    return values.astype(np.int64)


def load_mnist(
    data_dir: str | Path,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (once), parse, normalize, and optionally truncate MNIST."""
    root = Path(data_dir)
    paths = {name: _download_if_missing(root, filename) for name, filename in FILES.items()}
    train_x = _read_images(paths["train_images"])
    train_y = _read_labels(paths["train_labels"])
    test_x = _read_images(paths["test_images"])
    test_y = _read_labels(paths["test_labels"])
    return (
        train_x[:train_limit],
        train_y[:train_limit],
        test_x[:test_limit],
        test_y[:test_limit],
    )


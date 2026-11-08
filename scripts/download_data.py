#!/usr/bin/env python3
"""Download Fashion-MNIST and convert it to the CSV files the network reads.

Output (in data/ by default):
    fashion_mnist_train_vectors.csv   60000 lines, 784 comma-separated pixels (0-255)
    fashion_mnist_train_labels.csv    60000 lines, one label (0-9)
    fashion_mnist_test_vectors.csv    10000 lines
    fashion_mnist_test_labels.csv     10000 lines

Uses only the Python standard library.
Dataset: https://github.com/zalandoresearch/fashion-mnist (MIT License).
"""
import argparse
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
]

FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def fetch(name: str) -> bytes:
    """Download one gzipped IDX file, trying each mirror in turn."""
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(base + name, timeout=60) as response:
                return gzip.decompress(response.read())
        except Exception as err:  # try the next mirror
            print(f"  {base + name} failed: {err}", file=sys.stderr)
    sys.exit(f"Could not download {name}")


def parse_images(raw: bytes) -> list[bytes]:
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, "not an IDX image file"
    size = rows * cols
    return [raw[16 + i * size : 16 + (i + 1) * size] for i in range(count)]


def parse_labels(raw: bytes) -> bytes:
    magic, count = struct.unpack(">II", raw[:8])
    assert magic == 2049, "not an IDX label file"
    return raw[8 : 8 + count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data", help="output folder (default: data)")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    for split, (images_file, labels_file) in FILES.items():
        print(f"Downloading {split} set...")
        images = parse_images(fetch(images_file))
        labels = parse_labels(fetch(labels_file))

        with open(out / f"fashion_mnist_{split}_vectors.csv", "w", newline="\n") as f:
            for image in images:
                f.write(",".join(map(str, image)) + "\n")
        with open(out / f"fashion_mnist_{split}_labels.csv", "w", newline="\n") as f:
            f.writelines(f"{label}\n" for label in labels)

        print(f"  wrote {len(images)} samples to {out}/fashion_mnist_{split}_*.csv")


if __name__ == "__main__":
    main()

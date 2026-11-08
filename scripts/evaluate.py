#!/usr/bin/env python3
"""Print accuracy, per-class accuracy and the confusion matrix for a predictions file.

Usage:
    python3 scripts/evaluate.py [predictions] [labels] [--min-accuracy 0.88]
"""
import argparse
import sys

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def read_ints(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", nargs="?", default="test_predictions.csv")
    parser.add_argument("labels", nargs="?", default="data/fashion_mnist_test_labels.csv")
    parser.add_argument("--min-accuracy", type=float,
                        help="exit with status 1 if accuracy is below this value (used in CI)")
    args = parser.parse_args()

    pred, true = read_ints(args.predictions), read_ints(args.labels)
    if len(pred) != len(true):
        sys.exit(f"Length mismatch: {len(pred)} predictions, {len(true)} labels")

    confusion = [[0] * 10 for _ in range(10)]  # rows: true class, columns: predicted
    for t, p in zip(true, pred):
        confusion[t][p] += 1

    correct = sum(confusion[i][i] for i in range(10))
    accuracy = correct / len(true)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(true)})\n")

    print(f"{'class':<12} {'accuracy':>8}   most often confused with")
    for i, name in enumerate(CLASSES):
        total = sum(confusion[i])
        mistakes = [(confusion[i][j], j) for j in range(10) if j != i]
        count, j = max(mistakes)
        print(f"{name:<12} {confusion[i][i] / total:>8.1%}   {CLASSES[j]} ({count})")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print("     " + "".join(f"{j:>6}" for j in range(10)))
    for i in range(10):
        print(f"{i:>3}  " + "".join(f"{c:>6}" for c in confusion[i]))

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        sys.exit(f"\nAccuracy {accuracy:.2%} is below the required {args.min_accuracy:.2%}")


if __name__ == "__main__":
    main()

# Neural network from scratch in C++

A feed-forward neural network written in plain C++17, without any machine learning or
linear algebra libraries. It learns to classify images of clothing from the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset (10 classes,
28×28 grayscale images) and reaches **90.0% accuracy on the test set**.

I wrote it as the semester project for a neural networks course at Masaryk University
(2025). The rules were: only the C++ standard library, at least 88% test accuracy, and
the whole run (load data, train, predict) in under 10 minutes.

## What it does

- Loads the 60,000 training and 10,000 test images from CSV files and normalizes them.
- Keeps the last 10,000 training images aside as a validation set.
- Trains a network with layers 784 → 256 → 128 → 10 (ReLU in the hidden layers,
  softmax output, cross-entropy loss) using backpropagation.
- Uses mini-batches of 128 images, the Adam optimizer, He initialization and weight
  decay, and lowers the learning rate when validation accuracy stops improving.
- After 30 epochs, writes the predicted class of every training and test image to
  `train_predictions.csv` and `test_predictions.csv`.

Everything is implemented by hand in [`src/`](src): the matrix class, forward and
backward pass, activation and loss functions, and the optimizers (SGD, SGD with
momentum and Adam; Adam is used).

## Results

| Data set                 | Accuracy |
| ------------------------ | -------: |
| Test (10,000 images)     |   90.00% |
| Training (60,000 images) |   94.67% |

These come from the run I submitted in the course. Most mistakes are between similar
tops, for example shirts predicted as T-shirts.

## Build and run

Requirements: `g++` with C++17, `make`, and Python 3 (only for downloading the data).

```bash
make run        # downloads Fashion-MNIST into data/, compiles and trains
make evaluate   # prints test accuracy, per-class accuracy and the confusion matrix
```

Training takes a few minutes on a laptop CPU (single-threaded).

## Files

| File                       | Contents                                                   |
| -------------------------- | ---------------------------------------------------------- |
| `src/main.cpp`             | training loop, validation split, learning rate decay       |
| `src/network.h`            | network: layers, batch training, loss functions, prediction |
| `src/layer.h`              | fully connected layer: forward and backward pass           |
| `src/optimizer.h`          | SGD, SGD with momentum, Adam                               |
| `src/matrix.h`             | matrix class and weight initialization                     |
| `src/activations.h`        | activation functions                                       |
| `src/data_loader.h`        | CSV loading, normalization, one-hot labels                 |
| `scripts/download_data.py` | downloads Fashion-MNIST and converts it to CSV             |
| `scripts/evaluate.py`      | compares predictions with the true labels                  |

## License

[MIT](LICENSE)

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -flto -funroll-loops -march=native -fno-rtti -ffast-math -fprefetch-loop-arrays -Wall
DATA     := data/fashion_mnist_train_vectors.csv

# Build the binary
neural_net: src/*.cpp src/*.h
	$(CXX) $(CXXFLAGS) -o $@ src/*.cpp

# Download Fashion-MNIST into data/ (only if missing)
$(DATA):
	python3 scripts/download_data.py

data: $(DATA)

# Train, then write train_predictions.csv and test_predictions.csv
run: neural_net $(DATA)
	./neural_net

# Accuracy, per-class accuracy and confusion matrix on the test set
evaluate:
	python3 scripts/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv

clean:
	rm -f neural_net train_predictions.csv test_predictions.csv

.PHONY: data run evaluate clean

#include "network.h"
#include "data_loader.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>

/**
 * Main training function for Fashion-MNIST classification
 * Uses a 3-layer neural network (784->256->128->10) with Adam optimizer
 * Splits 60k training data into 50k train / 10k validation
 */
void train_network() {
    std::srand(42);
    
    // Load training and test datasets
    DataLoader data_loader;
    data_loader.load_data("data/fashion_mnist_train_vectors.csv",
                          "data/fashion_mnist_train_labels.csv");
    
    DataLoader test_data;
    test_data.load_data("data/fashion_mnist_test_vectors.csv",
                        "data/fashion_mnist_test_labels.csv");
    
    // Split training data into train (50k) and validation (10k) sets
    const size_t total_data_size = data_loader.size();
    const size_t training_size = 50000;
    const size_t validation_size = total_data_size - training_size;
    
    auto& all_features = data_loader.get_features();
    auto& all_labels = data_loader.get_labels();
    
    std::vector<std::vector<scalar_t>> train_features(all_features.begin(), 
                                                       all_features.begin() + training_size);
    std::vector<std::vector<scalar_t>> train_labels(all_labels.begin(), 
                                                     all_labels.begin() + training_size);
    std::vector<std::vector<scalar_t>> val_features(all_features.begin() + training_size, 
                                                     all_features.end());
    std::vector<std::vector<scalar_t>> val_labels(all_labels.begin() + training_size, 
                                                   all_labels.end());
    
    // Create network: 784 input (28x28 images), 2 hidden layers (256, 128), 10 output classes
    // Using ReLU activation for hidden layers, Softmax for output, Adam optimizer
    std::vector<size_t> topology = {784, 256, 128, 10};
    Network net(topology, RELU, SOFTMAX, CROSS_ENTROPY, 0.005, ADAM);
    net.initialize_weights_he();

    std::cout << "Training: " << training_size << " samples | Validation: " << validation_size << " samples" << std::endl;
    std::cout << "Architecture: 784 -> 256 -> 128 -> 10 | Epochs: 30 | Batch size: 128" << std::endl << std::endl;
    
    const int total_epochs = 30;
    const size_t mini_batch_size = 128;
    
    auto train_start_time = std::chrono::high_resolution_clock::now();
    
    double peak_val_acc = 0.0;
    
    // Training loop
    for (int ep = 0; ep < total_epochs; ep++) {
        auto ep_start = std::chrono::high_resolution_clock::now();
        
        std::cout << "Epoch " << (ep + 1) << "/" << total_epochs << ":" << std::endl;
        
        // Mini-batch training: process training data in batches of 128
        size_t train_batches = (training_size + mini_batch_size - 1) / mini_batch_size;
        for (size_t b = 0; b < train_batches; b++) {
            size_t batch_start = b * mini_batch_size;
            size_t batch_end = std::min(batch_start + mini_batch_size, training_size);
            
            std::vector<std::vector<scalar_t>> batch_x(train_features.begin() + batch_start,
                                                        train_features.begin() + batch_end);
            std::vector<std::vector<scalar_t>> batch_y(train_labels.begin() + batch_start,
                                                        train_labels.begin() + batch_end);
            
            net.train_batch(batch_x, batch_y);
        }
        
        // Evaluate on validation set after each epoch
        double current_val_acc = net.evaluate_accuracy(val_features, val_labels);
        
        auto ep_end = std::chrono::high_resolution_clock::now();
        auto ep_time = std::chrono::duration_cast<std::chrono::milliseconds>(ep_end - ep_start);
        
        std::cout << "  Validation Accuracy: " << std::fixed << std::setprecision(2) 
                  << (current_val_acc * 100) << "%";
        std::cout << " | Epoch Time: " << ep_time.count() << "ms" << std::endl;
        
        // Learning rate decay: reduce LR if validation accuracy doesn't improve
        if (current_val_acc <= peak_val_acc) {
            double updated_lr = net.get_learning_rate() * 0.6;
            net.set_learning_rate(updated_lr);
            std::cout << "  Learning rate decreased to: " << std::fixed << std::setprecision(6)
                      << updated_lr << std::endl;
        } else {
            peak_val_acc = current_val_acc;
        }
        std::cout << std::endl;
    }
    
    auto train_end_time = std::chrono::high_resolution_clock::now();
    auto total_train_time = std::chrono::duration_cast<std::chrono::seconds>(train_end_time - train_start_time);
    std::cout << "Training completed in " << total_train_time.count() << " seconds" << std::endl << std::endl;
    
    // Generate predictions for submission
    std::cout << "Generating predictions..." << std::endl;
    
    // Predict on all 60k training samples
    std::vector<int> training_preds;
    training_preds.reserve(all_features.size());
    for (const auto& input : all_features) {
        training_preds.push_back(net.predict(input));
    }
    DataLoader::save_predictions("train_predictions.csv", training_preds);
    std::cout << "Saved train_predictions.csv" << std::endl;
    
    // Predict on 10k test samples
    std::vector<int> testing_preds;
    testing_preds.reserve(test_data.get_features().size());
    for (const auto& input : test_data.get_features()) {
        testing_preds.push_back(net.predict(input));
    }
    DataLoader::save_predictions("test_predictions.csv", testing_preds);
    std::cout << "Saved test_predictions.csv" << std::endl;
}

int main() {
    train_network();
    return 0;
}

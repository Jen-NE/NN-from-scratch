#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <vector>
#include <string>
#include "optimizer.h"
#include "data_loader.h"

enum LossType {
    MEAN_SQUARED_ERROR,
    CROSS_ENTROPY
};

enum OptimizerType {
    SGD,
    SGD_MOMENTUM,
    ADAM
};

// Loss function implementations
scalar_t mse_loss(const std::vector<scalar_t>&, const std::vector<scalar_t>&);
std::vector<scalar_t> mse_gradient(const std::vector<scalar_t>&, const std::vector<scalar_t>&);
scalar_t cross_entropy_loss(const std::vector<scalar_t>&, const std::vector<scalar_t>&);
std::vector<scalar_t> cross_entropy_gradient(const std::vector<scalar_t>&, const std::vector<scalar_t>&);

/**
 * Neural Network class
 * Manages multiple layers, forward/backward propagation, and optimization
 */
class Network {
private:
    std::vector<Layer> layers;
    LossType loss_type;
    scalar_t learning_rate;
    OptimizerType optimizer_type;
    
    std::vector<AdamState> adam_states;
    std::vector<MomentumState> momentum_states;
    AdamConfig adam_config;
    MomentumConfig momentum_config;

public:
    Network(const std::vector<size_t>& layer_sizes,
            ActivationType hidden_activation,
            ActivationType output_activation,
            LossType loss,
            scalar_t learning_rate = 0.01,
            OptimizerType opt_type = ADAM):

        loss_type(loss),
        learning_rate(learning_rate),
        optimizer_type(opt_type),
        adam_config(learning_rate, 0.9, 0.999, 1e-8, 0.0001),
        momentum_config(learning_rate, 0.9, 0.0001){

        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            ActivationType act_type = (i == layer_sizes.size() - 2) ? output_activation : hidden_activation;
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], act_type);
        }
        
        if (optimizer_type == ADAM) {
            adam_states.resize(layers.size());
        } else if (optimizer_type == SGD_MOMENTUM) {
            momentum_states.resize(layers.size());
        }
        }
    
    void initialize_weights_xavier(){
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].initialize_weights_xavier();
        }
    }
    void initialize_weights_he(){
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].initialize_weights_he();
        }
    }

    std::vector<scalar_t> forward(const std::vector<scalar_t>& input){
        std::vector<scalar_t> activation = input;
        for (size_t i = 0; i < layers.size(); i++) {
            activation = layers[i].forward(activation);
        }
        return activation;
    }
    
    // Backward pass for a single sample
    void backward(const std::vector<scalar_t>& target){
        const Matrix& last_act_matrix = layers.back().get_last_activation();
        std::vector<scalar_t> output(last_act_matrix.get_cols());
        std::copy(last_act_matrix.get_data().begin(), 
                  last_act_matrix.get_data().begin() + last_act_matrix.get_cols(),
                  output.begin());
        
        std::vector<scalar_t> loss_gradient;
        if (loss_type == MEAN_SQUARED_ERROR) {
            loss_gradient = mse_gradient(output, target);
        } else if (loss_type == CROSS_ENTROPY) {
            loss_gradient = cross_entropy_gradient(output, target);
        } else {
            throw std::invalid_argument("Unsupported loss type");
        }
        
        std::vector<scalar_t> gradient = loss_gradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers[i].backward(gradient);
        }
    }
    
    // Update parameters using selected optimizer
    void update_parameters(){
        for (size_t i = 0; i < layers.size(); i++) {
            if (optimizer_type == ADAM) {
                adam_update(layers[i].get_weights(), 
                           layers[i].get_biases(),
                           layers[i].get_weight_gradients(),
                           layers[i].get_bias_gradients(),
                           adam_states[i],
                           adam_config);
            } else if (optimizer_type == SGD_MOMENTUM) {
                momentum_update(layers[i].get_weights(),
                               layers[i].get_biases(),
                               layers[i].get_weight_gradients(),
                               layers[i].get_bias_gradients(),
                               momentum_states[i],
                               momentum_config);
            } else {
                sgd_update(layers[i].get_weights(),
                          layers[i].get_biases(),
                          layers[i].get_weight_gradients(),
                          layers[i].get_bias_gradients(),
                          learning_rate);
            }
            layers[i].zero_gradients();
        }
    }
    
    void train_sample(const std::vector<scalar_t>& input, const std::vector<scalar_t>& target){
        forward(input);
        backward(target);
        update_parameters();
    }
    
    // Train on a batch of samples
    void train_batch(const std::vector<std::vector<scalar_t>>& inputs,
                     const std::vector<std::vector<scalar_t>>& targets){
        if (inputs.empty()) return;
        
        size_t batch_size = inputs.size();
        size_t input_dim = inputs[0].size();
        size_t output_dim = targets[0].size();
        
        Matrix X_batch(batch_size, input_dim);
        Matrix y_batch(batch_size, output_dim);
        
        for (size_t i = 0; i < batch_size; i++) {
            std::copy(inputs[i].begin(), inputs[i].end(), 
                     X_batch.get_data().begin() + i * input_dim);
            std::copy(targets[i].begin(), targets[i].end(),
                     y_batch.get_data().begin() + i * output_dim);
        }
        
        train_batch_matrix(X_batch, y_batch);
    }
    
    // Train on a batch of samples given as matrices
    // Batch training using matrix operations for efficiency
    void train_batch_matrix(const Matrix& X_batch, const Matrix& y_batch){
        size_t batch_size = X_batch.get_rows();
        
        // Reset gradients for all layers
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].zero_gradients();
        }
        
        // Forward pass through all layers
        Matrix activation = X_batch;
        for (size_t i = 0; i < layers.size(); i++) {
            Matrix output;
            layers[i].forward_batch(activation, output);
            activation = output;
        }
        
        // Compute loss gradient (output - target)
        // This is the gradient of cross-entropy with respect to the softmax
        // inputs, so it assumes a softmax output layer with cross-entropy loss
        // (the softmax layer passes it through unchanged in backward_batch).
        Matrix loss_grad(batch_size, layers.back().get_output_size());
        auto& output_data = activation.get_data();
        auto& target_data = y_batch.get_data();
        auto& grad_data = loss_grad.get_data();
        
        for (size_t i = 0; i < output_data.size(); i++) {
            grad_data[i] = output_data[i] - target_data[i];
        }
        
        // Backward pass through all layers
        Matrix gradient = loss_grad;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Matrix input_grad;
            layers[i].backward_batch(gradient, input_grad);
            gradient = input_grad;
        }
        
        // Average gradients over batch
        scalar_t scale = 1.0f / batch_size;
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].get_weight_gradients().scale(scale);
            auto& bias_grads = layers[i].get_bias_gradients();
            for (size_t j = 0; j < bias_grads.size(); j++) {
                bias_grads[j] *= scale;
            }
        }
        
        update_parameters();
    }
    
    int predict(const std::vector<scalar_t>& input){
        std::vector<scalar_t> output = forward(input);
        return one_hot_decode(output);
    }
    
    scalar_t compute_loss(const std::vector<scalar_t>& predicted, const std::vector<scalar_t>& target){
        if (loss_type == MEAN_SQUARED_ERROR) {
            return mse_loss(predicted, target);
        } else if (loss_type == CROSS_ENTROPY) {
            return cross_entropy_loss(predicted, target);
        } else {
            throw std::invalid_argument("Unsupported loss type");
        }
    }
    
    std::vector<scalar_t> compute_loss_gradient(const std::vector<scalar_t>& predicted, 
                                               const std::vector<scalar_t>& target){
        if (loss_type == MEAN_SQUARED_ERROR) {
            return mse_gradient(predicted, target);
        } else if (loss_type == CROSS_ENTROPY) {
            return cross_entropy_gradient(predicted, target);
        } else {
            throw std::invalid_argument("Unsupported loss type");
        }
    }

    scalar_t evaluate_accuracy(const std::vector<std::vector<scalar_t>>& inputs,
                            const std::vector<std::vector<scalar_t>>& targets){
        size_t correct = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            int predicted_class = predict(inputs[i]);
            int true_class = one_hot_decode(targets[i]);
            if (predicted_class == true_class) {
                correct++;
            }
        }
        return static_cast<scalar_t>(correct) / inputs.size();
    }
    
    void set_learning_rate(scalar_t lr){
        learning_rate = lr;
        adam_config.learning_rate = lr;
        momentum_config.learning_rate = lr;
    }
    
    scalar_t get_learning_rate() const {
        return learning_rate;
    }
    
    size_t get_num_layers() const{
        return layers.size();
    }
};

inline scalar_t mse_loss(const std::vector<scalar_t>& predicted, const std::vector<scalar_t>& target){
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes must match for MSE loss");
    }
    scalar_t loss = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        scalar_t diff = predicted[i] - target[i];
        loss += 0.5 * diff * diff;
    }
    return loss;
}

inline std::vector<scalar_t> mse_gradient(const std::vector<scalar_t>& predicted, const std::vector<scalar_t>& target){
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes must match for MSE gradient");
    }
    std::vector<scalar_t> gradient(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        gradient[i] = predicted[i] - target[i];
    }
    return gradient;
}

inline scalar_t cross_entropy_loss(const std::vector<scalar_t>& predicted, const std::vector<scalar_t>& target){
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes must match for cross-entropy loss");
    }
    scalar_t loss = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        loss -= target[i] * std::log(predicted[i] + 1e-15);
    }
    return loss;
}

inline std::vector<scalar_t> cross_entropy_gradient(const std::vector<scalar_t>& predicted, 
                                          const std::vector<scalar_t>& target){
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes must match for cross-entropy gradient");
    }
    std::vector<scalar_t> gradient(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        gradient[i] = predicted[i] - target[i];
    }
    return gradient;
}

#endif
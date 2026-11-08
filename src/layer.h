#ifndef LAYER_H
#define LAYER_H

#include "types.h"
#include "matrix.h"
#include "activations.h"
#include <vector>
#include <stdexcept>
#include "vector_ops.h"

/**
 * Layer class - represents a single layer in the neural network
 * Handles forward pass, backward pass, and gradient computation for batches
 */
class Layer {
private:
    Matrix weights;
    std::vector<scalar_t> biases;
    size_t input_size;
    size_t output_size;
    ActivationType activation;
    
    Matrix last_input;
    Matrix last_z;
    Matrix last_activation;
    
    Matrix weight_gradients;
    std::vector<scalar_t> bias_gradients;

public:
    Layer(size_t input_size, size_t output_size, ActivationType activation_type)
    : weights(input_size, output_size), biases(output_size, 0.0), input_size(input_size),
      output_size(output_size), activation(activation_type),
      weight_gradients(input_size, output_size, 0.0), bias_gradients(output_size, 0.0) {}
    
    // Weight initialization methods
    void initialize_weights_random(scalar_t min_val, scalar_t max_val){
        weights.fill_random(min_val, max_val);
    }
    void initialize_weights_xavier(){
        weights.fill_xavier(output_size, input_size);
    }
    void initialize_weights_he(){
        weights.fill_he(output_size);
    }
    void initialize_biases(scalar_t value = 0.0){
        biases.assign(output_size, value);
    }

    /**
     * Forward pass for a batch of inputs
     * Computes: output = activation(input * weights + bias)
     */
    void forward_batch(const Matrix& input, Matrix& output){
        size_t batch_size = input.get_rows();
        last_input = input;
        
        // Z = input * weights + bias
        last_z = Matrix(batch_size, output_size);
        input.multiply_matrix(weights, last_z);
        last_z.add_bias(biases);
        
        // Apply activation function
        last_activation = Matrix(batch_size, output_size);
        auto& z_data = last_z.get_data();
        auto& a_data = last_activation.get_data();
        
        if (activation == RELU) {
            for (size_t i = 0; i < z_data.size(); i++) {
                a_data[i] = z_data[i] > 0 ? z_data[i] : 0;
            }
        } else if (activation == SOFTMAX) {
            for (size_t row = 0; row < batch_size; row++) {
                // Find max for numerical stability
                scalar_t max_val = z_data[row * output_size];
                for (size_t j = 1; j < output_size; j++) {
                    if (z_data[row * output_size + j] > max_val) {
                        max_val = z_data[row * output_size + j];
                    }
                }
                scalar_t sum = 0.0f;
                for (size_t j = 0; j < output_size; j++) {
                    a_data[row * output_size + j] = std::exp(z_data[row * output_size + j] - max_val);
                    sum += a_data[row * output_size + j];
                }
                for (size_t j = 0; j < output_size; j++) {
                    a_data[row * output_size + j] /= sum;
                }
            }
        } else {
            a_data = z_data;
        }
        
        output = last_activation;
    }
    
    void backward_batch(const Matrix& output_gradient, Matrix& input_gradient){
        size_t batch_size = last_input.get_rows();
        
        // Compute delta = activation'(z) * output_gradient
        Matrix delta(batch_size, output_size);
        auto& delta_data = delta.get_data();
        auto& grad_data = output_gradient.get_data();
        auto& z_data = last_z.get_data();
        
        if (activation == SOFTMAX) {
            delta_data = grad_data;
        } else if (activation == RELU) {
            for (size_t i = 0; i < delta_data.size(); i++) {
                delta_data[i] = (z_data[i] > 0) ? grad_data[i] : 0;
            }
        } else {
            delta_data = grad_data;
        }
        
        // Weight gradients: input^T * delta
        Matrix input_T = last_input.transpose();
        Matrix grad_update(input_size, output_size);
        input_T.multiply_matrix(delta, grad_update);
        weight_gradients.add(grad_update);
        
        // Bias gradients: sum delta across batch
        for (size_t row = 0; row < batch_size; row++) {
            for (size_t j = 0; j < output_size; j++) {
                bias_gradients[j] += delta_data[row * output_size + j];
            }
        }
        
        // Input gradient: delta * weights^T
        Matrix weights_T = weights.transpose();
        input_gradient = Matrix(batch_size, input_size);
        delta.multiply_matrix(weights_T, input_gradient);
    }
    
    std::vector<scalar_t> forward(const std::vector<scalar_t>& input){
        Matrix input_batch(1, input_size);
        std::copy(input.begin(), input.end(), input_batch.get_data().begin());
        Matrix output_batch;
        forward_batch(input_batch, output_batch);
        
        std::vector<scalar_t> result(output_size);
        std::copy(output_batch.get_data().begin(), 
                  output_batch.get_data().begin() + output_size, 
                  result.begin());
        return result;
    }
    
    std::vector<scalar_t> backward(const std::vector<scalar_t>& output_gradient){
        Matrix grad_batch(1, output_size);
        std::copy(output_gradient.begin(), output_gradient.end(), grad_batch.get_data().begin());
        Matrix input_grad_batch;
        backward_batch(grad_batch, input_grad_batch);
        
        std::vector<scalar_t> result(input_size);
        std::copy(input_grad_batch.get_data().begin(),
                  input_grad_batch.get_data().begin() + input_size,
                  result.begin());
        return result;
    }
    
    void update_parameters(scalar_t learning_rate){
        weight_gradients.scale(-learning_rate);
        weights.add(weight_gradients);
        biases = vector_add(biases, vector_scale(bias_gradients, -learning_rate));
        zero_gradients();
    }
    
    void zero_gradients(){
        weight_gradients.fill(0.0);
        bias_gradients.assign(output_size, 0.0);
    }
    
    const Matrix& get_last_input() const{
        return last_input;
    }
    const Matrix& get_last_z() const{
        return last_z;
    }
    const Matrix& get_last_activation() const{
        return last_activation;
    }

    Matrix& get_weights(){
        return weights;
    }
    std::vector<scalar_t>& get_biases(){
        return biases;
    }
    Matrix& get_weight_gradients(){
        return weight_gradients;
    }
    std::vector<scalar_t>& get_bias_gradients(){
        return bias_gradients;
    }

    size_t get_input_size() const{
        return input_size;
    }
    size_t get_output_size() const{
        return output_size;
    }
};

#endif
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "types.h"

/**
 * Activation functions and their derivatives
 * Includes Sigmoid, ReLU, Tanh, Linear, and Softmax
 * Applies to both single values and vectors
 */

enum ActivationType {
    SIGMOID,
    RELU,
    TANH,
    LINEAR,
    SOFTMAX
};

inline scalar_t sigmoid(scalar_t x){
    return 1.0f / (1.0f + std::exp(-x));
}
inline scalar_t sigmoid_derivative(scalar_t x){
    scalar_t sig = sigmoid(x);
    return sig * (1.0f - sig);
}

inline scalar_t relu(scalar_t x){
    return std::max(0.0f, x);
}
inline scalar_t relu_derivative(scalar_t x){
    return x > 0 ? 1.0f : 0.0f;
}

inline scalar_t tanh_activation(scalar_t x){
    return std::tanh(x);
}
inline scalar_t tanh_derivative(scalar_t x){
    scalar_t tanh_val = tanh_activation(x);
    return 1.0f - tanh_val * tanh_val;
}

inline scalar_t linear(scalar_t x){
    return x;
}
inline scalar_t linear_derivative(scalar_t x){
    return 1.0f;
}

inline std::vector<scalar_t> softmax(const std::vector<scalar_t>& input){
    std::vector<scalar_t> output(input.size());
    scalar_t max_input = *std::max_element(input.begin(), input.end());
    scalar_t sum_exp = 0.0f;
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - max_input);
        sum_exp += output[i];
    }
    for (size_t i = 0; i < input.size(); i++) {
        output[i] /= sum_exp;
    }
    return output;
}

// Derivative of softmax is more complex; here we provide a simplified version
// that computes the derivative assuming the output is used in cross-entropy loss.
inline std::vector<scalar_t> softmax_derivative(const std::vector<scalar_t>& softmax_output){
    std::vector<scalar_t> output(softmax_output.size());
    for (size_t i = 0; i < softmax_output.size(); i++) {
        output[i] = softmax_output[i] * (1.0f - softmax_output[i]);
    }
    return output;
}


scalar_t activate_single(scalar_t x, ActivationType type){
    switch (type){
        case SIGMOID:
            return sigmoid(x);
            break;
        case RELU:
            return relu(x);
            break;
        case TANH:
            return tanh_activation(x);
            break;
        case LINEAR:
            return linear(x);
            break;
        case SOFTMAX:
            throw std::invalid_argument("Softmax activation should be applied to vectors");
            break;
        default:
            throw std::invalid_argument("Unknown activation type"); 
    }
}

scalar_t activate_derivative_single(scalar_t x, ActivationType type){
    switch (type){
        case SIGMOID:
            return sigmoid_derivative(x);
            break;
        case RELU:
            return relu_derivative(x);
            break;
        case TANH:
            return tanh_derivative(x);
            break;
        case LINEAR:
            return linear_derivative(x);
            break;
        case SOFTMAX:
            throw std::invalid_argument("Softmax derivative should be applied to vectors");
            break;
        default:
            throw std::invalid_argument("Unknown activation type"); 
    }
}

std::vector<scalar_t> activate_vector(const std::vector<scalar_t>& input, ActivationType type){
    if (type == SOFTMAX) {
        return softmax(input);
    }
    std::vector<scalar_t> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = activate_single(input[i], type);
    }
    return output;
}

std::vector<scalar_t> activate_derivative_vector(const std::vector<scalar_t>& input, ActivationType type){
    if (type == SOFTMAX){
        return softmax_derivative(input);
    }
    std::vector<scalar_t> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = activate_derivative_single(input[i], type);
    }
    return output;

}

#endif
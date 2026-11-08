#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"
#include <vector>
#include <stdexcept>

/**
 * Optimizer configuration structures and update functions
 * Implements SGD, SGD with Momentum, and Adam optimizers
 */

struct OptimizerConfig {
    scalar_t learning_rate;
    
    OptimizerConfig(scalar_t lr = 0.01) : learning_rate(lr) {}
};

struct MomentumConfig : public OptimizerConfig {
    scalar_t momentum;       // Velocity decay factor
    scalar_t weight_decay;   // L2 regularization strength
    
    MomentumConfig(scalar_t lr = 0.01, scalar_t mom = 0.9, scalar_t wd = 0.0)
        : OptimizerConfig(lr), momentum(mom), weight_decay(wd) {}
};

struct AdamConfig : public OptimizerConfig {
    scalar_t beta1;          // Exponential decay rate for first moment (mean)
    scalar_t beta2;          // Exponential decay rate for second moment (variance)
    scalar_t epsilon;        // Small constant to prevent division by zero
    scalar_t weight_decay;   // L2 regularization strength
    
    AdamConfig(scalar_t lr = 0.001, scalar_t b1 = 0.9, scalar_t b2 = 0.999, 
               scalar_t eps = 1e-8, scalar_t wd = 0.0)
        : OptimizerConfig(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd) {}
};

struct MomentumState {
    Matrix weight_velocity;              // Accumulated velocity for weights
    std::vector<scalar_t> bias_velocity; // Accumulated velocity for biases
    
    void initialize(size_t rows, size_t cols) {
        weight_velocity = Matrix(rows, cols, 0.0);
        bias_velocity.resize(rows, 0.0);
    }
};

struct AdamState {
    Matrix weight_m;                 // First moment estimate (mean) for weights
    Matrix weight_v;                 // Second moment estimate (variance) for weights
    std::vector<scalar_t> bias_m;    // First moment estimate for biases
    std::vector<scalar_t> bias_v;    // Second moment estimate for biases
    int timestep;                    // Current optimization step (for bias correction)
    
    AdamState() : timestep(0) {}
    
    void initialize(size_t rows, size_t cols) {
        weight_m = Matrix(rows, cols, 0.0);
        weight_v = Matrix(rows, cols, 0.0);
        bias_m.resize(rows, 0.0);
        bias_v.resize(rows, 0.0);
        timestep = 0;
    }
};

// Basic SGD: w = w - lr * grad
inline void sgd_update(Matrix& weights, 
                std::vector<scalar_t>& biases,
                const Matrix& weight_gradients,
                const std::vector<scalar_t>& bias_gradients,
                scalar_t learning_rate) {
    for (size_t i = 0; i < weights.get_rows(); i++) {
        for (size_t j = 0; j < weights.get_cols(); j++) {
            weights.at(i, j) -= learning_rate * weight_gradients.at(i, j);
        }
    }
    
    for (size_t i = 0; i < biases.size(); i++) {
        biases[i] -= learning_rate * bias_gradients[i];
    }
}

// SGD with Momentum: v = momentum * v - lr * grad; w = w + v
inline void momentum_update(Matrix& weights,
                    std::vector<scalar_t>& biases,
                    const Matrix& weight_gradients,
                    const std::vector<scalar_t>& bias_gradients,
                    MomentumState& state,
                    const MomentumConfig& config) {
    // Initialize state if needed
    if (state.weight_velocity.get_rows() != weights.get_rows() ||
        state.weight_velocity.get_cols() != weights.get_cols()) {
        state.initialize(weights.get_rows(), weights.get_cols());
    }
    
    for (size_t i = 0; i < weights.get_rows(); i++) {
        for (size_t j = 0; j < weights.get_cols(); j++) {
            // Update velocity: v = momentum * v - lr * grad
            state.weight_velocity.at(i, j) = config.momentum * state.weight_velocity.at(i, j) 
                                             - config.learning_rate * weight_gradients.at(i, j);
            // Update weight using velocity
            weights.at(i, j) += state.weight_velocity.at(i, j);
            
            // Apply L2 regularization
            if (config.weight_decay > 0.0) {
                weights.at(i, j) *= (1.0 - config.weight_decay);
            }
        }
    }
    
    for (size_t i = 0; i < biases.size(); i++) {
        state.bias_velocity[i] = config.momentum * state.bias_velocity[i] 
                                - config.learning_rate * bias_gradients[i];
        biases[i] += state.bias_velocity[i];
    }
}

// Adam optimizer: adaptive learning rates per parameter
inline void adam_update(Matrix& weights,
                std::vector<scalar_t>& biases,
                const Matrix& weight_gradients,
                const std::vector<scalar_t>& bias_gradients,
                AdamState& state,
                const AdamConfig& config) {
    // Initialize state if needed
    if (state.weight_m.get_rows() != weights.get_rows() ||
        state.weight_m.get_cols() != weights.get_cols()) {
        state.initialize(weights.get_rows(), weights.get_cols());
    }
    
    state.timestep++;
    
    // Bias correction factors for first and second moments
    scalar_t bias_correction1 = 1.0 - std::pow(config.beta1, state.timestep);
    scalar_t bias_correction2 = 1.0 - std::pow(config.beta2, state.timestep);
    
    auto& w_data = weights.get_data();
    auto& g_data = weight_gradients.get_data();
    auto& m_data = state.weight_m.get_data();
    auto& v_data = state.weight_v.get_data();
    
    for (size_t i = 0; i < w_data.size(); i++) {
        scalar_t grad = g_data[i];
        
        // Update first moment (exponential moving average of gradients)
        m_data[i] = config.beta1 * m_data[i] + (1.0 - config.beta1) * grad;
        
        // Update second moment (exponential moving average of squared gradients)
        v_data[i] = config.beta2 * v_data[i] + (1.0 - config.beta2) * grad * grad;
        
        // Bias-corrected estimates
        scalar_t m_hat = m_data[i] / bias_correction1;
        scalar_t v_hat = v_data[i] / bias_correction2;
        
        // Update parameter: w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
        w_data[i] -= config.learning_rate * m_hat / (std::sqrt(v_hat) + config.epsilon);
        
        // Apply L2 regularization
        if (config.weight_decay > 0.0) {
            w_data[i] *= (1.0 - config.weight_decay);
        }
    }
    
    // Same process for biases
    for (size_t i = 0; i < biases.size(); i++) {
        scalar_t grad = bias_gradients[i];
        
        state.bias_m[i] = config.beta1 * state.bias_m[i] + (1.0 - config.beta1) * grad;
        state.bias_v[i] = config.beta2 * state.bias_v[i] + (1.0 - config.beta2) * grad * grad;
        
        scalar_t m_hat = state.bias_m[i] / bias_correction1;
        scalar_t v_hat = state.bias_v[i] / bias_correction2;
        
        biases[i] -= config.learning_rate * m_hat / (std::sqrt(v_hat) + config.epsilon);
    }
}

#endif
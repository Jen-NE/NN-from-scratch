#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "types.h"

class Matrix {
private:
    std::vector<scalar_t> data;
    size_t rows;
    size_t cols;

public:
    Matrix() : rows(0), cols(0) {}
    Matrix(size_t rows, size_t cols): rows(rows), cols(cols) {
        data.resize(rows * cols, 0.0f);
    }
    Matrix(size_t rows, size_t cols, scalar_t initial_value): rows(rows), cols(cols) {
        data.resize(rows * cols, initial_value);
    }
    
    inline scalar_t& at(size_t row, size_t col){
        return data[row * cols + col];
    }
    inline const scalar_t& at(size_t row, size_t col) const{
        return data[row * cols + col];
    }
    inline size_t get_rows() const{
        return rows;
    }
    inline size_t get_cols() const{
        return cols;
    }
    inline size_t size() const{
        return data.size();
    }

    // Multiply matrix by vector: result = this * vec
    void multiply(const std::vector<scalar_t>& vec, std::vector<scalar_t>& result) const{
        std::fill(result.begin(), result.end(), 0.0f);
        for (size_t i = 0; i < rows; i++) {
            scalar_t sum = 0.0f;
            const scalar_t* row_data = &data[i * cols];
            for (size_t j = 0; j < cols; j++) {
                sum += row_data[j] * vec[j];
            }
            result[i] = sum;
        }
    }

    // Multiply matrix by matrix: result = this * weights   
    void multiply_matrix(const Matrix& weights, Matrix& result) const {
        const size_t batch_size = rows;
        const size_t input_dim = cols;
        const size_t output_dim = weights.get_cols();
        
        std::fill(result.data.begin(), result.data.end(), 0.0f);
        
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t k = 0; k < input_dim; k++) {
                const scalar_t a_ik = data[i * input_dim + k];
                const scalar_t* w_row = &weights.data[k * output_dim];
                scalar_t* r_row = &result.data[i * output_dim];
                for (size_t j = 0; j < output_dim; j++) {
                    r_row[j] += a_ik * w_row[j];
                }
            }
        }
    }
    
    Matrix transpose() const{
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[j * rows + i] = data[i * cols + j];
            }
        }
        return result;
    }
    
    inline void scale(scalar_t factor){
        for (auto& val : data) {
            val *= factor;
        }
    }

    inline void add(const Matrix& other){
        for (size_t i = 0; i < data.size(); i++) {
            data[i] += other.data[i];
        }
    }
    
    inline void add_bias(const std::vector<scalar_t>& bias){
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                data[i * cols + j] += bias[j];
            }
        }
    }

    inline void fill(scalar_t value){
        std::fill(data.begin(), data.end(), value);
    }

    // Fill matrix with random values in [min_val, max_val]
    inline void fill_random(scalar_t min_val, scalar_t max_val){
        for (auto& val : data) {
            val = min_val + static_cast<scalar_t>(rand()) / RAND_MAX * (max_val - min_val);
        }
    }

    void fill_xavier(size_t fan_in, size_t fan_out){
        scalar_t limit = std::sqrt(6.0f / (fan_in + fan_out));
        for (auto& val : data) {
            val = -limit + static_cast<scalar_t>(rand()) / RAND_MAX * (2.0f * limit);
        }
    }
    
    void fill_he(size_t fan_in){
        scalar_t stddev = std::sqrt(2.0f / fan_in);
        for (auto& val : data) {
            scalar_t u1 = static_cast<scalar_t>(rand()) / RAND_MAX;
            scalar_t u2 = static_cast<scalar_t>(rand()) / RAND_MAX;
            val = stddev * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        }
    }
    
    inline std::vector<scalar_t>& get_data(){
        return data;
    }
    inline const std::vector<scalar_t>& get_data() const{
        return data;
    }
};

#endif
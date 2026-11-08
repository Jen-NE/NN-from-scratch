#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include "types.h"
#include "matrix.h"

inline std::vector<scalar_t> vector_add(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b){
    std::vector<scalar_t> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

inline std::vector<scalar_t> vector_subtract(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b){
    std::vector<scalar_t> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

inline std::vector<scalar_t> vector_scale(const std::vector<scalar_t>& vec, scalar_t scalar){
    std::vector<scalar_t> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * scalar;
    }
    return result;
}

inline std::vector<scalar_t> vector_mul(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b){
    std::vector<scalar_t> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

inline scalar_t vector_dot(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b){
    scalar_t result = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

inline Matrix vector_outer_product(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b){
    Matrix result(a.size(), b.size());
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b.size(); j++) {
            result.at(i, j) = a[i] * b[j];
        }
    }
    return result;
}

inline scalar_t vector_norm(const std::vector<scalar_t>& vec){
    scalar_t sum_squares = 0.0f;
    for (const auto& val : vec) {
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}

inline std::vector<scalar_t> zeros(size_t size){
    return std::vector<scalar_t>(size, 0.0f);
}

inline std::vector<scalar_t> fill_vector(size_t size, scalar_t value){
    return std::vector<scalar_t>(size, value);
}

#endif
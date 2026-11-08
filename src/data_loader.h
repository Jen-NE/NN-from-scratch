#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <random>

// Utility functions for label encoding
std::vector<scalar_t> one_hot_encode(int label, size_t num_classes);
int one_hot_decode(const std::vector<scalar_t>& one_hot);
void normalize_features(std::vector<std::vector<scalar_t>>& features);
std::pair<scalar_t, scalar_t> normalize_features_meanstd(std::vector<std::vector<scalar_t>>& features);

/**
 * DataLoader class - loads and preprocesses Fashion-MNIST dataset
 * Handles CSV loading, normalization, one-hot encoding of labels, and batching
 */
class DataLoader {
private:
    std::vector<std::vector<scalar_t>> features;
    std::vector<std::vector<scalar_t>> labels;
    size_t current_index;

public:
    DataLoader() : current_index(0) {}
    
    bool load_data(const std::string& features_file, 
                   const std::string& labels_file,
                   bool normalize = true) {
        features.clear();
        labels.clear();
        
        std::ifstream features_fstream(features_file);
        std::ifstream labels_fstream(labels_file);
        
        if (!features_fstream.is_open() || !labels_fstream.is_open()) {
            std::cerr << "Error: Cannot open data files" << std::endl;
            return false;
        }
        
     std::string features_line, labels_line;

     while (std::getline(features_fstream, features_line) && 
         std::getline(labels_fstream, labels_line)) {
            
            if (features_line.empty() || labels_line.empty()) continue;
            
            std::vector<scalar_t> feature_vec;
            feature_vec.reserve(784);
            std::string token;
            size_t pos = 0;
            while (pos < features_line.length()) {
                size_t comma_pos = features_line.find(',', pos);
                if (comma_pos == std::string::npos) {
                    token = features_line.substr(pos);
                    feature_vec.push_back(std::stof(token));
                    break;
                } else {
                    token = features_line.substr(pos, comma_pos - pos);
                    feature_vec.push_back(std::stof(token));
                    pos = comma_pos + 1;
                }
            }
            
            int label_value = std::stoi(labels_line);
            std::vector<scalar_t> label_vec = one_hot_encode(label_value, 10);
            
            features.push_back(feature_vec);
            labels.push_back(label_vec);
        }
        
        features_fstream.close();
        labels_fstream.close();
        
        if (normalize) {
            normalize_features_meanstd(features);
        }
        
        std::cout << "Loaded " << features.size() << " samples" << std::endl;
        return features.size() > 0;
    }

    const std::vector<std::vector<scalar_t>>& get_features() const {
        return features;
    }
    
    const std::vector<std::vector<scalar_t>>& get_labels() const {
        return labels;
    }
    
    bool get_batch(std::vector<std::vector<scalar_t>>& batch_features,
                   std::vector<std::vector<scalar_t>>& batch_labels,
                   size_t batch_size) {
        batch_features.clear();
        batch_labels.clear();
        
        if (current_index >= features.size()) {
            return false;
        }
        
        size_t end_index = std::min(current_index + batch_size, features.size());
        
        for (size_t i = current_index; i < end_index; i++) {
            batch_features.push_back(features[i]);
            batch_labels.push_back(labels[i]);
        }
        
        current_index = end_index;
        return true;
    }

    void reset(){
        current_index = 0;
    }
    
    size_t size() const{
        return features.size();
    }
    
    static bool save_predictions(const std::string& filename, 
                                const std::vector<int>& predictions){
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        for (const auto& pred : predictions) {
            file << pred << "\n";
        }
        return true;
    }
};

inline std::vector<scalar_t> one_hot_encode(int label, size_t num_classes = 10){
    std::vector<scalar_t> one_hot(num_classes, 0.0);
    if (label >= 0 && static_cast<size_t>(label) < num_classes) {
        one_hot[label] = 1.0;
    }
    return one_hot;
}

inline int one_hot_decode(const std::vector<scalar_t>& one_hot){
    int max_class = 0;
    scalar_t max_value = one_hot[0];
    for (size_t i = 1; i < one_hot.size(); i++) {
        if (one_hot[i] > max_value) {
            max_value = one_hot[i];
            max_class = i;
        }
    }
    return max_class;
}

inline void normalize_features(std::vector<std::vector<scalar_t>>& features){
    for (auto& sample : features) {
        for (auto& pixel : sample) {
            pixel /= 255.0f;
        }
    }
}

// Normalize to zero mean and unit variance
inline std::pair<scalar_t, scalar_t> normalize_features_meanstd(std::vector<std::vector<scalar_t>>& features){
    if (features.empty() || features[0].empty()) {
        return {0.0f, 1.0f};
    }
    
    for (auto& sample : features) {
        for (auto& pixel : sample) {
            pixel /= 255.0f;
        }
    }
    
    scalar_t mean = 0.0f;
    size_t total_pixels = features.size() * features[0].size();
    for (const auto& sample : features) {
        for (const auto& pixel : sample) {
            mean += pixel;
        }
    }
    mean /= total_pixels;
    
    for (auto& sample : features) {
        for (auto& pixel : sample) {
            pixel -= mean;
        }
    }
    
    scalar_t variance = 0.0f;
    for (const auto& sample : features) {
        for (const auto& pixel : sample) {
            variance += pixel * pixel;
        }
    }
    variance /= (total_pixels - 1);
    scalar_t std = std::sqrt(variance);
    
    for (auto& sample : features) {
        for (auto& pixel : sample) {
            pixel /= std;
        }
    }
    
    return {mean, std};
}


#endif
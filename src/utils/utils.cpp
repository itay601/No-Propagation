#include "utils/utils.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <filesystem>


// Generate a simple synthetic dataset (for testing/fallback)
Eigen::MatrixXd generateSyntheticData(int numSamples, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(numSamples, dim);
    
    for (int i = 0; i < numSamples; i++) {
        // Generate points on a circle plus noise
        double theta = 2.0 * M_PI * i / numSamples;
        data(i, 0) = std::cos(theta) + 0.1 * distribution(gen);
        data(i, 1) = std::sin(theta) + 0.1 * distribution(gen);
    }
    
    return data;
}

// Create a directory if it doesn't exist
void createDirectory(const std::string& path) {
    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
}
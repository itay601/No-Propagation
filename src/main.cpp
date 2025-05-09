#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include "neural_network/layer.hpp"
#include "neural_network/activation.hpp"
#include "diffusion/diffusion_model.hpp"

// GPT-3.5 helper //
// Generate a simple synthetic dataset
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

int main() {
    // Parameters
    const int inputDim = 2;           // 2D data points
    const int latentDim = 2;          // Same dimension for simplicity
    const int hiddenDim = 64;         // Size of hidden layers
    const int timesteps = 100;        // Number of diffusion steps
    const int batchSize = 32;         // Batch size for training
    const int numEpochs = 1000;       // Number of training epochs
    const double learningRate = 0.001; // Learning rate
    
    std::cout << "Generating synthetic data..." << std::endl;
    Eigen::MatrixXd trainingData = generateSyntheticData(1000, inputDim);
    
    std::cout << "Creating diffusion model..." << std::endl;
    DiffusionModel model(inputDim, latentDim, hiddenDim, timesteps);
    
    std::cout << "Training diffusion model..." << std::endl;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // Create a random batch of data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distribution(0, trainingData.rows() - 1);
        
        Eigen::MatrixXd batch(batchSize, inputDim);
        for (int i = 0; i < batchSize; i++) {
            int idx = distribution(gen);
            batch.row(i) = trainingData.row(idx);
        }
        
        // Perform a training step
        double loss = model.trainStep(batch, learningRate);
        
        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
    
    std::cout << "Sampling from trained model..." << std::endl;
    Eigen::MatrixXd samples = model.sample(100);
    
    std::cout << "First 5 generated samples:" << std::endl;
    std::cout << samples.topRows(5) << std::endl;
    
    return 0;
}
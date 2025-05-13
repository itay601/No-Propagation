#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <string>
#include <filesystem>
#include "neural_network/layer.hpp"
#include "neural_network/activation.hpp"
#include "diffusion/diffusion_model.hpp"
#include "utils/data_loader.hpp"

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

int main(int argc, char** argv) {
    // Check for data file argument
    std::string dataFile = "/home/opc/development-envierment/No-Propagation/src/data/cifar10_data.bin";
    //std::string dataFile = "data/cifar10_data.bin";
    if (argc > 1) {
        dataFile = argv[1];
    }
    
    // Parameters for the model
    const int batchSize = 32;          // Batch size for training
    const int numEpochs = 1000;        // Number of training epochs
    const double learningRate = 0.001; // Learning rate
    const int timesteps = 100;         // Number of diffusion steps
    const int hiddenDim = 256;         // Size of hidden layers
    
    // Try to load CIFAR-10 data
    ImageDataLoader dataLoader;
    bool useSyntheticData = false;
    int inputDim = 0;
    
    if (std::filesystem::exists(dataFile)) {
        std::cout << "Loading CIFAR-10 data from " << dataFile << "..." << std::endl;
        if (dataLoader.loadCIFAR10Binary(dataFile)) {
            inputDim = dataLoader.getImageSize();
            std::cout << "Successfully loaded CIFAR-10 data with dimension: " << inputDim << "\n" << std::endl;
        } else {
            std::cout << "Failed to load CIFAR-10 data. Falling back to synthetic data." << std::endl;
            useSyntheticData = true;
        }
    } else {
        std::cout << "Data file not found: " << dataFile << std::endl;
        std::cout << "Falling back to synthetic data." << std::endl;
        useSyntheticData = true;
    }
    
    // Use synthetic data if needed
    Eigen::MatrixXd trainingData;
    if (useSyntheticData) {
        inputDim = 2;  // 2D synthetic data
        std::cout << "Generating synthetic data..." << std::endl;
        trainingData = generateSyntheticData(1000, inputDim);
    }
    
    // Create the diffusion model
    std::cout << "Creating diffusion model..." << std::endl;
    DiffusionModel model(inputDim, inputDim, hiddenDim, timesteps);
    
    // Create output directory for samples
    createDirectory("output");
    
    // Training loop
    std::cout << "Training diffusion model..." << std::endl;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        Eigen::MatrixXd batch;
        
        // Get training batch
        if (useSyntheticData) {
            // Create a random batch from synthetic data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> distribution(0, trainingData.rows() - 1);
            
            batch = Eigen::MatrixXd(batchSize, inputDim);
            for (int i = 0; i < batchSize; i++) {
                int idx = distribution(gen);
                batch.row(i) = trainingData.row(idx);
            }
        } else {
            // Get batch from data loader
            batch = dataLoader.getBatch(batchSize);
        }
        
        // Perform a training step
        double loss = model.trainStep(batch, learningRate);
        
        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            
            // Generate samples every 100 epochs and save them
            if (!useSyntheticData) {
                // Sample from the model
                Eigen::MatrixXd samples = model.sample(5);
                
                // Save samples as PPM images
                for (int i = 0; i < samples.rows(); i++) {
                    std::string filename = "output/sample_epoch_" + std::to_string(epoch) + "_" + std::to_string(i) + ".json";
                    if (dataLoader.saveTensorAsJSON(samples.row(i), filename)) {
                        std::cout << "Saved sample to " << filename << std::endl;
                    }
                }
            }
        }
    }
    
    // Generate final samples
    std::cout << "Sampling from trained model..." << std::endl;
    int numSamples = 10;
    Eigen::MatrixXd samples = model.sample(numSamples);
    
    if (useSyntheticData) {
        // Print first 5 generated samples for synthetic data
        std::cout << "First 5 generated samples:" << std::endl;
        std::cout << samples.topRows(5) << std::endl;
    } else {
        // Save samples as PPM images
        for (int i = 0; i < samples.rows(); i++) {
            std::string filename = "output/test_epoch_" + std::to_string(i) + "_" + std::to_string(i) + ".json";
            if (dataLoader.saveTensorAsJSON(samples.row(i), filename)) {
                std::cout << "Saved sample to " << filename << std::endl;
            }
        }
    }
    
    return 0;
}
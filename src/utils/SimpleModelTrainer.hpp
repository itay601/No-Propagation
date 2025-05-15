#ifndef DIFFUSION_TRAINER_HPP
#define DIFFUSION_TRAINER_HPP

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
#include "utils/utils.hpp"

// The DiffusionTrainer class encapsulates the entire training and sampling process.
class SimpleModelTrainer {
public:
    SimpleModelTrainer(int argc, char** argv)
        : batchSize(32),
          numEpochs(1000),
          learningRate(0.001),
          timesteps(100),
          hiddenDim(256),
          useSyntheticData(false),
          inputDim(0)
    {
        // Set default data file path. Override it if an argument is provided.
        dataFile = "/home/opc/development-envierment/No-Propagation/src/data/cifar10_data.bin";
        if (argc > 1) {
            dataFile = argv[1];
        }
    }
    
    // The run method performs data loading, model initialization, training, and sampling.
    void run() {
        // Attempt to load CIFAR-10 data.
        std::cout << "Attempting to load data from: " << dataFile << std::endl;
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
        
        // Use synthetic data if necessary.
        if (useSyntheticData) {
            inputDim = 2;  // Using 2D synthetic data.
            std::cout << "Generating synthetic data..." << std::endl;
            trainingData = generateSyntheticData(1000, inputDim);
        }
        
        // Create the diffusion model.
        std::cout << "Creating diffusion model..." << std::endl;
        model = std::make_unique<DiffusionModel>(inputDim, inputDim, hiddenDim, timesteps);
        model->printNetworkStructure(*model);
        
        // Create the output directory.
        createDirectory("output");
        
        // Training loop.
        std::cout << "Training diffusion model..." << std::endl;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            Eigen::MatrixXd batch;
            
            // Prepare a training batch.
            if (useSyntheticData) {
                batch = Eigen::MatrixXd(batchSize, inputDim);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> distribution(0, trainingData.rows() - 1);
                
                for (int i = 0; i < batchSize; i++) {
                    int idx = distribution(gen);
                    batch.row(i) = trainingData.row(idx);
                }
            } else {
                batch = dataLoader.getBatch(batchSize);
            }
            
            // Perform a training step.
            double loss = model->trainStep(batch, learningRate);
            
            // Print progress every 100 epochs.
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
                
                // When working with CIFAR-10 data, you might want to sample and save generated images.
                if (!useSyntheticData) {
                    // Example (commented out):
                    // Eigen::MatrixXd samples = model->sample(5);
                    // for (int i = 0; i < samples.rows(); i++) {
                    //     std::string filename = "output/sample_epoch_" + std::to_string(epoch) + "_" + std::to_string(i) + ".json";
                    //     if (dataLoader.saveTensorAsJSON(samples.row(i), filename)) {
                    //         std::cout << "Saved sample to " << filename << std::endl;
                    //     }
                    // }
                }
            }
        }
        
        // Final sampling after training.
        std::cout << "Sampling from trained model..." << std::endl;
        int numSamples = 10;
        Eigen::MatrixXd samples = model->sample(numSamples);
        
        if (useSyntheticData) {
            std::cout << "First 5 generated samples:" << std::endl;
            std::cout << samples.topRows(5) << std::endl;
        } else {
            // Save or process samples as needed for CIFAR-10.
            // Example (commented out):
            // for (int i = 0; i < samples.rows(); i++) {
            //     std::string filename = "output/test_epoch_" + std::to_string(i) + "_" + std::to_string(i) + ".json";
            //     if (dataLoader.saveTensorAsJSON(samples.row(i), filename)) {
            //         std::cout << "Saved sample to " << filename << std::endl;
            //     }
            // }
        }
    }

private:
    // Parameters and file paths.
    std::string dataFile;
    const int batchSize;
    const int numEpochs;
    const double learningRate;
    const int timesteps;
    const int hiddenDim;
    bool useSyntheticData;
    int inputDim;
    
    // Members for handling data and the diffusion model.
    ImageDataLoader dataLoader;
    Eigen::MatrixXd trainingData;
    std::unique_ptr<DiffusionModel> model;
};

#endif // DIFFUSION_TRAINER_HPP
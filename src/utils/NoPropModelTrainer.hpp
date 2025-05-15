#ifndef NOPROP_TRAINER_HPP
#define NOPROP_TRAINER_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <string>
#include "neural_network/activation.hpp"
#include "diffusion/diffusion_model_noProp.hpp"
#include "utils/data_loader.hpp" 


class NoPropModelTrainer {
public:
    NoPropModelTrainer(int argc, char** argv)
        : inputDim(3072),          // Dimension of input data
          latentDim(3072),         // Dimension of latent space (same as input for simplicity)
          hiddenDim(1000),         // Hidden layer dimension
          timesteps(50),         // Number of diffusion timesteps
          perturbStrength(0.001), // Strength of weight perturbations
          updateFreq(10),        // Frequency of perturbation updates
          batchSize(32),         // Batch size
          numEpochs(1000),       // Number of training epochs
          learningRate(0.001),   // Learning rate
          sampleFrequency(100),  // How often to generate samples
          useSyntheticData(false) // Whether to use synthetic data
    {
        // Set default data file path. Override it if an argument is provided.
        dataFile = "/home/opc/development-envierment/No-Propagation/src/data/cifar10_data.bin";
        if (argc > 1) {
            dataFile = argv[1];
        }
    }
    
    void run() {
        // Print configuration
        printConfiguration();

        // Create NoProp diffusion model
        model = std::make_unique<DiffusionModelNoProp>(
            inputDim, latentDim, hiddenDim, timesteps, perturbStrength, updateFreq);
        
        // Load or generate training data
        prepareTrainingData();
        
        // Create output directory
        createOutputDirectory();
        
        // Training loop
        trainModel();
        
        // Generate final samples
        generateFinalSamples();
    }

private:
    // Configuration parameters
    int inputDim;
    int latentDim;
    int hiddenDim;
    int timesteps;
    double perturbStrength;
    int updateFreq;
    int batchSize;
    int numEpochs;
    double learningRate;
    int sampleFrequency;
    bool useSyntheticData;
    std::string dataFile;
    
    // Model and data
    std::unique_ptr<DiffusionModelNoProp> model;
    Eigen::MatrixXd trainingData;
    ImageDataLoader dataLoader; // Assuming you have this from the other code
    
    void printConfiguration() {
        std::cout << "NoProp Diffusion Model Configuration:" << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Input dimension: " << inputDim << std::endl;
        std::cout << "Hidden dimension: " << hiddenDim << std::endl;
        std::cout << "Timesteps: " << timesteps << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Number of epochs: " << numEpochs << std::endl;
        std::cout << "Learning rate: " << learningRate << std::endl;
        std::cout << "Perturbation strength: " << perturbStrength << std::endl;
        std::cout << "Update frequency: " << updateFreq << std::endl;
        std::cout << "Using synthetic data: " << (useSyntheticData ? "Yes" : "No") << std::endl;
        if (!useSyntheticData) {
            std::cout << "Data file: " << dataFile << std::endl;
        }
        std::cout << "--------------------------------" << std::endl;
    }
    
    void prepareTrainingData() {
        if (!useSyntheticData) {
            // Attempt to load data from file
            std::cout << "Attempting to load data from: " << dataFile << std::endl;
            if (std::filesystem::exists(dataFile)) {
                std::cout << "Loading data from " << dataFile << "..." << std::endl;
                if (dataLoader.loadCIFAR10Binary(dataFile)) {
                    inputDim = dataLoader.getImageSize();
                    std::cout << "Successfully loaded data with dimension: " << inputDim << "\n" << std::endl;
                    // Update latent dimension to match input dimension
                    latentDim = inputDim;
                    return;
                } else {
                    std::cout << "Failed to load data. Falling back to synthetic data." << std::endl;
                    useSyntheticData = true;
                }
            } else {
                std::cout << "Data file not found: " << dataFile << std::endl;
                std::cout << "Falling back to synthetic data." << std::endl;
                useSyntheticData = true;
            }
        }
        
        // Generate synthetic data if needed
        if (useSyntheticData) {
            int numTrainingSamples = 1000;
            std::cout << "Generating " << numTrainingSamples << " synthetic training samples..." << std::endl;
            trainingData = generateSyntheticData(numTrainingSamples, inputDim);
        }
    }
    
    void createOutputDirectory() {
        std::filesystem::create_directory("output");
        std::cout << "Created output directory for samples" << std::endl;
    }
    
    void trainModel() {
        std::cout << "Starting training..." << std::endl;
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            auto startTime = std::chrono::high_resolution_clock::now();
            double epochLoss = 0.0;
            int numBatches = 0;
            
            // Process mini-batches
            if (useSyntheticData) {
                // Process synthetic data in batches
                int numTrainingSamples = trainingData.rows();
                for (int i = 0; i < numTrainingSamples; i += batchSize) {
                    int currentBatchSize = std::min(batchSize, numTrainingSamples - i);
                    if (currentBatchSize < 2) continue; // Skip too small batches
                    
                    // Extract batch
                    Eigen::MatrixXd batch = trainingData.block(i, 0, currentBatchSize, inputDim);
                    
                    // Train on batch
                    double batchLoss = trainBatch(batch, epoch);
                    epochLoss += batchLoss;
                    numBatches++;
                }
            } else {
                // Process real data using dataLoader
                for (int i = 0; i < 10; i++) { // Multiple batches per epoch
                    Eigen::MatrixXd batch = dataLoader.getBatch(batchSize);
                    // Train on batch
                    double batchLoss = trainBatch(batch, epoch);
                    epochLoss += batchLoss;
                    // Track time
                    auto endTime = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = endTime - startTime;
                    // print progress
                    printProgress(epoch + 1, numEpochs,  numBatches > 0 ? epochLoss / numBatches : 0.0 , elapsed.count());
                    numBatches++;
                }
            }
            
            // Calculate average loss
            double avgLoss = numBatches > 0 ? epochLoss / numBatches : 0.0;
            
            // Track time
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = endTime - startTime;
            
            // Print progress
            printProgress(epoch + 1, numEpochs, avgLoss, elapsed.count());
            
            // Generate samples periodically
            if ((epoch + 1) % sampleFrequency == 0 || epoch == numEpochs - 1) {
                generateSamples(epoch + 1);
            }
        }
        
        std::cout << std::endl << "Training complete!" << std::endl;
    }
    
    double trainBatch(const Eigen::MatrixXd& batch, int epoch) {
        // Choose training method (alternating between different NoProp approaches)
        double batchLoss = 0.0;
        switch (epoch % 3) {
            case 0:
                // Direct Feedback Alignment
                batchLoss = model->trainStepNoProp(batch, learningRate);
                break;
            case 1:
                // Forward-Forward Algorithm
                batchLoss = model->trainStepForwardForward(batch, learningRate);
                break;
            case 2:
                // Weight Perturbation
                batchLoss = model->trainStepPerturbation(batch, learningRate);
                break;
        }
        
        return batchLoss;
    }
    
    void generateSamples(int epoch) {
        std::cout << std::endl << "Generating samples at epoch " << epoch << "..." << std::endl;
        
        // Generate samples
        Eigen::MatrixXd samples = model->sample(5);
        
        // Save samples
        std::string filename = "output/samples_epoch_" + std::to_string(epoch) + ".txt";
        saveSamples(samples, filename);
        
        // Print first sample
        std::cout << "Sample 1: " << std::endl;
        std::cout << samples.row(0) << std::endl;
    }
    
    void generateFinalSamples() {
        std::cout << "Generating final samples..." << std::endl;
        Eigen::MatrixXd finalSamples = model->sample(10);
        saveSamples(finalSamples, "output/final_samples.txt");
    }
    
    // Utility functions
    
    Eigen::MatrixXd generateSyntheticData(int numSamples, int dataDim) {
        Eigen::MatrixXd data(numSamples, dataDim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Generate simple patterns (e.g., circles, lines, etc.)
        for (int i = 0; i < numSamples; i++) {
            // Create a simple pattern (e.g., a gradient)
            for (int j = 0; j < dataDim; j++) {
                data(i, j) = 0.5 + 0.5 * std::sin(j * 3.14159 / dataDim + i * 0.1);
            }
            
            // Add some noise
            for (int j = 0; j < dataDim; j++) {
                data(i, j) += 0.1 * (dist(gen) - 0.5);
            }
        }
        
        return data;
    }
    
    void saveSamples(const Eigen::MatrixXd& samples, const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << samples;
            file.close();
            std::cout << "Saved samples to " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file for saving samples." << std::endl;
        }
    }
    
    void printProgress(int epoch, int totalEpochs, double loss, double elapsedTime) {
        int barWidth = 30;
        float progress = static_cast<float>(epoch) / totalEpochs;
        
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << int(progress * 100.0) << "% | Epoch: " << epoch << "/" << totalEpochs
                  << " | Loss: " << std::fixed << std::setprecision(6) << loss
                  << " | Time: " << std::fixed << std::setprecision(2) << elapsedTime << "s\r";
        std::cout.flush();
    }
};

#endif
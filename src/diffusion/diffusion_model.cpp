// diffusion_model.cpp
#include "diffusion_model.hpp"
#include <random>
#include <cmath>
#include <iostream>

DiffusionModel::DiffusionModel(int inputDim, int latentDim, int hiddenDim, int timesteps)
    : latentDim(latentDim), timesteps(timesteps), scheduler(NoiseScheduler(timesteps)) {
    
    // Create a simple UNet-like architecture
    auto relu = std::make_shared<ReLU>();
    auto sigmoid = std::make_shared<Sigmoid>();
    
    // Input layer includes time step embedding
    layers.push_back(std::make_shared<Layer>(inputDim + 1, hiddenDim, relu));
    
    // Hidden layers
    layers.push_back(std::make_shared<Layer>(hiddenDim, hiddenDim, relu));
    layers.push_back(std::make_shared<Layer>(hiddenDim, hiddenDim, relu));
    
    // Output layer
    layers.push_back(std::make_shared<Layer>(hiddenDim, inputDim, sigmoid));
}

Eigen::MatrixXd DiffusionModel::forwardDiffusion(const Eigen::MatrixXd& x0, int t) {
    return scheduler.addNoise(x0, t);
}

Eigen::MatrixXd DiffusionModel::predictNoise(const Eigen::MatrixXd& xt, int t) {
    // Create time embedding
    Eigen::MatrixXd timeEmbedding = Eigen::MatrixXd::Constant(xt.rows(), 1, t / static_cast<double>(timesteps));
    
    // Concatenate input with time embedding
    Eigen::MatrixXd input(xt.rows(), xt.cols() + 1);
    input << xt, timeEmbedding;
    
    // Forward pass through the network
    Eigen::MatrixXd output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    
    return output;
}

Eigen::MatrixXd DiffusionModel::sampleStep(const Eigen::MatrixXd& xt, int t) {
    if (t == 0) {
        return xt;
    }
    
    // Predict noise
    Eigen::MatrixXd predictedNoise = predictNoise(xt, t);
    
    // Get schedule parameters
    double alpha = scheduler.getAlpha(t);
    double beta = scheduler.getBeta(t);
    
    // Random noise for stochastic sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    Eigen::MatrixXd randomNoise = Eigen::MatrixXd::Zero(xt.rows(), xt.cols());
    for (int i = 0; i < randomNoise.rows(); i++) {
        for (int j = 0; j < randomNoise.cols(); j++) {
            randomNoise(i, j) = distribution(gen);
        }
    }
    
    // Perform denoising step
    double alphaRecip = 1.0 / std::sqrt(alpha);
    double sigmaT = std::sqrt(beta);
    
    return alphaRecip * (xt - (beta / std::sqrt(1 - scheduler.getAlphaCumprod(t))) * predictedNoise) + 
           sigmaT * randomNoise;
}

Eigen::MatrixXd DiffusionModel::sample(int batchSize) {
    // Start with random noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    Eigen::MatrixXd xt = Eigen::MatrixXd::Zero(batchSize, latentDim);
    for (int i = 0; i < xt.rows(); i++) {
        for (int j = 0; j < xt.cols(); j++) {
            xt(i, j) = distribution(gen);
        }
    }
    
    // Iteratively denoise
    for (int t = timesteps - 1; t >= 0; t--) {
        xt = sampleStep(xt, t);
    }
    
    return xt;
}

double DiffusionModel::trainStep(const Eigen::MatrixXd& batch, double learningRate) {
    double totalLoss = 0.0;
    
    // For each data point
    for (int i = 0; i < batch.rows(); i++) {
        // Extract single data point
        Eigen::MatrixXd x0 = batch.row(i);
        
        // Choose random timestep
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distribution(0, timesteps - 1);
        int t = distribution(gen);
        
        // Add noise to the data point
        Eigen::MatrixXd noise;
        Eigen::MatrixXd xt = scheduler.addNoise(x0, t, &noise);
        
        // Predict noise
        Eigen::MatrixXd predictedNoise = predictNoise(xt, t);
        
        // Calculate MSE loss
        Eigen::MatrixXd diff = predictedNoise - noise;
        double loss = diff.array().square().sum() / diff.size();
        totalLoss += loss;
        
        // Backpropagation (simplified)
        Eigen::MatrixXd gradient = 2.0 * (predictedNoise - noise) / diff.size();
        
        // Backward pass through the network
        for (int j = layers.size() - 1; j >= 0; j--) {
            gradient = layers[j]->backward(gradient, learningRate);
        }
    }
    
    return totalLoss / batch.rows();
}

void DiffusionModel::printNetworkStructure(const DiffusionModel& model) {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "         DIFFUSION MODEL STRUCTURE           " << std::endl;
    std::cout << "=============================================" << std::endl;
    
    std::cout << "Latent dimension: " << model.getLatentDim() << std::endl;
    std::cout << "Number of timesteps: " << model.getTimesteps() << std::endl;
    
    std::cout << "\nLAYER STRUCTURE:" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    
    // You'll need to add accessor methods to your DiffusionModel class
    // to expose the layers and their properties
    
    for (int i = 0; i < model.getNumLayers(); i++) {
        auto& layer = model.getLayer(i);
        const auto& weights = layer->getWeights();
        const auto& biases = layer->getBiases();
        
        std::cout << "Layer " << i+1 << ":" << std::endl;
        std::cout << "  Type: " << (i == 0 ? "Input" : 
                                   i == model.getNumLayers()-1 ? "Output" : "Hidden") << std::endl;
        std::cout << "  Weights shape: [" << weights.rows() << ", " << weights.cols() << "]" << std::endl;
        std::cout << "  Biases shape: [" << biases.rows() << "]" << std::endl;
        std::cout << "  Activation: " << 
            (i == model.getNumLayers()-1 ? "Sigmoid" : "ReLU") << std::endl;
        
        if (i < model.getNumLayers()-1) {
            std::cout << "  Output dimension: " << weights.rows() << std::endl;
        }
        
        std::cout << "---------------------------------------------" << std::endl;
    }
    
    std::cout << "\nNOISE SCHEDULER PARAMETERS:" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Beta start: 0.0001" << std::endl;
    std::cout << "Beta end: 0.02" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}
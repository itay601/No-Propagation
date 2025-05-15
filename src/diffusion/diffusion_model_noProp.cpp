#include "diffusion_model_noprop.hpp"
#include <random>
#include <cmath>
#include <iostream>

DiffusionModelNoProp::DiffusionModelNoProp(int inputDim, int latentDim, int hiddenDim, int timesteps,
                                     double perturbationStrength, int updateFrequency)
    : latentDim(latentDim), timesteps(timesteps), scheduler(NoiseScheduler(timesteps)),
      perturbationStrength(perturbationStrength), updateFrequency(updateFrequency), updateCounter(0) {
    
    // Create a simple UNet-like architecture with NoProp layers
    auto relu = std::make_shared<ReLU>();
    auto sigmoid = std::make_shared<Sigmoid>();
    
    // Input layer includes time step embedding
    layers.push_back(std::make_shared<LayerNoProp>(inputDim + 1, hiddenDim, relu, perturbationStrength));
    
    // Hidden layers
    layers.push_back(std::make_shared<LayerNoProp>(hiddenDim, hiddenDim, relu, perturbationStrength));
    layers.push_back(std::make_shared<LayerNoProp>(hiddenDim, hiddenDim, relu, perturbationStrength));
    
    // Output layer
    layers.push_back(std::make_shared<LayerNoProp>(hiddenDim, inputDim, sigmoid, perturbationStrength));
}

Eigen::MatrixXd DiffusionModelNoProp::forwardDiffusion(const Eigen::MatrixXd& x0, int t) {
    return scheduler.addNoise(x0, t);
}

Eigen::MatrixXd DiffusionModelNoProp::predictNoise(const Eigen::MatrixXd& xt, int t) {
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

Eigen::MatrixXd DiffusionModelNoProp::sampleStep(const Eigen::MatrixXd& xt, int t) {
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
    
    return alphaRecip * (xt - (beta / std::sqrt(1.0 - scheduler.getAlphaCumprod(t))) * predictedNoise) + 
           sigmaT * randomNoise;
}

Eigen::MatrixXd DiffusionModelNoProp::sample(int batchSize) {
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

double DiffusionModelNoProp::trainStepNoProp(const Eigen::MatrixXd& batch, double learningRate) {
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
        
        // NoProp update approach
        // Create time embedding for each layer
        Eigen::MatrixXd timeEmbedding = Eigen::MatrixXd::Constant(xt.rows(), 1, t / static_cast<double>(timesteps));
        Eigen::MatrixXd input(xt.rows(), xt.cols() + 1);
        input << xt, timeEmbedding;
        
        // Update each layer using NoProp (Direct Feedback Alignment)
        Eigen::MatrixXd output = input;
        Eigen::MatrixXd target = noise; // Target is the actual noise
        
        for (int j = 0; j < layers.size(); j++) {
            output = layers[j]->forward(output);
            // Last layer gets direct target, others get synthetic feedback
            if (j == layers.size() - 1) {
                layers[j]->updateNoProp(target, learningRate);
            } else {
                // Generate synthetic target for this layer
                std::mt19937 layerGen(rd());
                std::normal_distribution<double> noiseDist(0.0, 0.1);
                Eigen::MatrixXd layerTarget = Eigen::MatrixXd::Zero(output.rows(), output.cols());
                for (int r = 0; r < layerTarget.rows(); r++) {
                    for (int c = 0; c < layerTarget.cols(); c++) {
                        layerTarget(r, c) = noiseDist(layerGen);
                    }
                }
                layers[j]->updateNoProp(layerTarget, learningRate * 0.1); // Lower learning rate for hidden layers
            }
        }
    }
    
    // Occasionally apply weight perturbation for exploration
    updateCounter++;
    if (updateCounter % updateFrequency == 0) {
        for (auto& layer : layers) {
            layer->perturbWeights(perturbationStrength);
        }
    }
    
    return totalLoss / batch.rows();
}

double DiffusionModelNoProp::trainStepForwardForward(const Eigen::MatrixXd& batch, double learningRate) {
    double totalLoss = 0.0;
    
    // Need at least 2 samples for forward-forward
    if (batch.rows() < 2) {
        return 0.0;
    }
    
    for (int i = 0; i < batch.rows() - 1; i += 2) {
        // Use pairs of samples as positive/negative examples
        Eigen::MatrixXd positiveSample = batch.row(i);
        Eigen::MatrixXd negativeSample = batch.row(i+1);
        
        // Choose random timestep
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distribution(0, timesteps - 1);
        int t = distribution(gen);
        
        // Add noise to create positive and negative examples
        Eigen::MatrixXd noise;
        Eigen::MatrixXd xtPos = scheduler.addNoise(positiveSample, t, &noise);
        Eigen::MatrixXd xtNeg = scheduler.addNoise(negativeSample, t);
        
        // Add time embedding
        Eigen::MatrixXd timeEmbedding = Eigen::MatrixXd::Constant(1, 1, t / static_cast<double>(timesteps));
        
        Eigen::MatrixXd inputPos(1, xtPos.cols() + 1);
        inputPos << xtPos, timeEmbedding;
        
        Eigen::MatrixXd inputNeg(1, xtNeg.cols() + 1);
        inputNeg << xtNeg, timeEmbedding;
        
        // Train each layer with forward-forward
        for (auto& layer : layers) {
            double layerLoss = layer->forwardForward(inputPos, inputNeg);
            totalLoss += layerLoss;
        }
    }
    
    return totalLoss / (batch.rows() / 2);
}

double DiffusionModelNoProp::trainStepPerturbation(const Eigen::MatrixXd& batch, double learningRate) {
    // Weight perturbation approach based on evolutionary strategies
    double totalLoss = 0.0;
    double bestLoss = std::numeric_limits<double>::max();
    
    // Save original weights
    std::vector<Eigen::MatrixXd> originalWeights;
    std::vector<Eigen::VectorXd> originalBiases;
    
    for (auto& layer : layers) {
        originalWeights.push_back(layer->getWeights());
        originalBiases.push_back(layer->getBiases());
    }
    
    // Number of perturbations to try
    const int numPerturbations = 5;
    
    std::vector<std::vector<Eigen::MatrixXd>> perturbedWeights(numPerturbations);
    std::vector<std::vector<Eigen::VectorXd>> perturbedBiases(numPerturbations);
    std::vector<double> perturbationLosses(numPerturbations);
    
    // Generate perturbations and evaluate them
    for (int p = 0; p < numPerturbations; p++) {
        // Apply perturbation to all layers
        for (auto& layer : layers) {
            layer->perturbWeights(perturbationStrength);
        }
        
        // Save perturbed weights
        perturbedWeights[p].clear();
        perturbedBiases[p].clear();
        for (auto& layer : layers) {
            perturbedWeights[p].push_back(layer->getWeights());
            perturbedBiases[p].push_back(layer->getBiases());
        }
        
        // Evaluate loss for this perturbation
        double loss = 0.0;
        for (int i = 0; i < batch.rows(); i++) {
            Eigen::MatrixXd x0 = batch.row(i);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> distribution(0, timesteps - 1);
            int t = distribution(gen);
            
            Eigen::MatrixXd noise;
            Eigen::MatrixXd xt = scheduler.addNoise(x0, t, &noise);
            Eigen::MatrixXd predictedNoise = predictNoise(xt, t);
            
            Eigen::MatrixXd diff = predictedNoise - noise;
            loss += diff.array().square().sum() / diff.size();
        }
        loss /= batch.rows();
        
        perturbationLosses[p] = loss;
        
        // Restore original weights for next perturbation
        for (size_t j = 0; j < layers.size(); j++) {
            // Cast to LayerNoProp& and directly modify weights/biases
            // Note: This is a bit hacky but avoids adding setters to the class
            const_cast<Eigen::MatrixXd&>(layers[j]->getWeights()) = originalWeights[j];
            const_cast<Eigen::VectorXd&>(layers[j]->getBiases()) = originalBiases[j];
        }
    }
    
    // Find best perturbation
    int bestPerturbation = 0;
    for (int p = 0; p < numPerturbations; p++) {
        if (perturbationLosses[p] < bestLoss) {
            bestLoss = perturbationLosses[p];
            bestPerturbation = p;
        }
    }
    
    // Apply best perturbation
    if (bestLoss < std::numeric_limits<double>::max()) {
        for (size_t j = 0; j < layers.size(); j++) {
            const_cast<Eigen::MatrixXd&>(layers[j]->getWeights()) = perturbedWeights[bestPerturbation][j];
            const_cast<Eigen::VectorXd&>(layers[j]->getBiases()) = perturbedBiases[bestPerturbation][j];
        }
        totalLoss = bestLoss;
    } else {
        // If no improvement, keep original weights
        for (size_t j = 0; j < layers.size(); j++) {
            const_cast<Eigen::MatrixXd&>(layers[j]->getWeights()) = originalWeights[j];
            const_cast<Eigen::VectorXd&>(layers[j]->getBiases()) = originalBiases[j];
        }
    }
    
    return totalLoss;
}
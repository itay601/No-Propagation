#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "../neural_network/layer_noprop.hpp"
#include "noise.hpp"

class DiffusionModelNoProp {
private:
    // UNet-like architecture with NoProp layers
    std::vector<std::shared_ptr<LayerNoProp>> layers;
    NoiseScheduler scheduler;
    int latentDim;
    int timesteps;
    
    // NoProp specific parameters
    double perturbationStrength;
    int updateFrequency;
    int updateCounter;

public:
    DiffusionModelNoProp(int inputDim, int latentDim, int hiddenDim, int timesteps,
                     double perturbationStrength = 0.001, int updateFrequency = 5);
    
    // Forward diffusion process (adding noise)
    Eigen::MatrixXd forwardDiffusion(const Eigen::MatrixXd& x0, int t);
    
    // Predict noise to remove
    Eigen::MatrixXd predictNoise(const Eigen::MatrixXd& xt, int t);
    
    // Reverse diffusion process (removing noise)
    Eigen::MatrixXd sampleStep(const Eigen::MatrixXd& xt, int t);
    
    // Sample from the model
    Eigen::MatrixXd sample(int batchSize);
    
    // Train the model using NoProp methods
    double trainStepNoProp(const Eigen::MatrixXd& batch, double learningRate);
    
    // Alternative NoProp training methods
    double trainStepForwardForward(const Eigen::MatrixXd& batch, double learningRate);
    double trainStepPerturbation(const Eigen::MatrixXd& batch, double learningRate);
};
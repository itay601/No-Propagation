// diffusion_model.h
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "../neural_network/layer.hpp"
#include "noise.hpp"

class DiffusionModel {
private:
    // UNet-like architecture
    std::vector<std::shared_ptr<Layer>> layers;
    NoiseScheduler scheduler;
    int latentDim;
    int timesteps;

public:
    DiffusionModel(int inputDim, int latentDim, int hiddenDim, int timesteps);
    
    // Forward diffusion process (adding noise)
    Eigen::MatrixXd forwardDiffusion(const Eigen::MatrixXd& x0, int t);
    
    // Predict noise to remove
    Eigen::MatrixXd predictNoise(const Eigen::MatrixXd& xt, int t);
    
    // Reverse diffusion process (removing noise)
    Eigen::MatrixXd sampleStep(const Eigen::MatrixXd& xt, int t);
    
    // Sample from the model
    Eigen::MatrixXd sample(int batchSize);
    
    // Train the model on a batch
    double trainStep(const Eigen::MatrixXd& batch, double learningRate);

    // Print structure of NN 
    void printNetworkStructure(const DiffusionModel& model);
    
    int getLatentDim() const { return latentDim; }
    int getTimesteps() const { return timesteps; }
    int getNumLayers() const { return layers.size(); }
    const std::shared_ptr<Layer>& getLayer(int idx) const { return layers[idx]; }
};


#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include "activation.hpp"

class Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::shared_ptr<Activation> activation;
    Eigen::MatrixXd lastInput;
    Eigen::MatrixXd lastOutput;
    Eigen::MatrixXd lastActivation;

public:
    Layer(int inputSize, int outputSize, std::shared_ptr<Activation> activation);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& outputGradient, double learningRate);
    
    // Getters
    const Eigen::MatrixXd& getWeights() const { return weights; }
    const Eigen::VectorXd& getBiases() const { return biases; }
};

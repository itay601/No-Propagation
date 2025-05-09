// layer.cpp
#include "layer.hpp"
#include <random>

Layer::Layer(int inputSize, int outputSize, std::shared_ptr<Activation> activation)
    : activation(activation) {
    
    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / (inputSize + outputSize)));
    
    weights = Eigen::MatrixXd::Zero(outputSize, inputSize);
    biases = Eigen::VectorXd::Zero(outputSize);
    
    // Fill weights with random values
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weights(i, j) = distribution(gen);
        }
    }
}

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd& input) {
    lastInput = input;
    lastOutput = (weights * input.transpose()).colwise() + biases;
    lastActivation = activation->forward(lastOutput);
    return lastActivation.transpose();
}

Eigen::MatrixXd Layer::backward(const Eigen::MatrixXd& outputGradient, double learningRate) {
    // Calculate gradient through activation function
    Eigen::MatrixXd activationGradient = activation->backward(lastOutput, outputGradient.transpose());
    
    // Calculate gradients for weights and biases
    Eigen::MatrixXd weightsGradient = activationGradient * lastInput;
    Eigen::VectorXd biasesGradient = activationGradient.rowwise().sum();
    
    // Calculate gradient to pass to previous layer
    Eigen::MatrixXd inputGradient = weights.transpose() * activationGradient;
    
    // Update weights and biases
    weights -= learningRate * weightsGradient;
    biases -= learningRate * biasesGradient;
    
    return inputGradient.transpose();
}
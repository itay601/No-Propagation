#include "layer_noprop.hpp"
#include <random>
#include <iostream>
#include <cmath>

LayerNoProp::LayerNoProp(int inputSize, int outputSize, std::shared_ptr<Activation> activation,
                     double perturbationSize)
    : activation(activation), perturbationSize(perturbationSize), rng(std::random_device{}()) {
    
    // Initialize weights with Xavier/Glorot initialization
    std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / (inputSize + outputSize)));
    
    weights = Eigen::MatrixXd::Zero(outputSize, inputSize);
    biases = Eigen::VectorXd::Zero(outputSize);
    
    // Fill weights with random values
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weights(i, j) = distribution(rng);
        }
    }
}

Eigen::MatrixXd LayerNoProp::forward(const Eigen::MatrixXd& input) {
    lastInput = input;
    lastOutput = (weights * input.transpose()).colwise() + biases;
    lastActivation = activation->forward(lastOutput);
    return lastActivation.transpose();
}

void LayerNoProp::updateNoProp(const Eigen::MatrixXd& targetOutput, double learningRate) {
    // NoProp approach 1: Direct Feedback Alignment
    // Instead of backpropagating errors, generate direct feedback for each layer
    
    // Calculate error at the output
    Eigen::MatrixXd outputError = targetOutput.transpose() - lastActivation;
    
    // Generate random fixed feedback weights (in practice these could be stored)
    std::normal_distribution<double> dist(0.0, 0.1);
    Eigen::MatrixXd feedbackWeights = Eigen::MatrixXd::Zero(weights.rows(), weights.rows());
    for (int i = 0; i < feedbackWeights.rows(); i++) {
        for (int j = 0; j < feedbackWeights.cols(); j++) {
            feedbackWeights(i, j) = dist(rng);
        }
    }
    
    // Calculate update signals using direct feedback
    Eigen::MatrixXd updateSignal = feedbackWeights * outputError;
    
    // Update weights and biases directly based on layer activation functions
    Eigen::MatrixXd weightUpdate = updateSignal * lastInput;
    Eigen::VectorXd biasUpdate = updateSignal.rowwise().sum();
    
    weights += learningRate * weightUpdate;
    biases += learningRate * biasUpdate;
}

void LayerNoProp::perturbWeights(double magnitude) {
    // NoProp approach 2: Weight Perturbation
    // Randomly perturb weights to explore parameter space
    std::normal_distribution<double> dist(0.0, magnitude);
    
    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            weights(i, j) += dist(rng);
        }
    }
    
    for (int i = 0; i < biases.size(); i++) {
        biases(i) += dist(rng);
    }
}

double LayerNoProp::forwardForward(const Eigen::MatrixXd& positiveSample, 
                               const Eigen::MatrixXd& negativeSample) {
    // NoProp approach 3: Forward-Forward Algorithm
    // Use the difference between positive and negative samples to update weights
    
    // Process positive sample
    Eigen::MatrixXd posOutput = (weights * positiveSample.transpose()).colwise() + biases;
    Eigen::MatrixXd posActivation = activation->forward(posOutput);
    
    // Process negative sample
    Eigen::MatrixXd negOutput = (weights * negativeSample.transpose()).colwise() + biases;
    Eigen::MatrixXd negActivation = activation->forward(negOutput);
    
    // Calculate goodness of fit (squared sum of activations)
    double posGoodness = posActivation.array().square().sum();
    double negGoodness = negActivation.array().square().sum();
    
    // Calculate update based on difference
    double learningRate = 0.01;
    Eigen::MatrixXd weightUpdate = learningRate * (
        posActivation * positiveSample - negActivation * negativeSample
    );
    
    weights += weightUpdate;
    
    // Return loss value (negative samples should have lower activation)
    return negGoodness - posGoodness;
}
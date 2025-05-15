#pragma once
#include <Eigen/Dense>
#include <memory>
#include <random>
#include <string>
#include "activation.hpp"

class LayerNoProp {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::shared_ptr<Activation> activation;
    Eigen::MatrixXd lastInput;
    Eigen::MatrixXd lastOutput;
    Eigen::MatrixXd lastActivation;
    
    // NoProp specific members
    double perturbationSize;
    std::mt19937 rng;
    
public:
    LayerNoProp(int inputSize, int outputSize, std::shared_ptr<Activation> activation, 
               double perturbationSize = 0.01);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    
    // NoProp update method (replacement for backward)
    void updateNoProp(const Eigen::MatrixXd& targetOutput, double learningRate);
    
    // Direct weight perturbation method
    void perturbWeights(double magnitude);
    
    // Forward-Forward algorithm method (alternative NoProp approach)
    double forwardForward(const Eigen::MatrixXd& positiveSample, const Eigen::MatrixXd& negativeSample);
    
    // Getters
    const Eigen::MatrixXd& getWeights() const { return weights; }
    const Eigen::VectorXd& getBiases() const { return biases; }
};
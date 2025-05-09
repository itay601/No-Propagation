// activation.cpp
#include "activation.hpp"
#include <cmath>

// ReLU Implementation
Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& input) {
    return input.cwiseMax(0.0);
}

Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) {
    Eigen::MatrixXd result = input;
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            result(i, j) = input(i, j) > 0 ? 1.0 : 0.0;
        }
    }
    return result.cwiseProduct(gradient);
}

// Sigmoid Implementation
Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd result = input;
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            result(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return result;
}

Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) {
    Eigen::MatrixXd sigmoid = forward(input);
    return gradient.cwiseProduct(sigmoid.cwiseProduct(
        Eigen::MatrixXd::Ones(sigmoid.rows(), sigmoid.cols()) - sigmoid));
}

// Tanh Implementation
Eigen::MatrixXd Tanh::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd result = input;
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            result(i, j) = std::tanh(input(i, j));
        }
    }
    return result;
}

Eigen::MatrixXd Tanh::backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) {
    Eigen::MatrixXd tanhx = forward(input);
    return gradient.cwiseProduct(
        Eigen::MatrixXd::Ones(tanhx.rows(), tanhx.cols()) - tanhx.cwiseProduct(tanhx));
}
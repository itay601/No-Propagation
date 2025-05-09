// noise_scheduler.cpp
#include "noise.hpp"
#include <random>
#include <cmath>

NoiseScheduler::NoiseScheduler(int timesteps, double betaStart, double betaEnd)
    : timesteps(timesteps) {
    
    // Linear beta schedule
    betas.resize(timesteps);
    alphas.resize(timesteps);
    alphasCumprod.resize(timesteps);
    
    for (int i = 0; i < timesteps; i++) {
        betas[i] = betaStart + (betaEnd - betaStart) * i / (timesteps - 1);
        alphas[i] = 1.0 - betas[i];
    }
    
    // Calculate cumulative product of alphas
    alphasCumprod[0] = alphas[0];
    for (int i = 1; i < timesteps; i++) {
        alphasCumprod[i] = alphasCumprod[i-1] * alphas[i];
    }
}

double NoiseScheduler::getBeta(int t) const {
    return betas[t];
}

double NoiseScheduler::getAlpha(int t) const {
    return alphas[t];
}

double NoiseScheduler::getAlphaCumprod(int t) const {
    return alphasCumprod[t];
}

Eigen::MatrixXd NoiseScheduler::addNoise(const Eigen::MatrixXd& x0, int t, Eigen::MatrixXd* noise) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    Eigen::MatrixXd epsilon = Eigen::MatrixXd::Zero(x0.rows(), x0.cols());
    
    // Generate random noise
    for (int i = 0; i < epsilon.rows(); i++) {
        for (int j = 0; j < epsilon.cols(); j++) {
            epsilon(i, j) = distribution(gen);
        }
    }
    
    // Store noise if pointer is provided
    if (noise != nullptr) {
        *noise = epsilon;
    }
    
    // Add noise according to the schedule
    double alphaCumprod = getAlphaCumprod(t);
    return std::sqrt(alphaCumprod) * x0 + std::sqrt(1.0 - alphaCumprod) * epsilon;
}

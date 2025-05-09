// noise_scheduler.h
#pragma once
#include <Eigen/Dense>
#include <vector>

class NoiseScheduler {
private:
    std::vector<double> betas;
    std::vector<double> alphas;
    std::vector<double> alphasCumprod;
    int timesteps;

public:
    NoiseScheduler(int timesteps, double betaStart = 0.0001, double betaEnd = 0.02);
    
    // Get noise schedule parameters
    double getBeta(int t) const;
    double getAlpha(int t) const;
    double getAlphaCumprod(int t) const;
    
    // Add noise to the input according to the schedule
    Eigen::MatrixXd addNoise(const Eigen::MatrixXd& x0, int t, Eigen::MatrixXd* noise = nullptr);
};
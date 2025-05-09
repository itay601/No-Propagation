// activation.h
#pragma once
#include <Eigen/Dense>
#include <string>

class Activation {
public:
    virtual ~Activation() = default;
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) = 0;
};

class ReLU : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) override;
};

class Sigmoid : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) override;
};

class Tanh : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) override;
};
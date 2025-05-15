#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <string>

// Generates a simple synthetic dataset (for testing or fallback)
Eigen::MatrixXd generateSyntheticData(int numSamples, int dim);

// Creates a directory if it doesn't exist
void createDirectory(const std::string& path);

#endif // UTILS_HPP
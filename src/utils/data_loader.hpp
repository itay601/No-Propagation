#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class ImageDataLoader {
private:
    int numSamples;
    int channels;
    int height;
    int width;
    std::vector<Eigen::MatrixXd> images;  // Each image flattened to a single row

public:
    ImageDataLoader() : numSamples(0), channels(0), height(0), width(0) {}
    
    bool loadCIFAR10Binary(const std::string& filename);
    
    // Get a batch of images
    Eigen::MatrixXd getBatch(int batchSize) const;
    
    // Convert a flattened image back to RGB format for visualization
    Eigen::MatrixXd reshapeToImage(const Eigen::MatrixXd& flattenedImage) const;
    
    // Save an image to a PPM file (simple image format)
    bool saveImageAsPPM(const Eigen::MatrixXd& flattenedImage, const std::string& filename) const;
    bool saveTensorAsJSON(const Eigen::MatrixXd& flattenedImage, const std::string& filename) const; 
    // Getters
    int getNumSamples() const { return numSamples; }
    int getChannels() const { return channels; }
    int getHeight() const { return height; }
    int getWidth() const { return width; }
    int getImageSize() const { return channels * height * width; }
};
#include "data_loader.hpp"

   // check if the binary dataset is in format .
   bool ImageDataLoader::loadCIFAR10Binary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        // --- read header (numSamples, channels, height, width) ---
        int32_t header[4] = {0,0,0,0};
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        if (file.gcount() != sizeof(header)) {
            std::cerr << "Error: only read " << file.gcount()
                    << " bytes for header, expected " << sizeof(header) << std::endl;
            return false;
        }
        numSamples = header[0];
        channels   = header[1];
        height     = header[2];
        width      = header[3];

        const int pixelsPerImage = channels * height * width;
        const int64_t totalBytes = int64_t(numSamples) * pixelsPerImage;  // one byte per pixel

        // --- sanity-check file size for uint8 data + 16-byte header ---
        file.seekg(0, std::ios::end);
        const auto fileSize = file.tellg();
        const auto expected = int64_t(sizeof(header)) + totalBytes * sizeof(uint8_t);
        if (fileSize != expected) {
            std::cerr << "Error: file size mismatch. On disk = " << fileSize
                    << " bytes, but header implies " << expected << " bytes." << std::endl;
            return false;
        }

        // --- back to first image byte ---
        file.seekg(sizeof(header), std::ios::beg);

        images.clear();
        images.reserve(numSamples);

        // --- read & convert one image at a time ---
        std::vector<uint8_t> byteBuf(pixelsPerImage);
        for (int i = 0; i < numSamples; ++i) {
            file.read(reinterpret_cast<char*>(byteBuf.data()), pixelsPerImage);
            if (!file) {
                std::cerr << "Error: failed reading image " << i
                        << " (read " << file.gcount() << " bytes)" << std::endl;
                return false;
            }

            // convert to float row‐vector
            Eigen::MatrixXd mat(1, pixelsPerImage);
            for (int j = 0; j < pixelsPerImage; ++j) {
                mat(0, j) = static_cast<float>(byteBuf[j]);
            }
            images.push_back(std::move(mat));
        }

        std::cout << "Loaded " << numSamples << " images, each "
                << channels << "x" << height << "x" << width
                << " (" << pixelsPerImage << " pixels)" << std::endl;
        return true;
    }
    
    // Get a batch of images
    Eigen::MatrixXd ImageDataLoader::getBatch(int batchSize) const {
        if (numSamples == 0 || batchSize > numSamples) {
            std::cerr << "Error: Invalid batch size or no data loaded" << std::endl;
            return Eigen::MatrixXd::Zero(1, 1);
        }
        
        // Random selection
        std::vector<int> indices(batchSize);
        for (int i = 0; i < batchSize; i++) {
            indices[i] = rand() % numSamples;
        }
        
        int pixelsPerImage = channels * height * width;
        Eigen::MatrixXd batch(batchSize, pixelsPerImage);
        
        for (int i = 0; i < batchSize; i++) {
            batch.row(i) = images[indices[i]];
        }
        
        return batch;
    }
    
    // Convert a flattened image back to RGB format for visualization
    Eigen::MatrixXd ImageDataLoader::reshapeToImage(const Eigen::MatrixXd& flattenedImage) const {
        if (flattenedImage.cols() != channels * height * width) {
            std::cerr << "Error: Invalid dimensions for reshaping" << std::endl;
            return Eigen::MatrixXd::Zero(1, 1);
        }
        
        // Reshape to [height, width, channels]
        Eigen::MatrixXd reshapedImage(height, width * channels);
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int flatIndex = c * height * width + h * width + w;
                    reshapedImage(h, w * channels + c) = flattenedImage(0, flatIndex);
                }
            }
        }
        
        return reshapedImage;
    }
    
    // Save an image to a PPM file (simple image format)
    bool ImageDataLoader::saveImageAsPPM(const Eigen::MatrixXd& flattenedImage, const std::string& filename) const {
        if (channels != 3) {
            std::cerr << "Error: PPM format requires 3 channels" << std::endl;
            return false;
        }
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        
        // Write PPM header
        file << "P6\n" << width << " " << height << "\n255\n";
        
        // Convert image values from [-1, 1] to [0, 255]
        std::vector<unsigned char> pixels(channels * height * width);
        for (int i = 0; i < channels * height * width; i++) {
            // Denormalize from [-1, 1] to [0, 255]
            float value = flattenedImage(0, i) * 0.5 + 0.5;
            pixels[i] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, value * 255.0f)));
        }
        
        // Reorder from CHW to HWC format for PPM
        std::vector<unsigned char> reordered(channels * height * width);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int srcIdx = c * height * width + h * width + w;
                    int dstIdx = (h * width + w) * channels + c;
                    reordered[dstIdx] = pixels[srcIdx];
                }
            }
        }
        
        // Write pixel data
        file.write(reinterpret_cast<char*>(reordered.data()), reordered.size());
        
        return !file.fail();
    }

    bool ImageDataLoader::saveTensorAsJSON(const Eigen::MatrixXd& flattenedImage, const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        
        file << "{" << std::endl;
        file << "  \"channels\": " << channels << "," << std::endl;
        file << "  \"height\": " << height << "," << std::endl;
        file << "  \"width\": " << width << "," << std::endl;
        file << "  \"data\": [";
        
        const int total = flattenedImage.cols();
        for (int i = 0; i < total; i++) {
            file << flattenedImage(0, i);
            if (i != total - 1) {
                file << ", ";
            }
        }
        file << "]" << std::endl;
        file << "}" << std::endl;
        
        return true;
    }
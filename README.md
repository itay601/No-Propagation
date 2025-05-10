# No-Prop Diffusion Function

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-username/nn_diffusion_project/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project implements a diffusion function combined with a neural network. It uses the Eigen library for linear algebra computations and CMake for managing the build process.

## Prerequisites

Before building the project, ensure you have the following installed:

- **Python 3**  
- **Virtualenv** (or `venv` module which is included in Python 3)
- **Eigen** (C++ linear algebra library)  
- **CMake**
- **A C++ compiler** (e.g., `g++`)

### Installing Prerequisites

**1. Python & Virtual Environment:**

Create a virtual Python environment to manage any Python dependencies (if needed):

```bash
python -m venv env
source env/bin/activate

sudo apt-get update
sudo apt-get install libeigen3-dev


sudo apt-get install cmake

# Create a build directory and navigate to it
mkdir build
cd build

# Configure the project
cmake ..

# Build the project
make

# Run the executable
./nn_diffusion

----------------------------------------------------------
## Project Structure

nn_diffusion_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── neural_network/
│   │   ├── layer.h
│   │   ├── layer.cpp
│   │   ├── activation.h
│   │   ├── activation.cpp
│   │   ├── loss.h
│   │   ├── loss.cpp
│   │   ├── optimizer.h
│   │   └── optimizer.cpp
│   ├── diffusion/
│   │   ├── diffusion_model.h
│   │   ├── diffusion_model.cpp
│   │   ├── noise_scheduler.h
│   │   └── noise_scheduler.cpp
│   └── utils/
│       ├── matrix.h
│       ├── matrix.cpp
│       ├── data_loader.h
│       └── data_loader.cpp
├── include/
│   └── external_libs/
├── tests/
│   ├── test_nn.cpp
│   └── test_diffusion.cpp
└── examples/
    ├── simple_nn.cpp
    └── simple_diffusion.cpp

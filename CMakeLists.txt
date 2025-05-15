cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkDiffusion)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen library
find_package(Eigen3 REQUIRED)
if(TARGET Eigen3::Eigen)
    # Use the imported target if available
    message(STATUS "Using Eigen3 imported target")
else()
    # Fallback to manual include paths
    message(STATUS "Using manual Eigen3 include paths")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Additional include directory for Eigen (based on your system path)
include_directories("/usr/local/include/eigen3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCES 
    src/main.cpp 
    src/neural_network/layer.cpp 
    src/neural_network/activation.cpp 
    src/neural_network/noPropLayer.cpp
    src/diffusion/diffusion_model.cpp 
    src/diffusion/noise.cpp
    src/diffusion/diffusion_model_noProp.cpp
    src/utils/data_loader.cpp
    src/utils/utils.cpp
    
)

# Create executable
add_executable(nn_diffusion ${SOURCES})

# Link against Eigen if using the imported target
if(TARGET Eigen3::Eigen)
    target_link_libraries(nn_diffusion Eigen3::Eigen)
endif()

target_link_libraries(nn_diffusion stdc++fs)

# If you want to install the executable
install(TARGETS nn_diffusion DESTINATION bin)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# Create data directory 
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)
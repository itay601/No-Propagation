# No-Prop (Difution Function)

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
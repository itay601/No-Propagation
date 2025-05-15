#include "utils/SimpleModelTrainer.hpp"

int main(int argc, char** argv) {
    SimpleModelTrainer trainer(argc, argv);
    trainer.run();
    return 0;
}
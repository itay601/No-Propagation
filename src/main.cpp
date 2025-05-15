#include "utils/SimpleModelTrainer.hpp"
#include "utils/NoPropModelTrainer.hpp"


int main(int argc, char** argv) {
    //SimpleModelTrainer trainer(argc, argv);
    //trainer.run();
    //return 0;
    NoPropModelTrainer trainer(argc, argv);
    trainer.run();
    return 0;
}

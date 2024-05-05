#include <string.h>
#include "surakarta_alphazero.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Usage:  %s model_path [args...]\n", argv[0]);
        printf("Notice: If model does not exist, a new model will be created and saved.\n");
        printf("        Trained model wil coverwrite the existing model, so backup if needed.\n");
        printf("Args:   -i|--iterations <int>    Number of iterations to train the model, default = 100\n");
        printf("        -s|--simulation <int>    Number of simulations per move, default = 50\n");
        printf("        -c|--cpuct <float>       CPUCT value, default = 1.0\n");
        printf("        -t|--temperature <float> Temperature value, default = 1.0\n");
        return 1;
    }
    int iterations = 100;
    int simulation_per_move = 50;
    float cpuct = 1.0f;
    float temperature = 1.0f;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            iterations = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--simulation") == 0) {
            simulation_per_move = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cpuct") == 0) {
            cpuct = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temperature") == 0) {
            temperature = std::stof(argv[++i]);
        }
    }

    auto train_util = SurakartaAlphazeroLoadTrainSaveUtil(
        std::make_shared<SurakartaAlphazeroNeuralNetworkFactory>(1, 30));
    train_util.Train(argv[1], iterations, simulation_per_move, cpuct, temperature, std::make_shared<SurakartaLoggerStdout>());

    return 0;
}

// int main(int argc, char** argv) {
//     int iterations = 100;
//     int simulation_per_move = 50;
//     float cpuct = 1.0f;
//     float temperature = 1.0f;

//     auto train_util = SurakartaAlphazeroLoadTrainSaveUtil(
//         std::make_shared<SurakartaAlphazeroNeuralNetworkFactory>(1, 30));
//     train_util.Train("tmp_model", iterations, simulation_per_move, cpuct, temperature, std::make_shared<SurakartaLoggerStdout>());

//     return 0;
// }

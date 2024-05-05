#pragma once
#include "surakarta.h"

class SurakartaAlphazeroNeuralNetworkBase {
   public:
    virtual ~SurakartaAlphazeroNeuralNetworkBase() = default;

    typedef struct {
        SurakartaMove move;
        float probability;
    } MoveWithProbability;

    typedef struct {
        std::unique_ptr<std::vector<MoveWithProbability>> move_probabilities;
        float current_status_value;
    } NeuralNetworkOutput;

    typedef struct {
        std::unique_ptr<SurakartaBoard> board;
        SurakartaGameInfo game_info;
        PieceColor my_color;
    } NeuralNetworkInput;

    virtual NeuralNetworkOutput Predict(NeuralNetworkInput input) = 0;

    typedef struct {
        NeuralNetworkInput input;
        NeuralNetworkOutput output;
    } TrainEntry;

    virtual void Train(std::unique_ptr<std::vector<TrainEntry>> train_data) = 0;

    virtual void SaveModel(const std::string& model_path) = 0;

    class ModelFactory {
       public:
        virtual std::unique_ptr<SurakartaAlphazeroNeuralNetworkBase> CreateModel(const std::string& model_path) = 0;
        virtual std::unique_ptr<SurakartaAlphazeroNeuralNetworkBase> LoadModel(const std::string& model_path) = 0;
    };
};

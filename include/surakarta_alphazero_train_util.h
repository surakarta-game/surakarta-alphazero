#pragma once
#include "surakarta_alphazero_neural_network_base.h"

class SurakartaAlphazeroTrainUtil {
   public:
    SurakartaAlphazeroTrainUtil(
        std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model,
        int simulation_per_move,
        float cpuct,
        float temperature)
        : model_(model),
          simulation_per_move_(simulation_per_move),
          cpuct_(cpuct),
          temperature_(temperature){};

    /// @brief Do a single self-play game and train the model
    void TrainSingleIteration(std::shared_ptr<SurakartaLogger> logger);

   private:
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model_;
    int simulation_per_move_;
    float cpuct_;
    float temperature_;
};

class SurakartaAlphazeroLoadTrainSaveUtil {
   public:
    SurakartaAlphazeroLoadTrainSaveUtil(
        std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase::ModelFactory> model_factory)
        : model_factory_(model_factory){};

    void Train(
        const std::string& model_path,
        int iterations,
        int simulation_per_move,
        float cpuct,
        float temperature,
        std::shared_ptr<SurakartaLogger> logger = std::make_shared<SurakartaLoggerNull>());

   private:
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase::ModelFactory> model_factory_;
};

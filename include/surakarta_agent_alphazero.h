#pragma once
#include "surakarta_agent_base.h"
#include "surakarta_alphazero_mcts.h"
#include "surakarta_alphazero_neural_network_base.h"
#include "surakarta_daemon.h"

class SurakartaAgentAlphazero : public SurakartaAgentBase {
   public:
    SurakartaAgentAlphazero(std::shared_ptr<SurakartaBoard> board,
                            std::shared_ptr<SurakartaGameInfo> game_info,
                            std::shared_ptr<SurakartaRuleManager> rule_manager,
                            PieceColor my_color,
                            std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model,
                            int simulation_per_move,
                            float cpuct,
                            float temperature)
        : SurakartaAgentBase(board, game_info, rule_manager),
          model_(model),
          my_color_(my_color),
          simulation_per_move_(simulation_per_move),
          cpuct_(cpuct),
          temperature_(temperature){};

    virtual SurakartaMove CalculateMove() override;

    /// @brief Event that is triggered when all the simulations are finished.
    /// This is used to train the neural network.
    SurakartaEvent<SurakartaAlphazeroMCTS&> OnSimulationsFinished;

   private:
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model_;
    PieceColor my_color_;
    int simulation_per_move_;
    float cpuct_;
    float temperature_;
};

class SurakartaAgentAlphazeroFactory : public SurakartaDaemon::AgentFactory {
   public:
    SurakartaAgentAlphazeroFactory(
        std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model,
        int simulation_per_move,
        float cpuct,
        float temperature)
        : model_(model),
          simulation_per_move_(simulation_per_move),
          cpuct_(cpuct),
          temperature_(temperature){};

    virtual std::unique_ptr<SurakartaAgentBase> CreateAgent(
        std::shared_ptr<SurakartaGameInfo> game_info,
        std::shared_ptr<SurakartaBoard> board,
        std::shared_ptr<SurakartaRuleManager> rule_manager,
        SurakartaDaemon& daemon,
        PieceColor my_color) override;

    void AddOnSimulationsFinishedHandler(std::function<void(SurakartaAlphazeroMCTS&)> handler) {
        on_simulations_finished_list_.push_back(handler);
    }

   private:
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model_;
    int simulation_per_move_;
    float cpuct_;
    float temperature_;
    std::vector<std::function<void(SurakartaAlphazeroMCTS&)>> on_simulations_finished_list_;
};

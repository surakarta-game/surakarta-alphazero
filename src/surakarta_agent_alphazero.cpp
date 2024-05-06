#include "surakarta_agent_alphazero.h"
#include "global_random_generator.h"
#include "surakarta_alphazero_mcts.h"

// returns a random float between 0 and 1
static float random_float() {
    auto& random_engine = GlobalRandomGenerator::getInstance();
    const auto random_int = random_engine();
    return static_cast<float>(random_int - random_engine.min()) /
           static_cast<float>(random_engine.max() - random_engine.min());
}

SurakartaMove SurakartaAgentAlphazero::CalculateMove() {
    auto mcts = SurakartaAlphazeroMCTS(
        board_,
        game_info_,
        my_color_,
        model_,
        cpuct_);
    for (int i = 0; i < simulation_per_move_; i++) {
        mcts.Simulate();
    }
    OnSimulationsFinished.Invoke(mcts);
    const auto possibilities = mcts.CalculateMoveProbabilities(temperature_);
    float cursor = 0;
    const auto random_value = random_float();
    for (const auto possibility : *possibilities) {
        cursor += possibility.probability;
        if (cursor >= random_value) {
            return possibility.move;
        }
    }
    // no move found, return a invalid move
    return SurakartaMove(SurakartaPosition(0, 0), SurakartaPosition(0, 0), my_color_);
}

std::unique_ptr<SurakartaAgentBase> SurakartaAgentAlphazeroFactory::CreateAgent(
    std::shared_ptr<SurakartaGameInfo> game_info,
    std::shared_ptr<SurakartaBoard> board,
    std::shared_ptr<SurakartaRuleManager> rule_manager,
    SurakartaDaemon& daemon,
    PieceColor my_color) {
    auto agent = std::make_unique<SurakartaAgentAlphazero>(
        board, game_info, rule_manager, my_color, model_, simulation_per_move_, cpuct_, temperature_);
    for (int i = 0; i < on_simulations_finished_list_.size(); i++) {
        agent->OnSimulationsFinished.AddListener(on_simulations_finished_list_[i]);
    }
    return agent;
}

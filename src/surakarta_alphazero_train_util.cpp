#include <filesystem>
#include <mutex>
#include <thread>
#include "surakarta.h"
#include "surakarta_alphazero.h"

void SurakartaAlphazeroTrainUtil::TrainSingleIteration(std::shared_ptr<SurakartaLogger> logger) {
    const int concurrency = std::thread::hardware_concurrency();
    auto threads = std::make_unique<std::thread[]>(concurrency);
    auto train_entries = std::make_unique<std::vector<SurakartaAlphazeroNeuralNetworkBase::TrainEntry>>(concurrency);
    std::mutex mutex;
    for (int i = 0; i < concurrency; i++) {
        threads[i] = std::thread([this, &train_entries, logger, &mutex]() {
            auto factory = std::make_shared<SurakartaAgentAlphazeroFactory>(model_, simulation_per_move_, cpuct_, temperature_);
            factory->AddOnSimulationsFinishedHandler([&train_entries, &mutex](SurakartaAlphazeroMCTS& mcts) {
                std::lock_guard<std::mutex> lock(mutex);
                train_entries->push_back(mcts.GetTrainEntriesWithoutValue());
            });
            auto daemon = SurakartaDaemon(BOARD_SIZE, MAX_NO_CAPTURE_ROUND, factory, factory);
            daemon.Execute();
            const auto game_info = daemon.CopyGameInfo();
            {
                std::lock_guard<std::mutex> lock(mutex);
                logger->Log("Game finished. total %d moves, stalemate: %s", game_info.num_round_,
                            game_info.Winner() == PieceColor::NONE ? "true" : "false");
            }
            const auto winner = daemon.CopyGameInfo().Winner();
            if (winner == PieceColor::NONE) {
                for (auto& entry : *train_entries) {
                    entry.output.current_status_value = 0.0f;
                }
            } else {
                for (auto& entry : *train_entries) {
                    entry.output.current_status_value = winner == entry.input.game_info.current_player_ ? 1.0f : -1.0f;
                }
            }
        });
    }
    for (int i = 0; i < concurrency; i++) {
        threads[i].join();
    }
    model_->Train(std::move(train_entries));
}

void SurakartaAlphazeroLoadTrainSaveUtil::Train(const std::string& model_path,
                                                int iterations,
                                                int simulation_per_move,
                                                float cpuct,
                                                float temperature,
                                                std::shared_ptr<SurakartaLogger> logger) {
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> model;
    if (std::filesystem::exists(model_path)) {
        logger->Log("Loading model from %s", model_path.c_str());
        model = model_factory_->LoadModel(model_path);
    } else {
        logger->Log("Model %s does not exist, creating a new model", model_path.c_str());
        model = model_factory_->CreateModel(model_path);
    }
    auto train_util = SurakartaAlphazeroTrainUtil(model, simulation_per_move, cpuct, temperature);
    logger->Log("Start training. Total: %d iterations", iterations);
    for (int i = 0; i < iterations; i++) {
        train_util.TrainSingleIteration(logger);
        model->SaveModel(model_path);
        logger->Log("Iteration %d/%d completed and new model saved", i + 1, iterations);
    }
}

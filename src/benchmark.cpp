// This file is modified from third-party/surakarta-core/src/main.cpp

#include <climits>
#include <cstring>
#include <thread>

#include "surakarta.h"
#include "surakarta_alphazero.h"

#define ANSI_CLEAR_SCREEN "\033[2J"
#define ANSI_MOVE_TO_START "\033[H"

#define WIN_MIME 1
#define WIN_RANDOM 2
#define STALEMATE 3

class ExposiveSurakartaDaemon : public SurakartaDaemon {
   public:
    ExposiveSurakartaDaemon(
        int board_size,
        int max_no_capture_round,
        std::shared_ptr<AgentFactory> black_agent_factory,
        std::shared_ptr<AgentFactory> white_agent_factory)
        : SurakartaDaemon(board_size, max_no_capture_round, black_agent_factory, white_agent_factory) {}

    std::shared_ptr<SurakartaGameInfo> GameInfo() {
        return game_.GetGameInfo();
    }

    std::shared_ptr<SurakartaBoard> Board() {
        return game_.GetBoard();
    }
};

int play(std::string model_path,
         int miliseconds = 50,
         bool display = true,
         int simulation_per_move = 50,
         float cpuct = 1.0f,
         float temperature = 1.0f,
         int traditional_agent_depth = SurakartaMoveWeightUtil::DefaultDepth,
         double traditional_agent_alpha = SurakartaMoveWeightUtil::DefaultAlpha,
         double traditional_agent_beta = SurakartaMoveWeightUtil::DefaultBeta) {
    const auto model_factory = std::make_shared<SurakartaAlphazeroNeuralNetworkFactory>();
    const auto agent_factory_alphazero = std::make_shared<SurakartaAgentAlphazeroFactory>(
        model_factory->LoadModel(model_path),
        simulation_per_move,
        cpuct,
        temperature);
    const auto move_weight_util_factory = std::make_shared<SurakartaAgentMineFactory::SurakartaMoveWeightUtilFactory>(traditional_agent_depth, traditional_agent_alpha, traditional_agent_beta);
    const auto agent_factory_traditional = std::make_shared<SurakartaAgentRandomFactory>();
    const auto my_colour = GlobalRandomGenerator().getInstance()() % 2 ? PieceColor::BLACK : PieceColor::WHITE;
    const auto agent_factory_black = my_colour == PieceColor::BLACK ? (std::shared_ptr<SurakartaDaemon::AgentFactory>)agent_factory_alphazero : agent_factory_traditional;
    const auto agent_factory_white = my_colour == PieceColor::WHITE ? (std::shared_ptr<SurakartaDaemon::AgentFactory>)agent_factory_alphazero : agent_factory_traditional;
    auto daemon = ExposiveSurakartaDaemon(BOARD_SIZE, MAX_NO_CAPTURE_ROUND, agent_factory_black, agent_factory_white);

    const auto black_pieces = std::make_shared<std::vector<SurakartaPositionWithId>>();
    const auto white_pieces = std::make_shared<std::vector<SurakartaPositionWithId>>();
    bool piece_lists_initialized = false;
    SurakartaOnBoardUpdateUtil on_board_update_util(black_pieces, white_pieces, daemon.Board());

    daemon.OnUpdateBoard.AddListener([&]() {
        if (!piece_lists_initialized) {
            const auto lists = SurakartaInitPositionListsUtil(daemon.Board()).InitPositionList();
            *black_pieces = *lists.black_list;
            *white_pieces = *lists.white_list;
            piece_lists_initialized = true;
        }
        const auto opt_trace = on_board_update_util.UpdateAndGetTrace();
        if (opt_trace.has_value()) {
            PieceColor moved_colour = PieceColor::NONE;
            for (auto& item : *black_pieces) {
                if (item.id == opt_trace->moved_piece.id)
                    moved_colour = PieceColor::BLACK;
            }
            for (auto& item : *white_pieces) {
                if (item.id == opt_trace->moved_piece.id)
                    moved_colour = PieceColor::WHITE;
            }
            if (moved_colour == PieceColor::NONE)
                throw std::runtime_error("moved piece not found in black_pieces or white_pieces");
            const auto guard = opt_trace.value().is_capture ? SurakartaTemporarilyChangeColorGuardUtil(
                                                                  daemon.Board(),
                                                                  SurakartaPosition(
                                                                      opt_trace.value().captured_piece.x,
                                                                      opt_trace.value().captured_piece.y),
                                                                  ReverseColor(moved_colour))
                                                            : SurakartaTemporarilyChangeColorGuardUtil();
            for (auto& fragment : opt_trace.value().path) {
                SurakartaTemporarilyChangeColorGuardUtil guard1(daemon.Board(), fragment.From(), PieceColor::NONE);
                SurakartaTemporarilyChangeColorGuardUtil guard2(daemon.Board(), fragment.To(), moved_colour);
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    fragment.is_curve ? miliseconds * 2 : 0));

                if (display) {
                    std::cout << ANSI_CLEAR_SCREEN << ANSI_MOVE_TO_START;
                    std::cout << "B: " << (my_colour == PieceColor::BLACK ? "Alphazero" : "Traditional") << std::endl;
                    std::cout << "W: " << (my_colour == PieceColor::WHITE ? "Alphazero" : "Traditional") << std::endl;
                    std::cout << std::endl;
                    std::cout << *daemon.Board() << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
            }
        }
    });

    daemon.Execute();

    const bool is_stalemate = daemon.GameInfo()->Winner() == SurakartaPlayer::NONE;
    const bool has_win = !((daemon.GameInfo()->Winner() == SurakartaPlayer::BLACK) ^ (my_colour == PieceColor::BLACK));
    return is_stalemate ? STALEMATE : (has_win ? WIN_MIME : WIN_RANDOM);
}

int main(int argc, char** argv) {
    int simulation_per_move = 50;
    float cpuct = 1.0f;
    float temperature = 1.0f;
    int depth = SurakartaMoveWeightUtil::DefaultDepth;
    double alpha = SurakartaMoveWeightUtil::DefaultAlpha;
    double beta = SurakartaMoveWeightUtil::DefaultBeta;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--simulation") == 0 || strcmp(argv[i], "-s") == 0) {
            simulation_per_move = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpuct") == 0 || strcmp(argv[i], "-c") == 0) {
            cpuct = atof(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 || strcmp(argv[i], "-t") == 0) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--traditional-agent-depth") == 0 || strcmp(argv[i], "-d") == 0) {
            depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--traditional-agent-alpha") == 0 || strcmp(argv[i], "-a") == 0) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--traditional-agent-beta") == 0 || strcmp(argv[i], "-b") == 0) {
            beta = atof(argv[++i]);
        }
    }
    if (argc > 2 && strcmp(argv[1], "play") == 0) {
        std::string model_path = argv[2];
        int delay = 500;
        for (int i = 1; i < argc - 1; i++) {
            if (strcmp(argv[i], "--delay") == 0 || strcmp(argv[i], "-D") == 0) {
                delay = atoi(argv[++i]);
            }
        }
        play(model_path, delay, true, simulation_per_move, cpuct, temperature, depth, alpha, beta);
    } else if (argc > 2 && strcmp(argv[1], "statistic") == 0) {
        std::string model_path = argv[2];
        int concurrency = std::thread::hardware_concurrency();
        int total_rounds = INT_MAX;
        for (int i = 1; i < argc - 1; i++) {
            if (strcmp(argv[i], "-j") == 0) {
                concurrency = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-n") == 0) {
                total_rounds = atoi(argv[++i]);
            }
        }

        int cnt_play = 0;
        int cnt_win = 0;
        int cnt_lost = 0;

        const auto threads = std::make_unique<std::thread[]>(concurrency);
        for (int i = 0; i < concurrency; i++) {
            threads[i] = std::thread([&]() {
                while (true) {
                    const int result = play(model_path, 0, false, simulation_per_move, cpuct, temperature, depth, alpha, beta);
                    cnt_play++;
                    if (result == WIN_MIME)
                        cnt_win++;
                    if (result == WIN_RANDOM)
                        cnt_lost++;
                    if (cnt_play <= total_rounds)
                        printf(
                            "Win Rate: %6.2f   Not Lost Rate: %6.2f   Win: %5d Lost: %5d Stalemate: %5d\n",
                            100.0 * cnt_win / cnt_play,
                            100.0 * (cnt_play - cnt_lost) / cnt_play,
                            cnt_win,
                            cnt_lost,
                            cnt_play - cnt_win - cnt_lost);
                    if (cnt_play >= total_rounds)
                        break;
                }
            });
        }
        for (int i = 0; i < concurrency; i++) {
            threads[i].join();
        }
    } else {
        std::cout << "Usage: " << argv[0] << " play <model_path> [args..] [--delay|-D <delay>]" << std::endl;
        std::cout << "       " << argv[0] << " statistic <model_path> [args..] [-j <concurrency>] [-n <total_rounds>]" << std::endl;
        std::cout << "Description:" << std::endl;
        std::cout << "  play       Play a game between alphazero and the tradtion agent" << std::endl;
        std::cout << "  statistic  Run a statistic test between alphazero and the tradtion agent" << std::endl;
        std::cout << "Args:" << std::endl;
        std::cout << "  --simulation|-s <int>                  Number of simulations per move, default: 50" << std::endl;
        std::cout << "  --cpuct|-c <float>                     CPUCT value, default: 1.0" << std::endl;
        std::cout << "  --temperature|-t <float>               Temperature value, default: 1.0" << std::endl;
        std::cout << "  --traditional-agent-depth|-d <depth>  The depth of the search tree for traditional agent, default: " << SurakartaMoveWeightUtil::DefaultDepth << std::endl;
        std::cout << "  --traditional-agent-alpha|-a <alpha>  The reduce rate for being captured for traditional agent, default: " << SurakartaMoveWeightUtil::DefaultAlpha << std::endl;
        std::cout << "  --traditional-agent-beta|-b <beta>    The reduce rate for capturing for traditional agent, default: " << SurakartaMoveWeightUtil::DefaultBeta << std::endl;
    }
    return 0;
}
#include "surakarta.h"

class SurakartaAlphazeroNeuralNetworkBase {
   public:
    ~SurakartaAlphazeroNeuralNetworkBase() = default;

    typedef struct {
        SurakartaMove move;
        float probability;
    } MoveWithProbability;

    typedef struct {
        std::unique_ptr<std::vector<MoveWithProbability>> move_probabilities;
        float current_status_value;
    } NeuralNetworkOutput;

    virtual NeuralNetworkOutput Predict(
        std::shared_ptr<SurakartaBoard> board,
        std::shared_ptr<SurakartaGameInfo> game_info,
        PieceColor my_color) = 0;
};

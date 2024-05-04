// This class is a cpp re-implementation of https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

#include "surakarta.h"
#include "surakarta_alphazero_neural_network_base.h"

class SurakartaAlphazeroMCTS {
   public:
    SurakartaAlphazeroMCTS(std::shared_ptr<SurakartaBoard> board,
                           std::shared_ptr<SurakartaGameInfo> game_info,
                           PieceColor my_color,
                           std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> neural_network,
                           float cpuct);
    ~SurakartaAlphazeroMCTS();

    typedef struct {
        SurakartaMove move;
        float probability;
    } MoveWithProbability;

    /// @brief
    /// Calculate the move probabilities using MCTS. This probability is used to select the next move:
    /// In real playing (not training), temperature should be set to 0, and the method will return
    /// [0, ..., 0, 1, 0, ..., 0] where 1 is the best move.
    /// @return
    /// A vector of moves with their probabilities.
    std::unique_ptr<std::vector<MoveWithProbability>>
    CalculateMoveProbabilities(float temperature);  // def getActionProb(self, canonicalBoard, temp=1):

    void Simulate();

   private:
    std::shared_ptr<SurakartaBoard> board_;
    std::shared_ptr<SurakartaGameInfo> game_info_;
    PieceColor my_color_;
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> neural_network_;
    SurakartaGetAllLegalMovesUtil possible_moves_util_;
    const float cpuct_;  // self.args.cpuct

    struct Node {
        std::vector<SurakartaMove> possible_moves_;
        int simulation_count_;                                            // self.Ns[s]
        float Q;                                                          // self.Qsa[(father status, father to this status action)]
        std::vector<std::unique_ptr<Node>> childs_;                       // This vector should have the same size as possible_moves_
        std::vector<float> neural_network_predicted_move_probabilities_;  // self.Ps[s][.]
                                                                          // This vector should have the same size as possible_moves_
        float neural_network_predicted_value_;                            // self.Vs[s]
    };
    std::unique_ptr<Node> CreateNode();

    std::unique_ptr<Node> root_;

    float SimulateAndReturnValue(Node& node);  // def getActionProb(self, canonicalBoard, temp=1):
};

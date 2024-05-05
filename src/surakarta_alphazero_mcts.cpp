// This class is a cpp re-implementation of https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

#include "surakarta_alphazero_mcts.h"
#include <assert.h>
#include <algorithm>

SurakartaAlphazeroMCTS::SurakartaAlphazeroMCTS(
    std::shared_ptr<SurakartaBoard> board,
    std::shared_ptr<SurakartaGameInfo> game_info,
    PieceColor my_color,
    std::shared_ptr<SurakartaAlphazeroNeuralNetworkBase> neural_network,
    float cpuct)
    : board_(board),
      game_info_(game_info),
      my_color_(my_color),
      neural_network_(neural_network),
      possible_moves_util_(board),
      cpuct_(cpuct) {
    root_ = CreateNode();
}

SurakartaAlphazeroMCTS::~SurakartaAlphazeroMCTS() {}

std::unique_ptr<SurakartaAlphazeroMCTS::Node> SurakartaAlphazeroMCTS::CreateNode() {
    auto node = std::make_unique<Node>();
    node->possible_moves_ = *possible_moves_util_.GetAllLegalMoves(my_color_);
    node->simulation_count_ = 0;
    node->Q = 0;
    node->childs_.resize(node->possible_moves_.size());

    /*
    if s not in self.Ps:
        # leaf node
        self.Ps[s], v = self.nnet.predict(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
        sum_Ps_s = np.sum(self.Ps[s])
        if sum_Ps_s > 0:
            self.Ps[s] /= sum_Ps_s  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable
            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            log.error("All valid moves were masked, doing a workaround.")
            self.Ps[s] = self.Ps[s] + valids
            self.Ps[s] /= np.sum(self.Ps[s])

        self.Vs[s] = valids
        self.Ns[s] = 0
        return -v
    */
    const auto neural_network_output = neural_network_->Predict({.board = std::make_unique<SurakartaBoard>(*board_),
                                                                 .game_info = *game_info_,
                                                                 .my_color = my_color_});
    node->neural_network_predicted_move_probabilities_.resize(node->possible_moves_.size());
    if (node->possible_moves_.size() > 0) {
        for (int i = 0; i < node->possible_moves_.size(); i++) {
            node->neural_network_predicted_move_probabilities_[i] = 0;
        }
        for (auto& output_entry : *neural_network_output.move_probabilities) {
            int move_index = -1;
            for (int i = 0; i < node->possible_moves_.size(); i++) {
                if (node->possible_moves_[i].from == output_entry.move.from && node->possible_moves_[i].to == output_entry.move.to) {
                    move_index = i;
                    break;
                }
            }
            if (move_index >= 0) {  // is valid move
                node->neural_network_predicted_move_probabilities_[move_index] = output_entry.probability;
            }
        }
        const auto sum = std::accumulate(
            node->neural_network_predicted_move_probabilities_.begin(), node->neural_network_predicted_move_probabilities_.end(), 0.0f);
        if (sum > 0) {
            for (auto& probability : node->neural_network_predicted_move_probabilities_) {
                probability /= sum;
            }
        } else {
            fprintf(stderr, "All valid moves were masked, doing a workaround.\n");
            for (auto& probability : node->neural_network_predicted_move_probabilities_) {
                probability = 1.0f / node->possible_moves_.size();
            }
        }
    }
    node->Q = neural_network_output.current_status_value;
    node->neural_network_predicted_value_ = neural_network_output.current_status_value;
    return node;
}

// def getActionProb(self, canonicalBoard, temp=1):
std::unique_ptr<std::vector<SurakartaAlphazeroMCTS::MoveWithProbability>>
SurakartaAlphazeroMCTS::CalculateMoveProbabilities(float temperature) const {
    /*
    s = self.game.stringRepresentation(canonicalBoard)
    counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

    if temp == 0:
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.choice(bestAs)
        probs = [0] * len(counts)
        probs[bestA] = 1
        return probs

    counts = [x ** (1. / temp) for x in counts]
    counts_sum = float(sum(counts))
    probs = [x / counts_sum for x in counts]
    return probs
    */

    if (root_->possible_moves_.size() == 0) {
        return std::make_unique<std::vector<MoveWithProbability>>();
    }
    int simulation_count_max = 0;
    for (int i = 0; i < root_->possible_moves_.size(); i++) {
        if (root_->childs_[i] != nullptr && root_->childs_[i]->simulation_count_ > simulation_count_max) {
            simulation_count_max = root_->childs_[i]->simulation_count_;
        }
    }
    if (simulation_count_max == 0)
        throw std::runtime_error("SurakartaAlphazeroMCTS::Simulate() should be called more than twice, but it's not");
    if (temperature == 0) {
        auto best_move_indexes = std::vector<int>();
        for (int i = 0; i < root_->possible_moves_.size(); i++) {
            if (root_->childs_[i] != nullptr && root_->childs_[i]->simulation_count_ == simulation_count_max) {
                best_move_indexes.push_back(i);
            }
        }
        const auto best_move_index = best_move_indexes[GlobalRandomGenerator::getInstance()() % best_move_indexes.size()];
        auto ret = std::make_unique<std::vector<MoveWithProbability>>(root_->possible_moves_.size());
        for (int i = 0; i < root_->possible_moves_.size(); i++) {
            if (i == best_move_index) {
                (*ret)[i] = {root_->possible_moves_[i], 1.0f};
            } else {
                (*ret)[i] = {root_->possible_moves_[i], 0.0f};
            }
        }
        return ret;
    } else {
        auto counts_with_temperature = std::vector<float>();
        auto counts_with_temperature_moves = std::vector<SurakartaMove>();
        for (int i = 0; i < root_->possible_moves_.size(); i++) {
            if (root_->childs_[i] != nullptr) {
                assert(root_->childs_[i]->simulation_count_ > 0);
                counts_with_temperature.push_back(std::pow(root_->childs_[i]->simulation_count_, 1.0f / temperature));
                counts_with_temperature_moves.push_back(root_->possible_moves_[i]);
            }
        }
        const auto counts_sum = std::accumulate(counts_with_temperature.begin(), counts_with_temperature.end(), 0.0f);
        auto ret = std::make_unique<std::vector<MoveWithProbability>>(counts_with_temperature_moves.size());
        for (int i = 0; i < counts_with_temperature_moves.size(); i++) {
            (*ret)[i] = {counts_with_temperature_moves[i], counts_with_temperature[i] / counts_sum};
        }
        return ret;
    }
}

void SurakartaAlphazeroMCTS::Simulate() {
    SimulateAndReturnValue(*root_);
}

SurakartaAlphazeroNeuralNetworkBase::TrainEntry SurakartaAlphazeroMCTS::GetTrainEntriesWithoutValue() const {
    auto ret = SurakartaAlphazeroNeuralNetworkBase::TrainEntry();
    ret.input = std::make_unique<SurakartaAlphazeroNeuralNetworkBase::NeuralNetworkInput>();
    ret.input->board = std::make_unique<SurakartaBoard>(*board_);
    ret.input->game_info = *game_info_;
    ret.input->my_color = my_color_;
    ret.output = std::make_unique<SurakartaAlphazeroNeuralNetworkBase::NeuralNetworkOutput>();
    ret.output->current_status_value = 0.0f;
    ret.output->move_probabilities = std::make_unique<std::vector<SurakartaAlphazeroNeuralNetworkBase::MoveWithProbability>>();
    int simulation_count_debug = 0;
    for (int i = 0; i < root_->possible_moves_.size(); i++) {
        if (root_->childs_[i] != nullptr) {
            auto move_with_probability = SurakartaAlphazeroNeuralNetworkBase::MoveWithProbability();
            move_with_probability.move = root_->possible_moves_[i];
            move_with_probability.probability = static_cast<float>(root_->childs_[i]->simulation_count_) / root_->simulation_count_;
            ret.output->move_probabilities->push_back(move_with_probability);
            simulation_count_debug += root_->childs_[i]->simulation_count_;
        }
    }
    assert(simulation_count_debug == root_->simulation_count_);
    return ret;
}

/*
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
*/
// This method do not return the negative value of the value of the current status as the original code does.
float SurakartaAlphazeroMCTS::SimulateAndReturnValue(Node& node) {
    /*
    if (s, a) in self.Qsa:
        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
    else:
        self.Qsa[(s, a)] = v
        self.Nsa[(s, a)] = 1

    self.Ns[s] += 1
    return -v
    */
#define RETURN_VALUE(value)                                                                  \
    {                                                                                        \
        node.Q = (node.simulation_count_ * node.Q + (value)) / (node.simulation_count_ + 1); \
        node.simulation_count_++;                                                            \
        return (value);                                                                      \
    }

    /*
    s = self.game.stringRepresentation(canonicalBoard)
    if s not in self.Es:
        self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
    if self.Es[s] != 0:
        # terminal node
        return -self.Es[s]
    */
    if (game_info_->IsEnd()) {
        if (game_info_->Winner() == my_color_)
            RETURN_VALUE(1.0f)
        else if (game_info_->Winner() == ReverseColor(my_color_))
            RETURN_VALUE(-1.0f)
        else
            RETURN_VALUE(0.0f)
    }
    if (node.possible_moves_.size() == 0) {
        RETURN_VALUE(-1.0f)  // cannot move, thus lose
    }

    /*
    valids = self.Vs[s]
    cur_best = -float('inf')
    best_act = -1

    # pick the action with the highest upper confidence bound
    for a in range(self.game.getActionSize()):
        if valids[a]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

    a = best_act
    next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
    next_s = self.game.getCanonicalForm(next_s, next_player)

    v = self.search(next_s)
    */
    float current_best = -std::numeric_limits<float>::infinity();
    int best_move_index;
    for (int i = 0; i < node.possible_moves_.size(); i++) {
        float u;
        if (node.childs_[i] != nullptr) {
            u = node.childs_[i]->Q +
                cpuct_ * node.neural_network_predicted_move_probabilities_[i] * std::sqrt(node.simulation_count_) /
                    (1 + node.childs_[i]->simulation_count_);
        } else {
            u = cpuct_ * node.neural_network_predicted_move_probabilities_[i] * std::sqrt(node.simulation_count_);
        }
        if (u > current_best) {
            current_best = u;
            best_move_index = i;
        }
    }

    SurakartaTemporarilyApplyMoveWithGameInfoGuardUtil guard(board_, game_info_, node.possible_moves_[best_move_index]);
    my_color_ = ReverseColor(my_color_);
    if (node.childs_[best_move_index] == nullptr) {
        node.childs_[best_move_index] = CreateNode();
        node.childs_[best_move_index]->simulation_count_++;
        RETURN_VALUE(-node.childs_[best_move_index]->neural_network_predicted_value_)
    }
    const auto value = -SimulateAndReturnValue(*node.childs_[best_move_index]);
    my_color_ = ReverseColor(my_color_);
    RETURN_VALUE(value)
}

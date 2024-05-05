#include "surakarta_alphazero_neural_network_factory.h"
#include "tiny_dnn/tiny_dnn.h"

constexpr int input_vector_size = BOARD_SIZE * BOARD_SIZE + 2;
constexpr int output_vector_size_probabilities = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE;
constexpr int output_vector_size_value = 1;
constexpr int hidden_layer_size = (input_vector_size + output_vector_size_probabilities + 1);

static tiny_dnn::tensor_t ConvertInput(SurakartaAlphazeroNeuralNetworkBase::NeuralNetworkInput& input) {
    tiny_dnn::vec_t vec(input_vector_size);
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            const auto color = (*input.board)[i][j]->GetColor();
            vec[i * BOARD_SIZE + j] =
                color == input.my_color                 ? 1.0f
                : color == ReverseColor(input.my_color) ? -1.0f
                                                        : 0.0f;
        }
    }
    vec[input_vector_size - 2] = input.game_info.num_round_;
    vec[input_vector_size - 1] = input.game_info.last_captured_round_;
    return {vec};
}

static SurakartaAlphazeroNeuralNetworkBase::NeuralNetworkOutput ConvertOutput(tiny_dnn::tensor_t output) {
    const auto& probability_output = output[0];
    const auto& value_output = output[1];
    SurakartaAlphazeroNeuralNetworkBase::NeuralNetworkOutput ret;
    ret.move_probabilities = std::make_unique<std::vector<SurakartaAlphazeroNeuralNetworkBase::MoveWithProbability>>();
    for (int i = 0; i < probability_output.size(); i++) {
        SurakartaAlphazeroNeuralNetworkBase::MoveWithProbability move_with_probability{
            .move = SurakartaMove(
                i / (BOARD_SIZE * BOARD_SIZE) / BOARD_SIZE,
                i / (BOARD_SIZE * BOARD_SIZE) % BOARD_SIZE,
                i % (BOARD_SIZE * BOARD_SIZE) / BOARD_SIZE,
                i % (BOARD_SIZE * BOARD_SIZE) % BOARD_SIZE,
                SurakartaPlayer::UNKNOWN),
            .probability = probability_output[i],
        };
        ret.move_probabilities->push_back(move_with_probability);
    }
    ret.current_status_value = value_output[0];
    return ret;
}

/// @return first: probability_output, second: value_output
static tiny_dnn::tensor_t ConvertOutput(
    SurakartaAlphazeroNeuralNetworkBase::NeuralNetworkOutput& output) {
    tiny_dnn::vec_t probability_output(output_vector_size_probabilities);
    tiny_dnn::vec_t value_output(1);
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int k = 0; k < BOARD_SIZE; k++) {
                for (int l = 0; l < BOARD_SIZE; l++) {
                    const int index = i * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE + j * BOARD_SIZE * BOARD_SIZE + k * BOARD_SIZE + l;
                    int found_index = -1;
                    for (int m = 0; m < output.move_probabilities->size(); m++) {
                        const auto move = output.move_probabilities->at(m).move;
                        if (move.from.x == i &&
                            move.from.y == j &&
                            move.to.x == k &&
                            move.to.y == l) {
                            found_index = m;
                            break;
                        }
                    }
                    if (found_index == -1) {
                        probability_output[index] = 0.0f;
                    } else {
                        probability_output[index] = output.move_probabilities->at(found_index).probability;
                    }
                }
            }
        }
    }
    value_output[0] = output.current_status_value;
    return {probability_output, value_output};
}

class SurakartaAlphazeroNeuralNetworkImpl : public SurakartaAlphazeroNeuralNetworkBase {
   public:
    virtual NeuralNetworkOutput Predict(NeuralNetworkInput input) override {
        const auto input_tensor = ConvertInput(input);
        const auto output_tensor = network_->predict(input_tensor);
        return ConvertOutput(output_tensor);
    }

    virtual void Train(std::unique_ptr<std::vector<TrainEntry>> train_data) override {
        std::vector<tiny_dnn::tensor_t> input_tensor(train_data->size());
        std::vector<tiny_dnn::tensor_t> output_tensor(train_data->size());
        for (int i = 0; i < train_data->size(); i++) {
            input_tensor[i] = ConvertInput(*train_data->at(i).input);
            output_tensor[i] = ConvertOutput(*train_data->at(i).output);
        }
        tiny_dnn::adam optimizer;
        network_->fit<tiny_dnn::mse>(optimizer, input_tensor, output_tensor, train_batch_size, epochs);
    }

    virtual void SaveModel(const std::string& model_path) override {
        network_->save(model_path);
    }

   private:
    friend class SurakartaAlphazeroNeuralNetworkFactory;
    std::unique_ptr<tiny_dnn::network<tiny_dnn::graph>> network_;
    const size_t train_batch_size;
    const size_t epochs;

    SurakartaAlphazeroNeuralNetworkImpl(size_t train_batch_size, size_t epochs, std::unique_ptr<tiny_dnn::network<tiny_dnn::graph>> network_)
        : train_batch_size(train_batch_size), epochs(epochs), network_(std::move(network_)) {}
};

static void CreateModelToFile(const std::string& model_path) {
    tiny_dnn::input_layer in(input_vector_size);
    tiny_dnn::fully_connected_layer hidden_1(input_vector_size, hidden_layer_size);
    tiny_dnn::tanh_layer hidden_1_activation(hidden_layer_size);
    tiny_dnn::fully_connected_layer hidden_2(hidden_layer_size, hidden_layer_size);
    tiny_dnn::tanh_layer hidden_2_activation(hidden_layer_size);
    tiny_dnn::fully_connected_layer hidden_3(hidden_layer_size, hidden_layer_size);
    tiny_dnn::tanh_layer hidden_3_activation(hidden_layer_size);
    tiny_dnn::fully_connected_layer hidden_4(hidden_layer_size, hidden_layer_size);
    tiny_dnn::tanh_layer hidden_4_activation(hidden_layer_size);
    tiny_dnn::fully_connected_layer hidden_5(hidden_layer_size, hidden_layer_size);
    tiny_dnn::tanh_layer hidden_5_activation(hidden_layer_size);
    tiny_dnn::fully_connected_layer out_1(hidden_layer_size, output_vector_size_probabilities);
    tiny_dnn::sigmoid_layer out_1_activation(output_vector_size_probabilities);
    tiny_dnn::fully_connected_layer out_2(hidden_layer_size, output_vector_size_value);
    tiny_dnn::tanh_layer out_2_activation(output_vector_size_value);
    auto& tmp = in << hidden_1 << hidden_1_activation
                   << hidden_2 << hidden_2_activation
                   << hidden_3 << hidden_3_activation
                   << hidden_4 << hidden_4_activation
                   << hidden_5 << hidden_5_activation;
    tmp << out_1 << out_1_activation;
    tmp << out_2 << out_2_activation;
    tiny_dnn::network<tiny_dnn::graph> network;
    tiny_dnn::construct_graph(network, {&in}, {&out_1_activation, &out_2_activation});
    network.save(model_path);
}

std::unique_ptr<SurakartaAlphazeroNeuralNetworkBase> SurakartaAlphazeroNeuralNetworkFactory::CreateModel(const std::string& model_path) {
    CreateModelToFile(model_path);
    auto network = std::make_unique<tiny_dnn::network<tiny_dnn::graph>>();
    network->load(model_path);
    auto impl = new SurakartaAlphazeroNeuralNetworkImpl(train_batch_size_, epochs_, std::move(network));
    return std::unique_ptr<SurakartaAlphazeroNeuralNetworkImpl>(impl);
}

std::unique_ptr<SurakartaAlphazeroNeuralNetworkBase> SurakartaAlphazeroNeuralNetworkFactory::LoadModel(const std::string& model_path) {
    auto network = std::make_unique<tiny_dnn::network<tiny_dnn::graph>>();
    network->load(model_path);
    auto impl = new SurakartaAlphazeroNeuralNetworkImpl(train_batch_size_, epochs_, std::move(network));
    return std::unique_ptr<SurakartaAlphazeroNeuralNetworkImpl>(impl);
}

#include "surakarta_alphazero_neural_network_base.h"

class SurakartaAlphazeroNeuralNetworkFactory : public SurakartaAlphazeroNeuralNetworkBase::ModelFactory {
   public:
    SurakartaAlphazeroNeuralNetworkFactory(size_t train_batch_size, size_t epochs)
        : train_batch_size_(train_batch_size), epochs_(epochs) {}
    virtual std::unique_ptr<SurakartaAlphazeroNeuralNetworkBase> CreateModel(const std::string& model_path) override;
    virtual std::unique_ptr<SurakartaAlphazeroNeuralNetworkBase> LoadModel(const std::string& model_path) override;

   private:
    const size_t train_batch_size_;
    const size_t epochs_;
};

#include <iostream>
#include "cerebrum/layers/Dense.h"
#include "cerebrum/layers/Input.h"
#include "cerebrum/layers/Softmax.h"
#include "cerebrum/layers/ReLU.h"
#include "cerebrum/layers/Conv2D.h"
#include "cerebrum/utils/TensorOps.h"

#include <unsupported/Eigen/CXX11/Tensor>

int main()
{
    auto *input    = new Input({2, 28, 28, 1});
    auto *convOne  = new Conv2D(input, 8, 3, 1, "same");
    auto *reluOne  = new ReLU(convOne);
    auto *convTwo  = new Conv2D(reluOne, 8, 3, 1, "same");
    auto *reluTwo  = new ReLU(convTwo);
    auto *dense    = new Dense(reluTwo, 2);
    auto *softmax  = new Softmax(dense);

    std::vector<Layer *> layers = {convOne, reluOne, convTwo, reluTwo, dense, softmax};

    Eigen::Tensor<double, 4, 0, long> inputs(2, 28, 28, 1);
    inputs.setZero();

    for(std::size_t i = 0; i < 28; i++) {
        for(std::size_t j = 0; j < 28; j++) {
            inputs(1, i, j, 0) = 1;
        }
    }

    Eigen::Tensor<double, 4, 0, long> outputs(2, 1, 1, 2);
    outputs.setZero();
    outputs(0, 0, 0, 0) = 1;
    outputs(1, 0, 0, 1) = 1;

    for(std::size_t i = 0; i < 10000; i++) {
        Eigen::Tensor<double, 4, 0, long> x = inputs;

        for(Layer *layer : layers) {
            x = layer->forward(x);
        }

        Eigen::Tensor<double, 4, 0, long> deltas = x - outputs;

        if((i % 1000) == 0) {
            Eigen::Tensor<double, 0> loss = deltas.abs().sum();
            loss /= loss.constant(inputs.dimension(0));

            std::cout << loss << std::endl;
        }

        for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
            Layer *layer = *it;
            deltas = layer->backward(deltas);
        }
    }

    return 0;
}

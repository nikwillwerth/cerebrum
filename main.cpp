#include <iostream>
#include "cerebrum/layers/Dense.h"
#include "cerebrum/layers/Input.h"
#include "cerebrum/layers/Softmax.h"
#include "cerebrum/layers/ReLU.h"

#include <unsupported/Eigen/CXX11/Tensor>

int main()
{
    auto *input    = new Input({2, 28, 28, 1});
    auto *denseOne = new Dense(input, 32);
    auto *reluOne  = new ReLU(denseOne);
    auto *denseTwo = new Dense(reluOne, 2);
    auto *softmax  = new Softmax(denseTwo);

    Eigen::Tensor<double, 4, 0, long> inputs(2, 28, 28, 1);
    inputs.setZero();

    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            inputs(1, i, j, 0) = 1;
        }
    }

    Eigen::Tensor<double, 4, 0, long> outputs(2, 2, 1, 1);
    outputs.setZero();
    outputs(0, 0, 0, 0) = 1;
    outputs(1, 1, 0, 0) = 1;

    std::vector<Layer *> layers = {denseOne, reluOne, denseTwo, softmax};

    for(int i = 0; i < 100000; i++) {
        Eigen::Tensor<double, 4, 0, long> x = inputs;

        for(Layer *layer : layers) {
            x = layer->forward(x);
        }

        Eigen::Tensor<double, 4, 0, long> deltas = x - outputs;

        if((i % 10000) == 0) {
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

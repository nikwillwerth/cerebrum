//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_DENSE_H
#define CEREBRUM_DENSE_H


#include "Layer.h"

class Dense: public Layer {
public:
    Dense(Layer *inputLayer, std::size_t numOutputs);

    Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) override;
    Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) override;

private:
    Eigen::Tensor<double, 2, 0, long> weights;
    Eigen::Tensor<double, 2, 0, long> biases;

    Eigen::Tensor<double, 2, 0, long> dWeights;
    Eigen::Tensor<double, 2, 0, long> dBiases;

    std::size_t numInputs;
    std::size_t numOutputs;
};


#endif //CEREBRUM_DENSE_H

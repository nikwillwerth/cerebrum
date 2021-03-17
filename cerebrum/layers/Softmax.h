//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_SOFTMAX_H
#define CEREBRUM_SOFTMAX_H


#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.h"

class Softmax: public Layer {
public:
    Softmax(Layer *inputLayer) : Layer(inputLayer) {}

    Eigen::Tensor<double, 4> forward(Eigen::Tensor<double, 4> x) override;
    Eigen::Tensor<double, 4> backward(Eigen::Tensor<double, 4> t) override;
};


#endif //CEREBRUM_SOFTMAX_H

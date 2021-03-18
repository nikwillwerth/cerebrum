//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_SOFTMAX_H
#define CEREBRUM_SOFTMAX_H


#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.h"

class Softmax: public Layer {
public:
    Softmax(Layer *inputLayer);

    Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) override;
    Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) override;

private:
    size_t numOutputs;
};


#endif //CEREBRUM_SOFTMAX_H

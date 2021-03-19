//
// Created by Nik Willwerth on 3/17/21.
//

#ifndef CEREBRUM_RELU_H
#define CEREBRUM_RELU_H


#include "Layer.h"

class ReLU: public Layer {
public:
    ReLU(Layer *inputLayer, std::string layerName = "ReLU");

    Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) override;
    Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) override;
};


#endif //CEREBRUM_RELU_H

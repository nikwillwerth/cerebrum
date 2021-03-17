//
// Created by Nik Willwerth on 3/17/21.
//

#ifndef CEREBRUM_RELU_H
#define CEREBRUM_RELU_H


#include "Layer.h"

class ReLU: public Layer {
public:
    ReLU(Layer *inputLayer);

    Eigen::Tensor<double, 4> forward(Eigen::Tensor<double, 4> x) override;
    Eigen::Tensor<double, 4> backward(Eigen::Tensor<double, 4> t) override;
};


#endif //CEREBRUM_RELU_H

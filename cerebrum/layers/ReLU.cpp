//
// Created by Nik Willwerth on 3/17/21.
//

#include "ReLU.h"
#include "../utils/TensorOps.h"

ReLU::ReLU(Layer *inputLayer) : Layer(inputLayer) {
    this->outputShape = inputLayer->outputShape;
}

Eigen::Tensor<double, 4, 0, long> ReLU::forward(Eigen::Tensor<double, 4, 0, long> x) {
    // max(0, x)
    this->previousX = x.cwiseMax(x.constant(0));

    return this->previousX;
}

Eigen::Tensor<double, 4, 0, long> ReLU::backward(Eigen::Tensor<double, 4, 0, long> t) {
    // reshape t to have shape of x
    Eigen::Tensor<double, 4, 0, long> reshapedT = TensorOps::reshape(t, this->previousX.dimension(0), this->previousX.dimension(1), this->previousX.dimension(2), this->previousX.dimension(3));

    // if input to this layer was greater than 0, we keep the gradients.
    // otherwise, we set them to 0
    return reshapedT * (this->previousX > 0.0);
}

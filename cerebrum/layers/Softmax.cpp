//
// Created by Nik Willwerth on 3/16/21.
//

#include "Softmax.h"
#include "../utils/TensorOps.h"

Softmax::Softmax(Layer *inputLayer) : Layer(inputLayer) {
    this->numOutputs = this->inputLayer->outputShape[3];
}

Eigen::Tensor<double, 4, 0, long> Softmax::forward(Eigen::Tensor<double, 4, 0, long> x) {
    // reshape x to be (batchSize, numOutputs)
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(x, this->batchSize, this->numOutputs);

    // exp(x - x.max(axis=1, keepdims=True))
    Eigen::Tensor<double, 2, 0, long> e = (reshapedX - TensorOps::broadcast(TensorOps::maxKeepDims(reshapedX, 1), 1, this->numOutputs)).exp();

    // e / sum(e, axis=1, keepdims=True)
    Eigen::Tensor<double, 2, 0, long> output = e / TensorOps::broadcast(TensorOps::sumKeepDims(e, 1), 1, this->numOutputs);

    // reshape output to be (batchSize, 1, 1, numOutputs)
    return TensorOps::reshape(output, this->batchSize, 1, 1, this->numOutputs);
}

Eigen::Tensor<double, 4, 0, long> Softmax::backward(Eigen::Tensor<double, 4, 0, long> t) {
    return t;
}

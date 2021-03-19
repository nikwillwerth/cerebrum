//
// Created by Nik Willwerth on 3/16/21.
//

#include "Softmax.h"

#include <utility>
#include <iostream>
#include "../utils/TensorOps.h"

Softmax::Softmax(Layer *inputLayer, std::string layerName) : Layer(inputLayer, std::move(layerName)) {
    this->_numOutputs = inputLayer->outputShape[3];
    this->hasWeights = false;
}

Eigen::Tensor<double, 4, 0, long> Softmax::forward(Eigen::Tensor<double, 4, 0, long> x) {
    // reshape x to be (batchSize, _numOutputs)
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(x, this->batchSize, this->_numOutputs);

    // exp(x - x.max(axis=1, keepdims=True))
    Eigen::Tensor<double, 2, 0, long> e = (reshapedX - TensorOps::broadcast(TensorOps::maxKeepDims(reshapedX, 1), 1, this->_numOutputs)).exp();

    // e / sum(e, axis=1, keepdims=True)
    Eigen::Tensor<double, 2, 0, long> output = e / TensorOps::broadcast(TensorOps::sumKeepDims(e, 1), 1, this->_numOutputs);

    // reshape output to be (batchSize, 1, 1, _numOutputs)
    return TensorOps::reshape(output, this->batchSize, 1, 1, this->_numOutputs);
}

Eigen::Tensor<double, 4, 0, long> Softmax::backward(Eigen::Tensor<double, 4, 0, long> t) {
    return t;
}

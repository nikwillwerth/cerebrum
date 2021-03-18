//
// Created by Nik Willwerth on 3/16/21.
//

#include <iostream>
#include "Dense.h"
#include "../utils/TensorOps.h"

Dense::Dense(Layer *inputLayer, std::size_t numOutputs) : Layer(inputLayer) {
    this->numOutputs = numOutputs;
    this->numInputs  = this->inputLayer->outputShape[1] * this->inputLayer->outputShape[2] * this->inputLayer->outputShape[3];

    this->weights = TensorOps::getRandomTensor(numOutputs, this->numInputs);
    this->biases  = TensorOps::getRandomTensor(1, numOutputs);

    this->outputShape = {this->batchSize, 1, 1, this->numOutputs};
}

Eigen::Tensor<double, 4, 0, long> Dense::forward(Eigen::Tensor<double, 4, 0, long> x) {
    this->previousX = x;

    // reshape x to be (batchSize, numInputs)
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(x, this->batchSize, this->numInputs);

    // x * w.t
    Eigen::Tensor<double, 2, 0, long> output = TensorOps::dot(reshapedX, TensorOps::transpose(this->weights));

    // broadcast biases from (1, numOutputs) to (batchSize, numOutputs)
    Eigen::Tensor<double, 2, 0, long> broadcastBiases = TensorOps::broadcast(this->biases, this->batchSize, 1);

    // reshape output from (batchSize, numOutputs) to (batchSize, 1, 1, numOutputs)
    Eigen::Tensor<double, 4, 0, long> reshapedOutput = TensorOps::reshape(output,          this->batchSize, 1, 1, this->numOutputs);
    Eigen::Tensor<double, 4, 0, long> reshapedBiases = TensorOps::reshape(broadcastBiases, this->batchSize, 1, 1, this->numOutputs);

    // output + biases
    return reshapedOutput + reshapedBiases;
}

Eigen::Tensor<double, 4, 0, long> Dense::backward(Eigen::Tensor<double, 4, 0, long> t) {
    // reshape t to be (batchSize, numOutputs)
    // reshape x to be (batchSize, numInputs)
    Eigen::Tensor<double, 2, 0, long> reshapedT = TensorOps::reshape(t, this->batchSize, this->numOutputs);
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(this->previousX, this->batchSize, this->numInputs);

    // t.t * x
    // t.sum(axis=0, keepdims=True)
    this->dWeights = TensorOps::dot(TensorOps::transpose(reshapedT), reshapedX) / this->weights.constant(this->batchSize);
    this->dBiases  = TensorOps::sumKeepDims(reshapedT, 0) / this->biases.constant(this->batchSize);

    this->weights = this->weights - (this->weights.constant(0.01) * this->dWeights);
    this->biases  = this->biases  - (this->biases.constant(0.01)  * this->dBiases);

    // t * w
    Eigen::Tensor<double, 2, 0, long> output = TensorOps::dot(reshapedT, this->weights);

    // reshape output to be (batchSize, 1, 1, numInputs)
    return TensorOps::reshape(output, this->batchSize, 1, 1, this->numInputs);
}

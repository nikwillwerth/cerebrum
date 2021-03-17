//
// Created by Nik Willwerth on 3/16/21.
//

#include <iostream>
#include "Dense.h"
#include "../utils/TensorOps.h"

Dense::Dense(Layer *inputLayer, std::size_t numOutputs) : Layer(inputLayer) {
    this->numInputs  = 1;
    this->numOutputs = numOutputs;

    for(std::size_t i = 1; i < this->inputLayer->outputShape.size(); i++) {
        this->numInputs *= this->inputLayer->outputShape[i];
    }

    this->weights = Eigen::Tensor<double, 2, 0, long>(long(numOutputs), long(this->numInputs));
    this->biases  = Eigen::Tensor<double, 2, 0, long>(1, long(numOutputs));

    this->weights.setRandom<Eigen::internal::NormalRandomGenerator<double>>();
    this->biases.setRandom<Eigen::internal::NormalRandomGenerator<double>>();

    this->weights *= this->weights.constant(0.1);
    this->biases  *= this->biases.constant(0.1);

    this->outputShape = {this->batchSize, this->numOutputs, 1, 1};
}

Eigen::Tensor<double, 4, 0, long> Dense::forward(Eigen::Tensor<double, 4, 0, long> x) {
    this->previousX = x;

    // reshape x to be (batchSize, numInputs)
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(x, this->batchSize, this->numInputs);

    // x * w.t
    Eigen::Tensor<double, 2, 0, long> output = TensorOps::dot(reshapedX, TensorOps::transpose(this->weights));

    // broadcast biases from (1, numOutputs) to (batchSize, numOutputs)
    Eigen::Tensor<double, 2, 0, long> broadcastBiases = TensorOps::broadcast(this->biases, this->batchSize, 1);

    // reshape output to have 4 dimensions
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

    return TensorOps::reshape(output, this->batchSize, 1, 1, this->numInputs);
}

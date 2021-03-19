//
// Created by Nik Willwerth on 3/16/21.
//

#include <utility>
#include "Dense.h"
#include "../utils/TensorOps.h"

Dense::Dense(Layer *inputLayer, std::size_t numOutputs, std::string layerName) : Layer(inputLayer, std::move(layerName)) {
    this->_numOutputs = numOutputs;
    this->_numInputs  = inputLayer->outputShape[1] * inputLayer->outputShape[2] * inputLayer->outputShape[3];

    this->_weights = TensorOps::getRandomTensor(numOutputs, this->_numInputs);
    this->_biases  = TensorOps::getRandomTensor(1, numOutputs);

    this->outputShape = {this->batchSize, 1, 1, this->_numOutputs};
}

Eigen::Tensor<double, 4, 0, long> Dense::forward(Eigen::Tensor<double, 4, 0, long> x) {
    this->previousX = x;

    // reshape x to be (batchSize, _numInputs)
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(x, this->batchSize, this->_numInputs);

    // x * w.t
    Eigen::Tensor<double, 2, 0, long> output = TensorOps::dot(reshapedX, TensorOps::transpose(this->_weights));

    // broadcast _biases from (1, _numOutputs) to (batchSize, _numOutputs)
    Eigen::Tensor<double, 2, 0, long> broadcastBiases = TensorOps::broadcast(this->_biases, this->batchSize, 1);

    // reshape output from (batchSize, _numOutputs) to (batchSize, 1, 1, _numOutputs)
    Eigen::Tensor<double, 4, 0, long> reshapedOutput = TensorOps::reshape(output,          this->batchSize, 1, 1, this->_numOutputs);
    Eigen::Tensor<double, 4, 0, long> reshapedBiases = TensorOps::reshape(broadcastBiases, this->batchSize, 1, 1, this->_numOutputs);

    // output + _biases
    return reshapedOutput + reshapedBiases;
}

Eigen::Tensor<double, 4, 0, long> Dense::backward(Eigen::Tensor<double, 4, 0, long> t) {
    // reshape t to be (batchSize, _numOutputs)
    // reshape x to be (batchSize, _numInputs)
    Eigen::Tensor<double, 2, 0, long> reshapedT = TensorOps::reshape(t, this->batchSize, this->_numOutputs);
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(this->previousX, this->batchSize, this->_numInputs);

    // t.t * x
    // t.sum(axis=0, keepdims=True)
    this->_dWeights = TensorOps::dot(TensorOps::transpose(reshapedT), reshapedX) / this->_weights.constant(this->batchSize);
    this->_dBiases  = TensorOps::sumKeepDims(reshapedT, 0) / this->_biases.constant(this->batchSize);

    // t * w
    Eigen::Tensor<double, 2, 0, long> output = TensorOps::dot(reshapedT, this->_weights);

    // reshape output to be (batchSize, 1, 1, _numInputs)
    return TensorOps::reshape(output, this->batchSize, 1, 1, this->_numInputs);
}

Eigen::Tensor<double, 2, 0, long> Dense::getWeights() {
    return this->_weights;
}

Eigen::Tensor<double, 2, 0, long> Dense::getBiases() {
    return this->_biases;
}

Eigen::Tensor<double, 2, 0, long> Dense::getDWeights() {
    return this->_dWeights;
}

Eigen::Tensor<double, 2, 0, long> Dense::getDBiases() {
    return this->_dBiases;
}

void Dense::setWeights(Eigen::Tensor<double, 2, 0, long> newWeights) {
    this->_weights = newWeights;
}

void Dense::setBiases(Eigen::Tensor<double, 2, 0, long> newBiases) {
    this->_biases = newBiases;
}

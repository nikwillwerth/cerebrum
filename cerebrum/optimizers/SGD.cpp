//
// Created by Nik Willwerth on 3/18/21.
//

#include <iostream>
#include "SGD.h"

SGD::SGD(double learningRate) {
    this->learningRate = learningRate;
}

void SGD::update(const std::vector<Layer *> &layers) {
    for(Layer *layer : layers) {
        if(layer->hasWeights) {
            Eigen::Tensor<double, 2, 0, long> weights  = layer->getWeights();
            Eigen::Tensor<double, 2, 0, long> biases   = layer->getBiases();
            Eigen::Tensor<double, 2, 0, long> dWeights = layer->getDWeights();
            Eigen::Tensor<double, 2, 0, long> dBiases  = layer->getDBiases();

            Eigen::Tensor<double, 2, 0, long> newWeights = weights - (weights.constant(this->learningRate) * dWeights);
            Eigen::Tensor<double, 2, 0, long> newBiases  = biases  - (biases.constant( this->learningRate) * dBiases);

            layer->setWeights(newWeights);
            layer->setBiases(newBiases);
        }
    }
}

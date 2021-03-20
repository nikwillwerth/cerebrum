//
// Created by Nik Willwerth on 3/18/21.
//

#include "Model.h"
#include "layers/Input.h"
#include "utils/TensorOps.h"
#include "data/Mnist.h"

#include <utility>
#include <iostream>

void Model::addLayer(Layer *layer) {
    this->_sortedLayers.emplace_back(layer);
}

void Model::compile(Optimizer *optimizer) {
    this->_optimizer = optimizer;

    if(this->_sortedLayers.empty()) {
        throw std::invalid_argument("No layers found when compiling model!");
    }

    if(dynamic_cast<Input*>(this->_sortedLayers[0]) == nullptr) {
        throw std::invalid_argument("First model layer must be an Input layer!");
    }

    // remove input layer from _layers
    this->_sortedLayers.erase(this->_sortedLayers.begin());
}

void Model::compile(Input *inputLayer, Optimizer *optimizer) {
    this->_optimizer = optimizer;

    std::vector<Layer *> visitedLayers;
    std::vector<Layer *> nodes;
    nodes.emplace_back(inputLayer);

    // topographical sort to determine order of execution of layers
    while(!nodes.empty()) {
        Layer *layer = nodes[0];
        nodes.erase(nodes.begin());

        this->_sortedLayers.emplace_back(layer);
        visitedLayers.emplace_back(layer);

        for(Layer *nextLayer : layer->outputLayers) {
            bool isDone = true;

            for(Layer *previousLayer : nextLayer->inputLayers) {
                if(!std::count(visitedLayers.begin(), visitedLayers.end(), previousLayer)) {
                    isDone = false;
                    break;
                }
            }

            if(isDone) {
                nodes.emplace_back(nextLayer);
            }
        }
    }

    // remove input layer from _layers
    this->_sortedLayers.erase(this->_sortedLayers.begin());
}

void Model::train(size_t batchSize, size_t epochs) {
    Eigen::Tensor<double, 4, 0, long> inputs(batchSize, 28, 28, 1);
    Eigen::Tensor<double, 4, 0, long> outputs(batchSize, 1, 1, 10);

    auto mnist = new Mnist();

    for(std::size_t i = 0; i < epochs; i++) {
        for(std::size_t j = 0; j < batchSize; j++) {
            Eigen::array<long, 4> offsets = {long(j), 0, 0, 0};
            Eigen::array<long, 4> extents = {1, inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)};

            inputs.slice(offsets, extents) = mnist->trainImages[j] / mnist->trainImages[j].constant(255);
        }

        for(std::size_t j = 0; j < batchSize; j++) {
            Eigen::array<long, 4> offsets = {long(j), 0, 0, 0};
            Eigen::array<long, 4> extents = {1, outputs.dimension(1), outputs.dimension(2), outputs.dimension(3)};

            outputs.slice(offsets, extents) = mnist->trainLabels[j];
        }

        Eigen::Tensor<double, 4, 0, long> x = inputs;

        for(Layer *layer : this->_sortedLayers) {
            x = layer->forward(x);
        }

        Eigen::Tensor<double, 4, 0, long> deltas = x - outputs;

        if((i % 10000) == 0) {
            Eigen::Tensor<double, 0> loss = deltas.abs().sum();
            loss /= loss.constant(inputs.dimension(0));

            std::cout << loss << std::endl;
        }

        for(auto it = this->_sortedLayers.rbegin(); it != this->_sortedLayers.rend(); ++it) {
            deltas = ((Layer *)*it)->backward(deltas);
        }

        this->_optimizer->update(this->_sortedLayers);
    }
}

//
// Created by Nik Willwerth on 3/15/21.
//

#include "Layer.h"

#include <utility>

Layer::Layer(Layer *inputLayer, std::string layerName) {
    this->name = std::move(layerName);

    if(inputLayer != nullptr) {
        this->batchSize = inputLayer->batchSize;

        this->inputLayers.emplace_back(inputLayer);
        inputLayer->outputLayers.emplace_back(this);
    }
}

Layer::Layer(std::vector<Layer *> inputLayers, std::string layerName) {
    this->name = std::move(layerName);

    if(!inputLayers.empty()) {
        // extend this->inputLayers
        this->inputLayers.insert(this->inputLayers.end(), inputLayers.begin(), inputLayers.end());

        this->batchSize = inputLayers[0]->batchSize;

        for(Layer *inputLayer : inputLayers) {
            inputLayer->outputLayers.emplace_back(this);
        }
    }
}

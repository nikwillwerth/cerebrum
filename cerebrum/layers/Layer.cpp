//
// Created by Nik Willwerth on 3/15/21.
//

#include "Layer.h"

Layer::Layer(Layer *inputLayer) {
    this->inputLayer = inputLayer;

    if(inputLayer != nullptr) {
        this->batchSize = inputLayer->batchSize;
    }
}

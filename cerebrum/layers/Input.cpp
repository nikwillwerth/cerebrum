//
// Created by Nik Willwerth on 3/16/21.
//

#include "Input.h"

#include <utility>

Input::Input(const std::vector<std::size_t> &inputShape, std::string layerName) : Layer(nullptr, std::move(layerName)) {
    this->outputShape = inputShape;
    this->batchSize   = inputShape[0];
    this->hasWeights  = false;
}

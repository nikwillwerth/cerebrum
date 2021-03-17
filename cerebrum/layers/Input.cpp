//
// Created by Nik Willwerth on 3/16/21.
//

#include "Input.h"

Input::Input(const std::vector<std::size_t> &inputShape) : Layer(nullptr) {
    this->outputShape = inputShape;
    this->batchSize   = inputShape[0];
}

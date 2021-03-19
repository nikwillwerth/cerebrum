//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_INPUT_H
#define CEREBRUM_INPUT_H


#include "Layer.h"

class Input: public Layer {
public:
    Input(const std::vector<std::size_t> &inputShape, std::string layerName = "Input");
};


#endif //CEREBRUM_INPUT_H

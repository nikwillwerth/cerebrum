//
// Created by Nik Willwerth on 3/18/21.
//

#ifndef CEREBRUM_OPTIMIZER_H
#define CEREBRUM_OPTIMIZER_H


#include <vector>
#include "../layers/Layer.h"

class Optimizer {
public:
    virtual void update(const std::vector<Layer *>& layers) {}
};


#endif //CEREBRUM_OPTIMIZER_H

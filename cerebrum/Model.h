//
// Created by Nik Willwerth on 3/18/21.
//

#ifndef CEREBRUM_MODEL_H
#define CEREBRUM_MODEL_H


#include "optimizers/SGD.h"
#include "layers/Input.h"

class Model {
public:
    void addLayer(Layer *layer);
    void compile(Optimizer *optimizer = new SGD());
    void compile(Input *inputLayer, Optimizer *optimizer = new SGD());
    void train(size_t batchSize, size_t epochs);

private:
    std::vector<Layer *> _sortedLayers;
    Optimizer *_optimizer;
};


#endif //CEREBRUM_MODEL_H

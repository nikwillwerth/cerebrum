//
// Created by Nik Willwerth on 3/18/21.
//

#ifndef CEREBRUM_SGD_H
#define CEREBRUM_SGD_H


#include "Optimizer.h"

class SGD: public Optimizer {
public:
    explicit SGD(double learningRate = 0.01);

    void update(const std::vector<Layer *>& layers) override;

private:
    double learningRate;
};


#endif //CEREBRUM_SGD_H

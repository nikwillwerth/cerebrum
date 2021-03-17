//
// Created by Nik Willwerth on 3/15/21.
//

#ifndef CEREBRUM___LAYER_H
#define CEREBRUM___LAYER_H

#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

class Layer {
public:
    explicit Layer(Layer *inputLayer);

    virtual Eigen::Tensor<double, 4> forward(Eigen::Tensor<double, 4> x) { return Eigen::Tensor<double, 4>(); };
    virtual Eigen::Tensor<double, 4> backward(Eigen::Tensor<double, 4> t) { return Eigen::Tensor<double, 4>(); };

    Eigen::Tensor<double, 4> previousX;

    std::vector<std::size_t> outputShape;
    std::size_t batchSize;

protected:
    Layer *inputLayer;
};


#endif //CEREBRUM___LAYER_H

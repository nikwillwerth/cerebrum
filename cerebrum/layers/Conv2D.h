//
// Created by Nik Willwerth on 3/17/21.
//

#ifndef CEREBRUM_CONV2D_H
#define CEREBRUM_CONV2D_H


#include "Layer.h"

class Conv2D: public Layer {
public:
    Conv2D(Layer *inputLayer, std::size_t numFilters, std::size_t filterSize, std::size_t stride = 1, const std::string& padding = "same");

    Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) override;
    Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) override;

private:
    Eigen::Tensor<double, 2, 0, long> im2col(Eigen::Tensor<double, 4, 0, long> x);
    Eigen::Tensor<double, 4, 0, long> col2im(Eigen::Tensor<double, 2, 0, long> x);

    Eigen::Tensor<double, 4, 0, long> weights;
    Eigen::Tensor<double, 1, 0, long> biases;

    Eigen::Tensor<double, 4, 0, long> dWeights;
    Eigen::Tensor<double, 1, 0, long> dBiases;

    Eigen::Tensor<double, 2, 0, long> cols;

    std::size_t filterSize;
    std::size_t stride;
    std::size_t pad;
    std::string padding;
};


#endif //CEREBRUM_CONV2D_H

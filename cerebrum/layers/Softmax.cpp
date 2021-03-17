//
// Created by Nik Willwerth on 3/16/21.
//

#include <iostream>
#include "Softmax.h"
#include "../utils/TensorOps.h"

Eigen::Tensor<double, 4> Softmax::forward(Eigen::Tensor<double, 4> x) {
    // reshape x to be (batchSize, numOutputs)
    Eigen::Tensor<double, 2, 0, long> reshapedX = TensorOps::reshape(x, x.dimension(0), x.dimension(1));

    // exp(x - x.max(axis=1, keepdims=True))
    Eigen::Tensor<double, 2, 0, long> e = (reshapedX - TensorOps::broadcast(TensorOps::maxKeepDims(reshapedX, 1), 1, x.dimension(1))).exp();

    // e / sum(e, axis=1, keepdims=True)
    Eigen::Tensor<double, 2, 0, long> output = e / TensorOps::broadcast(TensorOps::sumKeepDims(e, 1), 1, x.dimension(1));

    return TensorOps::reshape(output, output.dimension(0), output.dimension(1), 1, 1);
}

Eigen::Tensor<double, 4> Softmax::backward(Eigen::Tensor<double, 4> t) {
    return t;
}

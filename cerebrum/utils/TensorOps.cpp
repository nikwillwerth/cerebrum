//
// Created by Nik Willwerth on 3/16/21.
//

#include "TensorOps.h"

Eigen::Tensor<double, 2> TensorOps::dot(const Eigen::Tensor<double, 2>& a, const Eigen::Tensor<double, 2>& b) {
    Eigen::array<Eigen::IndexPair<int>, 1> dotProductDims = { Eigen::IndexPair<int>(1, 0) };

    return a.contract(b, dotProductDims);
}

Eigen::Tensor<double, 2> TensorOps::transpose(Eigen::Tensor<double, 2> x) {
    Eigen::array<int, 2> indices({1, 0});

    return x.shuffle(indices);
}

Eigen::Tensor<double, 2> TensorOps::broadcast(const Eigen::Tensor<double, 2>& x, size_t dimOne, size_t dimTwo) {
    Eigen::array<int, 2> xBroadcast({int(dimOne), int(dimTwo)});

    return x.broadcast(xBroadcast);
}

Eigen::Tensor<double, 2> TensorOps::maxKeepDims(const Eigen::Tensor<double, 2>& x, size_t axis) {
    size_t dimOne = (axis == 0) ? 1 : x.dimension(0);
    size_t dimTwo = (axis == 1) ? 1 : x.dimension(1);

    Eigen::array<int, 1> axisIndex({int(axis)});
    Eigen::Tensor<double, 1, 0, long> max = x.maximum(axisIndex);

    return TensorOps::reshape(max, dimOne, dimTwo);
}

Eigen::Tensor<double, 2> TensorOps::sumKeepDims(const Eigen::Tensor<double, 2>& x, size_t axis) {
    size_t dimOne = (axis == 0) ? 1 : x.dimension(0);
    size_t dimTwo = (axis == 1) ? 1 : x.dimension(1);

    Eigen::array<int, 1> axisIndex({int(axis)});
    Eigen::Tensor<double, 1, 0, long> sum = x.sum(axisIndex);

    return TensorOps::reshape(sum, dimOne, dimTwo);
}

Eigen::Tensor<double, 2> TensorOps::reshape(Eigen::Tensor<double, 1> x, size_t dimOne, size_t dimTwo) {
    Eigen::array<long, 2> newShape{{int(dimOne), int(dimTwo)}};

    return x.reshape(newShape);
}

Eigen::Tensor<double, 2> TensorOps::reshape(Eigen::Tensor<double, 4> x, size_t dimOne, size_t dimTwo) {
    Eigen::array<long, 2> newShape{{int(dimOne), int(dimTwo)}};

    return x.reshape(newShape);
}

Eigen::Tensor<double, 4> TensorOps::reshape(Eigen::Tensor<double, 2> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour) {
    Eigen::array<long, 4> newShape{{int(dimOne), int(dimTwo), int(dimThree), int(dimFour)}};

    return x.reshape(newShape);
}

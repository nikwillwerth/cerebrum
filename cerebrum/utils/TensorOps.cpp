//
// Created by Nik Willwerth on 3/16/21.
//

#include <iostream>
#include "TensorOps.h"

Eigen::Tensor<double, 2, 0, long> TensorOps::dot(const Eigen::Tensor<double, 2, 0, long>& a, const Eigen::Tensor<double, 2, 0, long>& b) {
    Eigen::array<Eigen::IndexPair<long>, 1> dotProductDims = { Eigen::IndexPair<long>(1, 0) };

    return a.contract(b, dotProductDims);
}

Eigen::Tensor<double, 2, 0, long> TensorOps::transpose(Eigen::Tensor<double, 2, 0, long> x) {
    Eigen::array<long, 2> indices({1, 0});

    return x.shuffle(indices);
}

Eigen::Tensor<double, 2, 0, long> TensorOps::broadcast(const Eigen::Tensor<double, 2, 0, long>& x, size_t dimOne, size_t dimTwo) {
    Eigen::array<long, 2> xBroadcast({long(dimOne), long(dimTwo)});

    return x.broadcast(xBroadcast);
}

Eigen::Tensor<double, 4, 0, long> TensorOps::broadcast(const Eigen::Tensor<double, 1, 0, long>& x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour) {
    Eigen::Tensor<double, 4, 0, long> reshapedX = TensorOps::reshape(x, 1, 1, 1, x.dimension(0));

    Eigen::array<long, 4> xBroadcast({long(dimOne), long(dimTwo), long(dimThree), long(dimFour)});

    return reshapedX.broadcast(xBroadcast);
}

Eigen::Tensor<double, 2, 0, long> TensorOps::maxKeepDims(const Eigen::Tensor<double, 2, 0, long>& x, size_t axis) {
    size_t dimOne = (axis == 0) ? 1 : x.dimension(0);
    size_t dimTwo = (axis == 1) ? 1 : x.dimension(1);

    Eigen::array<long, 1> axisIndex({long(axis)});
    Eigen::Tensor<double, 1, 0, long> max = x.maximum(axisIndex);

    return TensorOps::reshape(max, dimOne, dimTwo);
}

Eigen::Tensor<double, 2, 0, long> TensorOps::sumKeepDims(const Eigen::Tensor<double, 2, 0, long>& x, size_t axis) {
    size_t dimOne = (axis == 0) ? 1 : x.dimension(0);
    size_t dimTwo = (axis == 1) ? 1 : x.dimension(1);

    Eigen::array<long, 1> axisIndex({long(axis)});
    Eigen::Tensor<double, 1, 0, long> sum = x.sum(axisIndex);

    return TensorOps::reshape(sum, dimOne, dimTwo);
}

Eigen::Tensor<double, 2, 0, long> TensorOps::reshape(Eigen::Tensor<double, 1, 0, long> x, size_t dimOne, size_t dimTwo) {
    Eigen::array<long, 2> newShape{{long(dimOne), long(dimTwo)}};

    return x.reshape(newShape);
}

Eigen::Tensor<double, 4, 0, long> TensorOps::reshape(Eigen::Tensor<double, 1, 0, long> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour) {
    Eigen::array<long, 4> newShape{{long(dimOne), long(dimTwo), long(dimThree), long(dimFour)}};

    return x.reshape(newShape);
}

Eigen::Tensor<double, 2, 0, long> TensorOps::reshape(Eigen::Tensor<double, 4, 0, long> x, size_t dimOne, size_t dimTwo) {
    Eigen::array<long, 2> newShape{{long(dimOne), long(dimTwo)}};

    return x.reshape(newShape);
}

Eigen::Tensor<double, 4, 0, long> TensorOps::reshape(Eigen::Tensor<double, 2, 0, long> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour) {
    Eigen::array<long, 4> newShape{{long(dimOne), long(dimTwo), long(dimThree), long(dimFour)}};

    return x.reshape(newShape);
}

Eigen::Tensor<double, 4, 0, long> TensorOps::reshape(Eigen::Tensor<double, 4, 0, long> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour) {
    Eigen::array<long, 4> newShape{{long(dimOne), long(dimTwo), long(dimThree), long(dimFour)}};

    return x.reshape(newShape);
}

void TensorOps::printShape(const Eigen::Tensor<double, 1, 0, long>& x) {
    std::cout << "(" << x.dimension(0) << ", " << ")" << std::endl;
}

void TensorOps::printShape(const Eigen::Tensor<double, 2, 0, long>& x) {
    std::cout << "(" << x.dimension(0) << ", " << x.dimension(1) << ")" << std::endl;
}

void TensorOps::printShape(const Eigen::Tensor<double, 4, 0, long>& x) {
    std::cout << "(" << x.dimension(0) << ", " << x.dimension(1) << ", " << x.dimension(2) << ", " << x.dimension(3) << ")" << std::endl;
}

//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_TENSOROPS_H
#define CEREBRUM_TENSOROPS_H


#include <unsupported/Eigen/CXX11/Tensor>

class TensorOps {
public:
    static Eigen::Tensor<double, 2> dot(const Eigen::Tensor<double, 2>& a, const Eigen::Tensor<double, 2>& b);
    static Eigen::Tensor<double, 2> transpose(Eigen::Tensor<double, 2> x);
    static Eigen::Tensor<double, 2> broadcast(const Eigen::Tensor<double, 2>& x, size_t dimOne, size_t dimTwo);
    static Eigen::Tensor<double, 2> maxKeepDims(const Eigen::Tensor<double, 2>& x, size_t axis);
    static Eigen::Tensor<double, 2> sumKeepDims(const Eigen::Tensor<double, 2>& x, size_t axis);
    static Eigen::Tensor<double, 2> reshape(Eigen::Tensor<double, 1> x, size_t dimOne, size_t dimTwo);
    static Eigen::Tensor<double, 2> reshape(Eigen::Tensor<double, 4> x, size_t dimOne, size_t dimTwo);
    static Eigen::Tensor<double, 4> reshape(Eigen::Tensor<double, 2> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour);
};


#endif //CEREBRUM_TENSOROPS_H

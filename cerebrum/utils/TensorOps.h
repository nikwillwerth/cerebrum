//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_TENSOROPS_H
#define CEREBRUM_TENSOROPS_H


#include <unsupported/Eigen/CXX11/Tensor>

class TensorOps {
public:
    static Eigen::Tensor<double, 2, 0, long> dot(const Eigen::Tensor<double, 2, 0, long>& a, const Eigen::Tensor<double, 2, 0, long>& b);
    static Eigen::Tensor<double, 2, 0, long> transpose(Eigen::Tensor<double, 2, 0, long> x);
    static Eigen::Tensor<double, 2, 0, long> broadcast(const Eigen::Tensor<double, 2, 0, long>& x, size_t dimOne, size_t dimTwo);
    static Eigen::Tensor<double, 4, 0, long> broadcast(const Eigen::Tensor<double, 1, 0, long>& x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour);
    static Eigen::Tensor<double, 2, 0, long> maxKeepDims(const Eigen::Tensor<double, 2, 0, long>& x, size_t axis);
    static Eigen::Tensor<double, 2, 0, long> sumKeepDims(const Eigen::Tensor<double, 2, 0, long>& x, size_t axis);
    static Eigen::Tensor<double, 2, 0, long> reshape(Eigen::Tensor<double, 1, 0, long> x, size_t dimOne, size_t dimTwo);
    static Eigen::Tensor<double, 4, 0, long> reshape(Eigen::Tensor<double, 1, 0, long> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour);
    static Eigen::Tensor<double, 2, 0, long> reshape(Eigen::Tensor<double, 4, 0, long> x, size_t dimOne, size_t dimTwo);
    static Eigen::Tensor<double, 4, 0, long> reshape(Eigen::Tensor<double, 2, 0, long> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour);
    static Eigen::Tensor<double, 4, 0, long> reshape(Eigen::Tensor<double, 4, 0, long> x, size_t dimOne, size_t dimTwo, size_t dimThree, size_t dimFour);
    static void printShape(const Eigen::Tensor<double, 1, 0, long>& x);
    static void printShape(const Eigen::Tensor<double, 2, 0, long>& x);
    static void printShape(const Eigen::Tensor<double, 4, 0, long>& x);
};


#endif //CEREBRUM_TENSOROPS_H

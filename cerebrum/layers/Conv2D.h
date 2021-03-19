//
// Created by Nik Willwerth on 3/17/21.
//

#ifndef CEREBRUM_CONV2D_H
#define CEREBRUM_CONV2D_H


#include "Layer.h"

class Conv2D: public Layer {
public:
    Conv2D(Layer *inputLayer, std::size_t numFilters, std::size_t filterSize, std::size_t stride = 1, const std::string& padding = "same", std::string layerName = "Conv");

    Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) override;
    Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) override;

    Eigen::Tensor<double, 2, 0, long> getWeights()  override;
    Eigen::Tensor<double, 2, 0, long> getBiases()   override;
    Eigen::Tensor<double, 2, 0, long> getDWeights() override;
    Eigen::Tensor<double, 2, 0, long> getDBiases()  override;

    void setWeights(Eigen::Tensor<double, 2, 0, long> newWeights) override;
    void setBiases(Eigen::Tensor<double, 2, 0, long>  newBiases)  override;

private:
    Eigen::Tensor<double, 2, 0, long> im2col(const Eigen::Tensor<double, 4, 0, long>& x);
    Eigen::Tensor<double, 4, 0, long> col2im(Eigen::Tensor<double, 2, 0, long> x);

    Eigen::Tensor<double, 4, 0, long> _weights;
    Eigen::Tensor<double, 1, 0, long> _biases;

    Eigen::Tensor<double, 4, 0, long> _dWeights;
    Eigen::Tensor<double, 1, 0, long> _dBiases;

    Eigen::Tensor<double, 2, 0, long> _cols;

    std::size_t _numFilters;
    std::size_t _filterSize;
    std::size_t _stride;
    std::size_t _pad;

    std::size_t _inputHeight;
    std::size_t _inputWidth;
    std::size_t _inputChannels;
};


#endif //CEREBRUM_CONV2D_H

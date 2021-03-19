//
// Created by Nik Willwerth on 3/16/21.
//

#ifndef CEREBRUM_DENSE_H
#define CEREBRUM_DENSE_H


#include "Layer.h"

class Dense: public Layer {
public:
    Dense(Layer *inputLayer, std::size_t numOutputs, std::string layerName = "Dense");

    Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) override;
    Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) override;

    Eigen::Tensor<double, 2, 0, long> getWeights()  override;
    Eigen::Tensor<double, 2, 0, long> getBiases()   override;
    Eigen::Tensor<double, 2, 0, long> getDWeights() override;
    Eigen::Tensor<double, 2, 0, long> getDBiases()  override;

    void setWeights(Eigen::Tensor<double, 2, 0, long> newWeights) override;
    void setBiases(Eigen::Tensor<double, 2, 0, long>  newBiases)  override;

private:
    Eigen::Tensor<double, 2, 0, long> _weights;
    Eigen::Tensor<double, 2, 0, long> _biases;

    Eigen::Tensor<double, 2, 0, long> _dWeights;
    Eigen::Tensor<double, 2, 0, long> _dBiases;

    std::size_t _numInputs;
    std::size_t _numOutputs;
};


#endif //CEREBRUM_DENSE_H

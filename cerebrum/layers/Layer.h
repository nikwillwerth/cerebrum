//
// Created by Nik Willwerth on 3/15/21.
//

#ifndef CEREBRUM___LAYER_H
#define CEREBRUM___LAYER_H

#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

class Layer {
public:
    Layer(Layer *inputLayer, std::string layerName);
    Layer(std::vector<Layer *> inputLayers, std::string layerName);

    virtual Eigen::Tensor<double, 4, 0, long> forward(Eigen::Tensor<double, 4, 0, long> x) { return Eigen::Tensor<double, 4, 0, long>(); };
    virtual Eigen::Tensor<double, 4, 0, long> backward(Eigen::Tensor<double, 4, 0, long> t) { return Eigen::Tensor<double, 4, 0, long>(); };

    virtual Eigen::Tensor<double, 2, 0, long> getWeights() { return Eigen::Tensor<double, 2, 0, long>(); };
    virtual Eigen::Tensor<double, 2, 0, long> getBiases()  { return Eigen::Tensor<double, 2, 0, long>(); };
    virtual Eigen::Tensor<double, 2, 0, long> getDWeights() { return Eigen::Tensor<double, 2, 0, long>(); };
    virtual Eigen::Tensor<double, 2, 0, long> getDBiases()  { return Eigen::Tensor<double, 2, 0, long>(); };

    virtual void setWeights(Eigen::Tensor<double, 2, 0, long> newWeights) {};
    virtual void setBiases(Eigen::Tensor<double, 2, 0, long>  newBiases) {};

    Eigen::Tensor<double, 4, 0, long> previousX;

    std::vector<Layer *> inputLayers;
    std::vector<Layer *> outputLayers;
    std::vector<std::size_t> outputShape;
    std::size_t batchSize;
    std::string name;

    bool hasWeights = true;
};


#endif //CEREBRUM___LAYER_H

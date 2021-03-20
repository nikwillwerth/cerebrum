//
// Created by Nik Willwerth on 3/17/21.
//

#ifndef CEREBRUM_MNIST_H
#define CEREBRUM_MNIST_H


#include <string>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

class Mnist {
public:
    Mnist();

    std::vector<Eigen::Tensor<double, 4, 0, long>> trainImages;
    std::vector<Eigen::Tensor<double, 4, 0, long>> valImages;
    std::vector<Eigen::Tensor<double, 4, 0, long>> trainLabels;
    std::vector<Eigen::Tensor<double, 4, 0, long>> valLabels;

private:
    std::vector<std::string> getDataFiles();
    static std::vector<Eigen::Tensor<double, 4, 0, long>> readImages(const std::string& fileName);
    static std::vector<Eigen::Tensor<double, 4, 0, long>> readLabels(const std::string& fileName);
    static std::vector<unsigned char> decompressFile(const std::string& fileName);

    std::string trainImagesUrl = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-images-idx3-ubyte.gz";
    std::string trainLabelsUrl = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-labels-idx1-ubyte.gz";
    std::string valImagesUrl   = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-images-idx3-ubyte.gz";
    std::string valLabelsUrl   = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-labels-idx1-ubyte.gz";
};


#endif //CEREBRUM_MNIST_H

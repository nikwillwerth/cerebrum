//
// Created by Nik Willwerth on 3/17/21.
//

#include <iostream>
#include "Conv2D.h"
#include "../utils/TensorOps.h"

Conv2D::Conv2D(Layer *inputLayer, std::size_t numFilters, std::size_t filterSize, std::size_t stride, const std::string& padding) : Layer(inputLayer) {
    this->filterSize = filterSize;
    this->stride     = stride;
    this->padding    = padding;

    std::size_t inputHeight   = this->inputLayer->outputShape[1];
    std::size_t inputWidth    = this->inputLayer->outputShape[2];
    std::size_t inputChannels = this->inputLayer->outputShape[3];

    this->weights = Eigen::Tensor<double, 4, 0, long>(long(this->filterSize), long(this->filterSize), long(inputChannels), long(numFilters));
    this->biases  = Eigen::Tensor<double, 1, 0, long>(long(numFilters));

    this->weights.setRandom<Eigen::internal::NormalRandomGenerator<double>>();
    this->biases.setRandom<Eigen::internal::NormalRandomGenerator<double>>();

    this->weights *= this->weights.constant(0.1);
    this->biases  *= this->biases.constant(0.1);

    if(padding == "same") {
        this->pad         = std::size_t((this->filterSize - 1) / 2);
        this->outputShape = {this->batchSize, inputHeight, inputWidth, numFilters};
    } else if(padding == "valid") {
        std::size_t outputWidth  = std::size_t((inputWidth  - this->filterSize) / this->stride) + 1;
        std::size_t outputHeight = std::size_t((inputHeight - this->filterSize) / this->stride) + 1;

        this->outputShape = {this->batchSize, outputHeight, outputWidth, numFilters};
        this->pad         = 0;
    } else {
        throw std::invalid_argument("Conv2D: Invalid padding value: " + padding);
    }
}

Eigen::Tensor<double, 4, 0, long> Conv2D::forward(Eigen::Tensor<double, 4, 0, long> x) {
    this->previousX = x;

//    Eigen::PaddingType paddingType;
//
//    if(this->padding == "same") {
//        paddingType = Eigen::PADDING_SAME;
//    } else {
//        paddingType = Eigen::PADDING_VALID;
//    }
//
//    Eigen::Tensor<double, 4, 0, long> patches = x.extract_image_patches(this->filterSize, this->filterSize, this->stride, this->stride, 1, 1, paddingType, 0);
//
//    std::cout << patches << std::endl;

//    std::cout << x.dimension(0) << " " << x.dimension(1) << " " << x.dimension(2) << " " << x.dimension(3) << std::endl;

    this->cols = this->im2col(x);

    std::size_t weightsDimTwo = this->weights.dimension(0) * this->weights.dimension(1) * this->weights.dimension(2);

    Eigen::array<long, 4> weightsShuffleIndices({3, 2, 0, 1});
    Eigen::Tensor<double, 4, 0, long> transposedWeights = this->weights.shuffle(weightsShuffleIndices);
    Eigen::Tensor<double, 2, 0, long> reshapedWeights   = TensorOps::reshape(transposedWeights, this->outputShape[3], weightsDimTwo);

    Eigen::Tensor<double, 2, 0, long> output         = TensorOps::dot(reshapedWeights, this->cols);
    Eigen::Tensor<double, 4, 0, long> reshapedOutput = TensorOps::reshape(output, this->outputShape[3], this->outputShape[1], this->outputShape[2], this->outputShape[0]);

    Eigen::array<long, 4> outputShuffleIndices({3, 1, 2, 0});
    Eigen::Tensor<double, 4, 0, long> finalOutput = reshapedOutput.shuffle(outputShuffleIndices);

    Eigen::Tensor<double, 4, 0, long> broadcastBiases = TensorOps::broadcast(this->biases, this->outputShape[0], this->outputShape[1], this->outputShape[2], 1);

    return finalOutput + broadcastBiases;
}

Eigen::Tensor<double, 4, 0, long> Conv2D::backward(Eigen::Tensor<double, 4, 0, long> t) {
    Eigen::array<long, 3> axisIndex({0, 1, 2});
    Eigen::Tensor<double, 1, 0, long> sum = t.sum(axisIndex);
//    TensorOps::printShape(t);
//    TensorOps::printShape(sum);
//    TensorOps::printShape(this->biases);
    this->dBiases = t.sum(axisIndex) / this->biases.constant(batchSize);

    Eigen::array<long, 4> tShuffleIndices({3, 1, 2, 0});
    Eigen::Tensor<double, 4, 0, long> transposedT = t.shuffle(tShuffleIndices);

    std::size_t tDimTwo = this->outputShape[0] * this->outputShape[1] * this->outputShape[2];
    Eigen::Tensor<double, 2, 0, long> reshapedT = TensorOps::reshape(transposedT, this->outputShape[3], tDimTwo);

    Eigen::array<long, 4> weightsShuffleIndices({3, 2, 0, 1});
    Eigen::Tensor<double, 4, 0, long> transposedWeights = this->weights.shuffle(weightsShuffleIndices);

    Eigen::Tensor<double, 2, 0, long> weightGradients = TensorOps::dot(reshapedT, TensorOps::transpose(this->cols));

    Eigen::Tensor<double, 4, 0, long> reshapedDWeights = TensorOps::reshape(weightGradients, transposedWeights.dimension(0), transposedWeights.dimension(1), transposedWeights.dimension(2), transposedWeights.dimension(3));

    Eigen::array<long, 4> dWeightsShuffleIndices({2, 3, 1, 0});
    this->dWeights = reshapedDWeights.shuffle(dWeightsShuffleIndices);

    std::size_t weightsDimTwo = this->weights.dimension(0) * this->weights.dimension(1) * this->weights.dimension(2);
    Eigen::Tensor<double, 2, 0, long> reshapedWeights = TensorOps::reshape(transposedWeights, this->outputShape[3], weightsDimTwo);

    Eigen::Tensor<double, 2, 0, long> outputCols = TensorOps::dot(TensorOps::transpose(reshapedWeights), reshapedT);

    Eigen::Tensor<double, 4, 0, long> output = this->col2im(outputCols);

    Eigen::array<long, 4> outputShuffleIndices({0, 2, 3, 1});

    this->weights = this->weights - (this->weights.constant(0.01) * this->dWeights);
    this->biases  = this->biases  - (this->biases.constant(0.01)  * this->dBiases);

    return output.shuffle(outputShuffleIndices);
}

Eigen::Tensor<double, 2, 0, long> Conv2D::im2col(Eigen::Tensor<double, 4, 0, long> x) {
    Eigen::array<long, 4> shuffleIndices({0, 3, 2, 1});
    Eigen::Tensor<double, 4, 0, long> nchwX = x.shuffle(shuffleIndices);

    std::size_t n = nchwX.dimension(0);
    std::size_t c = nchwX.dimension(1);
    std::size_t h = nchwX.dimension(2);
    std::size_t w = nchwX.dimension(3);

    auto hh = std::size_t(((h + (2 * this->pad) - this->filterSize) / this->stride) + 1);
    auto ww = std::size_t(((w + (2 * this->pad) - this->filterSize) / this->stride) + 1);

    Eigen::Tensor<double, 4, 0, long> paddedX;

    if(this->pad != 0) {
        Eigen::array<std::pair<std::size_t, std::size_t>, 4> paddings;
        paddings[0] = std::make_pair(0, 0);
        paddings[1] = std::make_pair(0, 0);
        paddings[2] = std::make_pair(this->pad, this->pad);
        paddings[3] = std::make_pair(this->pad, this->pad);

        paddedX = nchwX.pad(paddings);
    } else {
        paddedX = nchwX;
    }

    Eigen::Tensor<double, 2, 0, long> imCols(c * this->filterSize * this->filterSize, n * hh * ww);

    for(std::size_t cc = 0; cc < c; cc++) {
        for(std::size_t ii = 0; ii < this->filterSize; ii++) {
            for(std::size_t jj = 0; jj < this->filterSize; jj++) {
                std::size_t row = (cc * this->filterSize * this->filterSize) + (ii * this->filterSize) + jj;

                for(std::size_t yy = 0; yy < hh; yy++) {
                    for(std::size_t xx = 0; xx < ww; xx++) {
                        for(std::size_t i = 0; i < n; i++) {
                            std::size_t col = (yy * ww * n) + (xx * n) + 1;

                            imCols(row, col) = paddedX(i, cc, (this->stride * yy) + ii, (this->stride * xx) + jj);
                        }
                    }
                }
            }
        }
    }

    return imCols;
}

Eigen::Tensor<double, 4, 0, long> Conv2D::col2im(Eigen::Tensor<double, 2, 0, long> outputCols) {
    std::size_t n = this->previousX.dimension(0);
    std::size_t h = this->previousX.dimension(1);
    std::size_t w = this->previousX.dimension(2);
    std::size_t c = this->previousX.dimension(3);

    auto hh = std::size_t(((h + (2 * this->pad) - this->filterSize) / this->stride) + 1);
    auto ww = std::size_t(((w + (2 * this->pad) - this->filterSize) / this->stride) + 1);

    Eigen::Tensor<double, 4, 0, long> x(n, c, h + (2 * this->pad), w + (w * this->pad));
    x.setZero();

    for(std::size_t cc = 0; cc < c; cc++) {
        for(std::size_t ii = 0; ii < this->filterSize; ii++) {
            for(std::size_t jj = 0; jj < this->filterSize; jj++) {
                std::size_t row = (cc * this->filterSize * this->filterSize) + (ii * this->filterSize) + jj;

                for(std::size_t yy = 0; yy < hh; yy++) {
                    for(std::size_t xx = 0; xx < ww; xx++) {
                        for(std::size_t i = 0; i < n; i++) {
                            std::size_t col = (yy * ww * n) + (xx * n) + 1;

                            x(i, cc, (this->stride * yy) + ii, (this->stride * xx) + jj) += outputCols(row, col);
                        }
                    }
                }
            }
        }
    }

    // TODO unpad x

    return x;
}

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

    this->weights = TensorOps::getRandomTensor(long(this->filterSize), long(this->filterSize), long(inputChannels), long(numFilters));
    this->biases  = TensorOps::getRandomTensor(numFilters);

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

    // im2col
    this->cols = this->im2col(x);

    std::size_t weightsDimTwo = this->weights.dimension(0) * this->weights.dimension(1) * this->weights.dimension(2);

    // shuffle weights (3, 2, 0, 1)
    // reshape weights to be (numFilters, w[0]*w[1]*w[2])
    Eigen::Tensor<double, 4, 0, long> shuffledWeights = TensorOps::shuffle(this->weights, 3, 2, 0, 1);
    Eigen::Tensor<double, 2, 0, long> reshapedWeights = TensorOps::reshape(shuffledWeights, this->outputShape[3], weightsDimTwo);

    // w * cols
    // reshape output to be (numFilters, outputHeight, outputWidth, batchSize)
    Eigen::Tensor<double, 2, 0, long> output         = TensorOps::dot(reshapedWeights, this->cols);
    Eigen::Tensor<double, 4, 0, long> reshapedOutput = TensorOps::reshape(output, this->outputShape[3], this->outputShape[1], this->outputShape[2], this->outputShape[0]);

    // shuffle output (3, 1, 2, 0)
    Eigen::Tensor<double, 4, 0, long> finalOutput = TensorOps::shuffle(reshapedOutput, 3, 1, 2, 0);

    // broadcast biases to be (batchSize, outputHeight, outputWidth, numFilters)
    Eigen::Tensor<double, 4, 0, long> broadcastBiases = TensorOps::broadcast(this->biases, this->outputShape[0], this->outputShape[1], this->outputShape[2], 1);

    // output + biases
    return finalOutput + broadcastBiases;
}

Eigen::Tensor<double, 4, 0, long> Conv2D::backward(Eigen::Tensor<double, 4, 0, long> t) {
    // reshape t to be the same shape as
    Eigen::Tensor<double, 4, 0, long> reshapedT = TensorOps::reshape(t, this->outputShape[0], this->outputShape[1], this->outputShape[2], this->outputShape[3]);

    // t.sum(asxis=(0, 1, 2)) / batchSize
    this->dBiases = TensorOps::sum(reshapedT, 0, 1, 2) / this->biases.constant(batchSize);

    // shuffle t (3, 1, 2, 0)
    Eigen::Tensor<double, 4, 0, long> transposedT = TensorOps::shuffle(reshapedT, 3, 1, 2, 0);

    // reshape t to be (numFilters, batchSize * outputHeight * outputWidth)
    std::size_t tDimTwo = this->outputShape[0] * this->outputShape[1] * this->outputShape[2];
    Eigen::Tensor<double, 2, 0, long> reshapedTT = TensorOps::reshape(transposedT, this->outputShape[3], tDimTwo);

    // shuffle weights (3, 2, 0, 1)
    Eigen::Tensor<double, 4, 0, long> transposedWeights = TensorOps::shuffle(this->weights, 3, 2, 0, 1);

    // t * cols.t
    Eigen::Tensor<double, 2, 0, long> weightGradients = TensorOps::dot(reshapedTT, TensorOps::transpose(this->cols));

    // reshape weight gradients to be (numFilters, inputChannels, filterSize, filterSze)
    Eigen::Tensor<double, 4, 0, long> reshapedDWeights = TensorOps::reshape(weightGradients, transposedWeights.dimension(0), transposedWeights.dimension(1), transposedWeights.dimension(2), transposedWeights.dimension(3));

    // shuffle dWeights (2, 3, 1, 0)
    this->dWeights = TensorOps::shuffle(reshapedDWeights, 2, 3, 1, 0);

    // reshape weights to be (numFilters, w[0]*w[1]*w[2])
    std::size_t weightsDimTwo = this->weights.dimension(0) * this->weights.dimension(1) * this->weights.dimension(2);
    Eigen::Tensor<double, 2, 0, long> reshapedWeights = TensorOps::reshape(transposedWeights, this->outputShape[3], weightsDimTwo);

    // w.t * t
    Eigen::Tensor<double, 2, 0, long> outputCols = TensorOps::dot(TensorOps::transpose(reshapedWeights), reshapedTT);

    // col2im
    Eigen::Tensor<double, 4, 0, long> output = this->col2im(outputCols);

    this->weights = this->weights - (this->weights.constant(0.01) * this->dWeights);
    this->biases  = this->biases  - (this->biases.constant(0.01)  * this->dBiases);

    // shuffle output (0, 2, 3, 1)
    return TensorOps::shuffle(output, 0, 2, 3, 1);
}

Eigen::Tensor<double, 2, 0, long> Conv2D::im2col(const Eigen::Tensor<double, 4, 0, long>& x) {
    // shuffle x from nhwc to nchw
    Eigen::Tensor<double, 4, 0, long> nchwX = TensorOps::shuffle(x, 0, 3, 2, 1);

    std::size_t n = nchwX.dimension(0);
    std::size_t c = nchwX.dimension(1);
    std::size_t h = nchwX.dimension(2);
    std::size_t w = nchwX.dimension(3);

    // calculate output size
    auto hh = std::size_t(((h + (2 * this->pad) - this->filterSize) / this->stride) + 1);
    auto ww = std::size_t(((w + (2 * this->pad) - this->filterSize) / this->stride) + 1);

    // pad x if necessary
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

    // calculate output size
    auto hh = std::size_t(((h + (2 * this->pad) - this->filterSize) / this->stride) + 1);
    auto ww = std::size_t(((w + (2 * this->pad) - this->filterSize) / this->stride) + 1);

    Eigen::Tensor<double, 4, 0, long> x(n, c, h + (2 * this->pad), w + (2 * this->pad));
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

    // unpad x if necessary
    Eigen::Tensor<double, 4, 0, long> unpaddedX;

    if(this->pad != 0) {
        Eigen::array<long, 4> offsets = {0, 0, long(this->pad), long(this->pad)};
        Eigen::array<long, 4> extents = {x.dimension(0), x.dimension(1), x.dimension(2) - long(2 * this->pad), x.dimension(3) - long(2 * this->pad)};

        unpaddedX = x.slice(offsets, extents);
    } else {
        unpaddedX = x;
    }

    return unpaddedX;
}

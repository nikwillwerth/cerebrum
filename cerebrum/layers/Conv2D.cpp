//
// Created by Nik Willwerth on 3/17/21.
//

#include <iostream>
#include <utility>
#include "Conv2D.h"
#include "../utils/TensorOps.h"

Conv2D::Conv2D(Layer *inputLayer, std::size_t numFilters, std::size_t filterSize, std::size_t stride, const std::string& padding, std::string layerName) : Layer(inputLayer, std::move(layerName)) {
    this->_numFilters = numFilters;
    this->_filterSize = filterSize;
    this->_stride     = stride;

    this->_inputHeight   = inputLayer->outputShape[1];
    this->_inputWidth    = inputLayer->outputShape[2];
    this->_inputChannels = inputLayer->outputShape[3];

    this->_weights = TensorOps::getRandomTensor(this->_filterSize, this->_filterSize, this->_inputChannels, this->_numFilters);
    this->_biases  = TensorOps::getRandomTensor(this->_numFilters);

    if(padding == "same") {
        this->_pad         = std::size_t((this->_filterSize - 1) / 2);
        this->outputShape = {this->batchSize, _inputHeight, _inputWidth, this->_numFilters};
    } else if(padding == "valid") {
        std::size_t outputWidth  = std::size_t((_inputWidth - this->_filterSize) / this->_stride) + 1;
        std::size_t outputHeight = std::size_t((_inputHeight - this->_filterSize) / this->_stride) + 1;

        this->outputShape = {this->batchSize, outputHeight, outputWidth, this->_numFilters};
        this->_pad         = 0;
    } else {
        throw std::invalid_argument("Conv2D: Invalid padding value: " + padding);
    }
}

Eigen::Tensor<double, 4, 0, long> Conv2D::forward(Eigen::Tensor<double, 4, 0, long> x) {
    this->previousX = x;

    // im2col
    this->_cols = this->im2col(x);

    std::size_t weightsDimTwo = this->_weights.dimension(0) * this->_weights.dimension(1) * this->_weights.dimension(2);

    // shuffle _weights (3, 2, 0, 1)
    // reshape _weights to be (_numFilters, w[0]*w[1]*w[2])
    Eigen::Tensor<double, 4, 0, long> shuffledWeights = TensorOps::shuffle(this->_weights, 3, 2, 0, 1);
    Eigen::Tensor<double, 2, 0, long> reshapedWeights = TensorOps::reshape(shuffledWeights, this->outputShape[3], weightsDimTwo);

    // w * _cols
    // reshape output to be (_numFilters, outputHeight, outputWidth, batchSize)
    Eigen::Tensor<double, 2, 0, long> output         = TensorOps::dot(reshapedWeights, this->_cols);
    Eigen::Tensor<double, 4, 0, long> reshapedOutput = TensorOps::reshape(output, this->outputShape[3], this->outputShape[1], this->outputShape[2], this->outputShape[0]);

    // shuffle output (3, 1, 2, 0)
    Eigen::Tensor<double, 4, 0, long> finalOutput = TensorOps::shuffle(reshapedOutput, 3, 1, 2, 0);

    // broadcast _biases to be (batchSize, outputHeight, outputWidth, _numFilters)
    Eigen::Tensor<double, 4, 0, long> broadcastBiases = TensorOps::broadcast(this->_biases, this->outputShape[0], this->outputShape[1], this->outputShape[2], 1);

    // output + _biases
    return finalOutput + broadcastBiases;
}

Eigen::Tensor<double, 4, 0, long> Conv2D::backward(Eigen::Tensor<double, 4, 0, long> t) {
    // reshape t to be the same shape as
    Eigen::Tensor<double, 4, 0, long> reshapedT = TensorOps::reshape(t, this->outputShape[0], this->outputShape[1], this->outputShape[2], this->outputShape[3]);

    // t.sum(asxis=(0, 1, 2)) / batchSize
    this->_dBiases = TensorOps::sum(reshapedT, 0, 1, 2) / this->_biases.constant(batchSize);

    // shuffle t (3, 1, 2, 0)
    Eigen::Tensor<double, 4, 0, long> transposedT = TensorOps::shuffle(reshapedT, 3, 1, 2, 0);

    // reshape t to be (_numFilters, batchSize * outputHeight * outputWidth)
    std::size_t tDimTwo = this->outputShape[0] * this->outputShape[1] * this->outputShape[2];
    Eigen::Tensor<double, 2, 0, long> reshapedTT = TensorOps::reshape(transposedT, this->outputShape[3], tDimTwo);

    // shuffle _weights (3, 2, 0, 1)
    Eigen::Tensor<double, 4, 0, long> transposedWeights = TensorOps::shuffle(this->_weights, 3, 2, 0, 1);

    // t * _cols.t
    Eigen::Tensor<double, 2, 0, long> weightGradients = TensorOps::dot(reshapedTT, TensorOps::transpose(this->_cols));

    // reshape weight gradients to be (_numFilters, _inputChannels, _filterSize, filterSze)
    Eigen::Tensor<double, 4, 0, long> reshapedDWeights = TensorOps::reshape(weightGradients, transposedWeights.dimension(0), transposedWeights.dimension(1), transposedWeights.dimension(2), transposedWeights.dimension(3));

    // shuffle _dWeights (2, 3, 1, 0)
    this->_dWeights = TensorOps::shuffle(reshapedDWeights, 2, 3, 1, 0);

    // reshape _weights to be (_numFilters, w[0]*w[1]*w[2])
    std::size_t weightsDimTwo = this->_weights.dimension(0) * this->_weights.dimension(1) * this->_weights.dimension(2);
    Eigen::Tensor<double, 2, 0, long> reshapedWeights = TensorOps::reshape(transposedWeights, this->outputShape[3], weightsDimTwo);

    // w.t * t
    Eigen::Tensor<double, 2, 0, long> outputCols = TensorOps::dot(TensorOps::transpose(reshapedWeights), reshapedTT);

    // col2im
    Eigen::Tensor<double, 4, 0, long> output = this->col2im(outputCols);

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
    auto hh = std::size_t(((h + (2 * this->_pad) - this->_filterSize) / this->_stride) + 1);
    auto ww = std::size_t(((w + (2 * this->_pad) - this->_filterSize) / this->_stride) + 1);

    // _pad x if necessary
    Eigen::Tensor<double, 4, 0, long> paddedX;

    if(this->_pad != 0) {
        Eigen::array<std::pair<std::size_t, std::size_t>, 4> paddings;
        paddings[0] = std::make_pair(0, 0);
        paddings[1] = std::make_pair(0, 0);
        paddings[2] = std::make_pair(this->_pad, this->_pad);
        paddings[3] = std::make_pair(this->_pad, this->_pad);

        paddedX = nchwX.pad(paddings);
    } else {
        paddedX = nchwX;
    }

    Eigen::Tensor<double, 2, 0, long> imCols(c * this->_filterSize * this->_filterSize, n * hh * ww);
    imCols.setZero();

    for(std::size_t cc = 0; cc < c; cc++) {
        for(std::size_t ii = 0; ii < this->_filterSize; ii++) {
            for(std::size_t jj = 0; jj < this->_filterSize; jj++) {
                std::size_t row = (cc * this->_filterSize * this->_filterSize) + (ii * this->_filterSize) + jj;

                for(std::size_t yy = 0; yy < hh; yy++) {
                    for(std::size_t xx = 0; xx < ww; xx++) {
                        for(std::size_t i = 0; i < n; i++) {
                            std::size_t col = (yy * ww * n) + (xx * n) + 1;

                            imCols(row, col) = paddedX(i, cc, (this->_stride * yy) + ii, (this->_stride * xx) + jj);
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
    auto hh = std::size_t(((h + (2 * this->_pad) - this->_filterSize) / this->_stride) + 1);
    auto ww = std::size_t(((w + (2 * this->_pad) - this->_filterSize) / this->_stride) + 1);

    Eigen::Tensor<double, 4, 0, long> x(n, c, h + (2 * this->_pad), w + (2 * this->_pad));
    x.setZero();

    for(std::size_t cc = 0; cc < c; cc++) {
        for(std::size_t ii = 0; ii < this->_filterSize; ii++) {
            for(std::size_t jj = 0; jj < this->_filterSize; jj++) {
                std::size_t row = (cc * this->_filterSize * this->_filterSize) + (ii * this->_filterSize) + jj;

                for(std::size_t yy = 0; yy < hh; yy++) {
                    for(std::size_t xx = 0; xx < ww; xx++) {
                        for(std::size_t i = 0; i < n; i++) {
                            std::size_t col = (yy * ww * n) + (xx * n) + 1;

                            x(i, cc, (this->_stride * yy) + ii, (this->_stride * xx) + jj) += outputCols(row, col);
                        }
                    }
                }
            }
        }
    }

    // unpad x if necessary
    Eigen::Tensor<double, 4, 0, long> unpaddedX;

    if(this->_pad != 0) {
        Eigen::array<long, 4> offsets = {0, 0, long(this->_pad), long(this->_pad)};
        Eigen::array<long, 4> extents = {x.dimension(0), x.dimension(1), x.dimension(2) - long(2 * this->_pad), x.dimension(3) - long(2 * this->_pad)};

        unpaddedX = x.slice(offsets, extents);
    } else {
        unpaddedX = x;
    }

    return unpaddedX;
}

Eigen::Tensor<double, 2, 0, long> Conv2D::getWeights() {
    std::size_t weightsDimTwo = this->_weights.dimension(1) * this->_weights.dimension(2) * this->_weights.dimension(3);

    return TensorOps::reshape(this->_weights, this->_weights.dimension(0), weightsDimTwo);
}

Eigen::Tensor<double, 2, 0, long> Conv2D::getBiases() {
    return TensorOps::reshape(this->_biases, this->_biases.dimension(0), 1);
}

Eigen::Tensor<double, 2, 0, long> Conv2D::getDWeights() {
    std::size_t dWeightsDimTwo = this->_dWeights.dimension(1) * this->_dWeights.dimension(2) * this->_dWeights.dimension(3);

    return TensorOps::reshape(this->_dWeights, this->_dWeights.dimension(0), dWeightsDimTwo);
}

Eigen::Tensor<double, 2, 0, long> Conv2D::getDBiases() {
    return TensorOps::reshape(this->_dBiases, this->_biases.dimension(0), 1);
}

void Conv2D::setWeights(Eigen::Tensor<double, 2, 0, long> newWeights) {
    this->_weights = TensorOps::reshape(newWeights, this->_filterSize, this->_filterSize, this->_inputChannels, this->_numFilters);
}

void Conv2D::setBiases(Eigen::Tensor<double, 2, 0, long> newBiases) {
    this->_biases = TensorOps::reshape(newBiases, this->_numFilters);
}

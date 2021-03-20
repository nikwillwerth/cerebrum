//
// Created by Nik Willwerth on 3/17/21.
//

#include <iostream>
#include <fstream>
#include <gzip/decompress.hpp>
#include "Mnist.h"
#include "../utils/NetIO.h"

Mnist::Mnist() {
    std::vector<std::string> dataFiles = this->getDataFiles();

    this->trainImages = Mnist::readImages(dataFiles[0]); // train images
    this->valImages   = Mnist::readImages(dataFiles[2]); // val images
    this->trainLabels = Mnist::readLabels(dataFiles[1]); // train labels
    this->valLabels   = Mnist::readLabels(dataFiles[3]); // val labels
}

std::vector<std::string> Mnist::getDataFiles() {
    std::vector<std::string> urls = {this->trainImagesUrl, this->trainLabelsUrl, this->valImagesUrl, this->valLabelsUrl};
    std::vector<std::string> dataFiles;

    for(const std::string& url : urls) {
        std::string fileName = url.substr(url.rfind('/') + 1);

        std::ifstream file(fileName);

        if(!file.good()) {
            NetIO::downloadFile(url, fileName);
        }

        file.close();

        dataFiles.emplace_back(fileName);
    }

    return dataFiles;
}

std::size_t charsToInt(std::vector<unsigned char> data, std::size_t offset)
{
    return (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | (data[offset + 3]);
}

std::vector<Eigen::Tensor<double, 4, 0, long>> Mnist::readImages(const std::string& fileName) {
    std::vector<unsigned char> data = Mnist::decompressFile(fileName);

    long numImages = charsToInt(data, 4);
    long numRows   = charsToInt(data, 8);
    long numCols   = charsToInt(data, 12);

    std::vector<Eigen::Tensor<double, 4, 0, long>> images;
    images.reserve(numImages * numRows * numCols);

    std::size_t imageSize = numRows * numCols;

    for(std::size_t offset = 16; offset < data.size(); offset += imageSize)
    {
        Eigen::Tensor<double, 4, 0, long> image(1, numCols, numRows, 1);
        double *imageData = image.data();

        for(std::size_t index = 0; index < imageSize; index++)
        {
            std::size_t thisIndex = (index % numRows) * numRows + (index / numRows);

            imageData[index] = static_cast<double>(data[offset + thisIndex]);
        }

        images.emplace_back(image);
    }

    return images;
}

std::vector<Eigen::Tensor<double, 4, 0, long>> Mnist::readLabels(const std::string &fileName) {
    std::vector<unsigned char> data = Mnist::decompressFile(fileName);

    long numLabels = charsToInt(data, 4);

    std::vector<Eigen::Tensor<double, 4, 0, long>> labels;
    labels.reserve(numLabels);

    for(std::size_t offset = 8; offset < data.size(); offset++)
    {
        long classIndex = static_cast<long>(data[offset]);

        Eigen::Tensor<double, 4, 0, long> label(1, 1, 1, 10);
        label.setZero();
        label(0, 0, 0, classIndex) = 1;

        labels.emplace_back(label);
    }

    return labels;
}

std::vector<unsigned char> Mnist::decompressFile(const std::string& fileName) {
    std::ifstream fileStream(fileName, std::ios_base::in | std::ios_base::binary);
    std::string fileString((std::istreambuf_iterator<char>(fileStream.rdbuf())),
                           std::istreambuf_iterator<char>());
    fileStream.close();

    gzip::Decompressor decompressor;
    std::vector<unsigned char> output;
    decompressor.decompress(output, fileString.data(), fileString.size());

    return output;
}

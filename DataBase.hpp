#ifndef MNIST_DATABASE_HPP
#define MNIST_DATABASE_HPP

#pragma once

#include ".../MNIST/Matrix.cpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <string>

class MNISTDataBase
{
    std::string pathTrainImages, pathTrainLabels, pathTestImages, pathTestLabels;
    std::vector<Matrix> testImages, trainImages;
    std::vector<int> testLabels, trainLabels;

public:

    MNISTDataBase();
    ~MNISTDataBase();
    static int intFromBytes(int i);
    void loadTestImages();
    void loadTrainImages();
    void loadTestLabels();
    void loadTrainLabels();
    void printImageAt(size_t index);
    Matrix matrixOfImageTrain(size_t index);
    int labelOfImageTrain(size_t index);
    Matrix matrixOfImageTest(size_t index);
    int labelOfImageTest(size_t index);
};



#endif //MNIST_DATABASE_HPP

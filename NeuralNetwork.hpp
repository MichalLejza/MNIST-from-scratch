#ifndef MNIST_NEURALNETWORK_HPP
#define MNIST_NEURALNETWORK_HPP

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "MNIST/DataBase.cpp"
#include "MNIST/Matrix.cpp"

class NeuralNetwork
{
public:
    size_t inputSize, hiddenSize, outputSize;
    Matrix hiddenMatrix, outputMatrix;
    double learningRate;
    int epochs;

    NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize, double learningRate, int epochs,Matrix hiddenMatrix, Matrix outputMatrix);
    ~NeuralNetwork();
    void trainNeuralNetworkOneImage(const Matrix &input, const Matrix &output);
    void trainNeuralNetwork(MNISTDataBase dataBase);
    Matrix predictImage(const Matrix &input);
    double calculatePredictions(MNISTDataBase dataBase, int howMany);
    void testImage(MNISTDataBase dataBase, size_t index);
    void test(Matrix &image);
};

NeuralNetwork loadNetwork()
{
    Matrix hidden(100, 784);
    Matrix output(10,100);
    NeuralNetwork network(784,300,10,0.1,10,hidden,output);
    network.hiddenMatrix.loadMatrix("HiddenMatrix");
    network.outputMatrix.loadMatrix("OutputMatrix");
    return network;
}

#endif //MNIST_NEURALNETWORK_HPP

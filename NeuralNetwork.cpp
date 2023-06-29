#pragma once

#include "MNIST/NeuralNetwork.hpp"


NeuralNetwork::NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize, double learningRate, int epochs, Matrix hiddenMatrix, Matrix outputMatrix) : hiddenMatrix(hiddenMatrix),
                                                                                                                                                                  outputMatrix(outputMatrix)
{
    this -> inputSize = inputSize;
    this -> hiddenSize = hiddenSize;
    this -> outputSize = outputSize;
    this -> learningRate = learningRate;
    this -> epochs = epochs;
    this -> hiddenMatrix.randomizeMatrix(hiddenSize);
    this -> outputMatrix.randomizeMatrix(outputSize);
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::trainNeuralNetworkOneImage(const Matrix &input, const Matrix &output)
{
    // input: 784 x 1
    // output 10 x 1
    // hiddenMatrix 300 x 784
    // outputMatrix 10 X 300
    Matrix hiddenInputs = multiply(hiddenMatrix, input); // 300 x 1
    Matrix hiddenOutputs = applyFunction(hiddenInputs, sigmoid); // 300 x 1
    Matrix finalInputs = multiply(outputMatrix, hiddenOutputs); // 10 x 1
    Matrix finalOutputs = applyFunction(finalInputs, sigmoid); // 10 x 1

    Matrix outputErrors = sub(output, finalOutputs); // 10 x 1
    Matrix transposedMat = transpose(outputMatrix); // 1 x 10
    Matrix hiddenErrors = multiply(transposedMat, outputErrors); //1 x 1

    Matrix sigmoidPrimed = sigmoidPrime(finalOutputs); // 10 x 1
    Matrix multipliedMat = dotProduct(outputErrors, sigmoidPrimed); // 10 x 1
    transposedMat = transpose(hiddenOutputs); // 1 x 300
    Matrix dotMat = multiply(multipliedMat, transposedMat); //10 x 300
    Matrix scaledMat = scale(dotMat, learningRate); // 10 x 300
    outputMatrix.addMatrix(scaledMat);

    sigmoidPrimed = sigmoidPrime(hiddenOutputs); // 300 x 1
    multipliedMat = dotProduct(hiddenErrors, sigmoidPrimed); // 300 x 1
    transposedMat = transpose(input); // 1 x 784
    dotMat = multiply(multipliedMat, transposedMat); // 300 x 784
    scaledMat = scale(dotMat, learningRate); // 300 x 784
    hiddenMatrix.addMatrix(scaledMat);
}

void NeuralNetwork::trainNeuralNetwork(MNISTDataBase dataBase)
{
    std::cout << ">Training Neural Network " << std::endl << ">number of epochs: " << epochs << std::endl;
    for(int i = 0; i < epochs; i++)
    {
        std::cout << ">epoch nr. " << i + 1 << std::endl;
        for(int j = 0; j < 15000; j++)
        {
            Matrix mat(dataBase.matrixOfImageTrain(j));
            mat.flattenMatrixToColumn();
            Matrix output(10,1);
            output.matrix[dataBase.labelOfImageTrain(j)][0] = 1;
            trainNeuralNetworkOneImage(mat, output);
        }
    }
    std::cout << ">Done training" << std::endl;
}

Matrix NeuralNetwork::predictImage(const Matrix &input)
{
    // input 784 x 1
    // hiddenMatrix 300 x 784
    // output Matrix 10 x 300
    Matrix hiddenInputs = multiply(hiddenMatrix, input); // 300 x 1
    Matrix hiddenOutputs = applyFunction(hiddenInputs, sigmoid); // 300 x 1
    Matrix finalInputs = multiply(outputMatrix, hiddenOutputs); // 10 x 1
    Matrix finalOutputs = applyFunction(finalInputs, sigmoid); // 10 x 1
    Matrix result = sigmoidPrime(finalOutputs);
    return finalOutputs;
}

double NeuralNetwork::calculatePredictions(MNISTDataBase dataBase, int howMany)
{
    int correct = 0;
    for(int i = 0; i < howMany; i++)
    {
        Matrix image = dataBase.matrixOfImageTest(i);
        image.flattenMatrixToColumn();
        Matrix predictions = predictImage(image);
        if(predictions.argMax() == dataBase.labelOfImageTest(i))
            correct++;

    }
    return 1.0 * correct / howMany;
}

void NeuralNetwork::testImage(MNISTDataBase dataBase, size_t index)
{
    Matrix image = dataBase.matrixOfImageTest(index);
    image.printMatrix();
    std::cout << "Which number the image represents: " << dataBase.labelOfImageTest(index) << std::endl;
    image.flattenMatrixToColumn();
    Matrix predictions = predictImage(image);
    predictions.printMatrix();
    std::cout << "Neural network prediction: " << predictions.argMax() << std::endl;
}


void NeuralNetwork::test(Matrix &image)
{
    image.printMatrix();
    image.flattenMatrixToColumn();
    Matrix predictions = predictImage(image);
    predictions.printMatrix();
    std::cout << "Neural network prediction: " << predictions.argMax() << std::endl;
}

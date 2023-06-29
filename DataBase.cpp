#pragma once

#include ".../MNIST/DataBase.hpp"

MNISTDataBase::MNISTDataBase()
{
    this -> pathTestImages = ".../t10k-images.idx3-ubyte";
    this -> pathTestLabels = ".../t10k-labels.idx1-ubyte";
    this -> pathTrainImages = ".../train-images.idx3-ubyte";
    this -> pathTrainLabels = ".../train-labels.idx1-ubyte";
    this -> testImages = std::vector<Matrix>();
    this -> trainImages = std::vector<Matrix>();
    this -> trainLabels = std::vector<int>();
    this -> testLabels = std::vector<int>();
}

MNISTDataBase::~MNISTDataBase() = default;

int MNISTDataBase::intFromBytes(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void MNISTDataBase::loadTestImages()
{
    std::ifstream testFile(pathTestImages);
    if(!testFile.is_open())
    {
        std::cout << ">Error at function: loadImages: couldn't load test Images";
        exit(1);
    }
    int magicNumber = 0;
    testFile.read((char*)&magicNumber,sizeof(magicNumber));
    magicNumber = intFromBytes(magicNumber);

    int numberImages = 0;
    testFile.read((char*)&numberImages,sizeof(numberImages));
    numberImages = intFromBytes(numberImages);

    int rows = 0;
    testFile.read((char*)&rows,sizeof(rows));
    rows = intFromBytes(rows);

    int cols = 0;
    testFile.read((char*)&cols,sizeof(cols));
    cols = intFromBytes(cols);

    for (int images = 0; images < numberImages; images++)
    {
        Matrix m(rows, cols);
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                unsigned char number = 0;
                testFile.read((char*)&number, sizeof(number));
                double v = (double)number / 255;
                m.matrix[i][j] = v;
            }
        }
        testImages.push_back(m);
    }
    testFile.close();
}

void MNISTDataBase::loadTrainImages()
{
    std::ifstream trainFile(pathTrainImages);
    if(!trainFile.is_open())
    {
        std::cout << ">Error at function: loadImages: couldn't load train Images";
        exit(1);
    }
    int magicNumber = 0;
    trainFile.read((char*)&magicNumber,sizeof(magicNumber));
    magicNumber = intFromBytes(magicNumber);

    int numberImages = 0;
    trainFile.read((char*)&numberImages,sizeof(numberImages));
    numberImages = intFromBytes(numberImages);

    int rows = 0;
    trainFile.read((char*)&rows,sizeof(rows));
    rows = intFromBytes(rows);

    int cols = 0;
    trainFile.read((char*)&cols,sizeof(cols));
    cols = intFromBytes(cols);

    for (int images = 0; images < numberImages; images++)
    {
        Matrix m(rows, cols);
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                unsigned char number = 0;
                trainFile.read((char*)&number, sizeof(number));
                double v = (double)number / 255;
                m.matrix[i][j] = v;
            }
        }
        trainImages.push_back(m);
    }
    trainFile.close();
}

void MNISTDataBase::loadTestLabels()
{
    std::ifstream testFile(pathTestLabels);
    if(!testFile.is_open())
    {
        std::cout << ">Error at function: loadImages: couldn't load test Labels";
        exit(1);
    }

    int magicNumber = 0;
    testFile.read((char*)&magicNumber,sizeof(magicNumber));
    magicNumber = intFromBytes(magicNumber);

    int numberLabels = 0;
    testFile.read((char*)&numberLabels,sizeof(numberLabels));
    numberLabels = intFromBytes(numberLabels);

    for (int i = 0; i < numberLabels; i++)
    {
        unsigned char number = 0;
        testFile.read((char*)&number, sizeof(number));
        testLabels.push_back(number);
    }
    testFile.close();
}

void MNISTDataBase::loadTrainLabels()
{
    std::ifstream trainFile(pathTrainLabels);
    if(!trainFile.is_open())
    {
        std::cout << ">Error at function: loadImages: couldn't load train Labels";
        exit(1);
    }

    int magicNumber = 0;
    trainFile.read((char*)&magicNumber,sizeof(magicNumber));
    magicNumber = intFromBytes(magicNumber);

    int numberLabels = 0;
    trainFile.read((char*)&numberLabels,sizeof(numberLabels));
    numberLabels = intFromBytes(numberLabels);

    for (int i = 0; i < numberLabels; i++)
    {
        unsigned char number = 0;
        trainFile.read((char*)&number, sizeof(number));
        trainLabels.push_back(number);
    }
    trainFile.close();
}

void MNISTDataBase::printImageAt(size_t index)
{
    std::cout << "Image represents number: " << trainLabels.at(index) << std::endl;
    for(size_t i = 0; i < trainImages.at(index).x; i++)
    {
        for(size_t j = 0; j < trainImages.at(index).y; j++)
        {
            if(trainImages.at(index).matrix[i][j] > 0)
                printf("\x1B[35m");
            std::cout << std::fixed << std::setprecision(2) << trainImages.at(index).matrix[i][j] << " ";
            printf("\033[0m");
        }
        std::cout << std::endl;
    }
}

Matrix MNISTDataBase::matrixOfImageTrain(size_t index)
{
    return trainImages.at(index);
}

int MNISTDataBase::labelOfImageTrain(size_t index)
{
    return trainLabels.at(index);
}

Matrix MNISTDataBase::matrixOfImageTest(size_t index)
{
    return testImages.at(index);
}

int MNISTDataBase::labelOfImageTest(size_t index)
{
    return testLabels.at(index);
}

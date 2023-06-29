#ifndef MNIST_MATRIX_HPP
#define MNIST_MATRIX_HPP

#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iomanip>
#include <fstream>
#include <random>

class Matrix
{
public:
    std::vector<std::vector<double>> matrix;
    size_t x,y;

    Matrix(size_t x, size_t y);
    Matrix(const Matrix &copy);
    Matrix(size_t x, size_t y, double v);
    ~Matrix();
    [[nodiscard]] bool checkSameDimensions(const Matrix &m) const;
    [[nodiscard]] bool checkDimensions(const Matrix &m) const;
    void printMatrix();
    void addMatrix(const Matrix &m);
    void subtractMatrix(const Matrix &m);
    void multiplyMatrix(const Matrix &m);
    void dotProductMatrix(const Matrix &m);
    void scaleMatrix(double scalar);
    void transposeMatrix();
    void addScalarMatrix(double scalar);
    void saveMatrix(const std::string& path);
    void loadMatrix(const std::string& path);
    void flattenMatrixToRow();
    void flattenMatrixToColumn();
    void randomizeMatrix(size_t n);
    void applyFunctionMatrix(double (*func)(double));
    void softmaxMatrix();
    double argMax();
};

Matrix add(const Matrix &m1, const Matrix &m2);
Matrix sub(const Matrix &m1, const Matrix &m2);
Matrix multiply(const Matrix &m1, const Matrix &m2);
Matrix dotProduct(const Matrix &m1, const Matrix &m2);
Matrix applyFunction(const Matrix &m, double(*func)(double));
Matrix scale(const Matrix &m, double scalar);
Matrix transpose(const Matrix &m);


double sigmoid(double input);
Matrix sigmoidPrime(const Matrix &m);
Matrix softmax(const Matrix &m);

#endif //MNIST_MATRIX_HPP

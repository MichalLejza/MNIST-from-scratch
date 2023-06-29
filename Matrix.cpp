#pragma once

#include "MNIST/Matrix.hpp"

Matrix::Matrix(size_t x, size_t y)
{
    this -> x = x;
    this -> y = y;
    this -> matrix = std::vector<std::vector<double>>(x, std::vector<double>(y));
}

Matrix::Matrix(const Matrix &copy)
{
    this -> x = copy.x;
    this -> y = copy.y;
    this -> matrix = std::vector<std::vector<double>>(x, std::vector<double>(y));
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] = copy.matrix[i][j];
}

Matrix::Matrix(size_t x, size_t y, double v)
{
    this -> x = x;
    this -> y = y;
    this -> matrix = std::vector<std::vector<double>>(x, std::vector<double>(y));
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] = v;
}

Matrix::~Matrix() = default;


bool Matrix::checkSameDimensions(const Matrix &m) const
{
    return x == m.x && y == m.y;
}

bool Matrix::checkDimensions(const Matrix &m) const
{
    return y == m.x;
}

void Matrix::printMatrix()
{
    std::cout << x << " " << y << std::endl;
    for(size_t i = 0; i < x; i++)
    {
        for(size_t j = 0; j < y; j++)
        {
            if(matrix[i][j] > 0)
                printf("\x1B[35m");
            std::cout << std::fixed << std::setprecision(2) << matrix[i][j] <<  "  ";
            printf("\033[0m");
        }
        std::cout << std::endl;
    }
}

void Matrix::addMatrix(const Matrix &m)
{
    if(!checkSameDimensions(m))
    {
        std::cout << ">Error addMatrix function: Dimensions mismatch\n";
        std::cout << "m1 dimensions: " << x << " " << y<< std::endl;
        std::cout << "m2 dimensions: " << m.x << " " << m.y<< std::endl;
        exit(1);
    }
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] += m.matrix[i][j];
}

void Matrix::subtractMatrix(const Matrix &m)
{
    if(!checkSameDimensions(m))
    {
        std::cout << ">Error subtractMatrix function: Dimensions mismatch\n";
        std::cout << "m1 dimensions: " << x << " " << y<< std::endl;
        std::cout << "m2 dimensions: " << m.x << " " << m.y<< std::endl;
        exit(1);
    }
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] -= m.matrix[i][j];
}

void Matrix::multiplyMatrix(const Matrix &m)
{
    if(!checkDimensions(m))
    {
        std::cout << ">Error multiplyMatrix function: Dimensions mismatch\n";
        std::cout << "m1 dimensions: " << x << " " << y<< std::endl;
        std::cout << "m2 dimensions: " << m.x << " " << m.y<< std::endl;
        exit(1);
    }
    auto copy = std::vector<std::vector<double>>(x, std::vector<double>(m.y));
    for(size_t i = 0; i < x; i++)
    {
        for(size_t j = 0; j < m.y; j++)
        {
            double sum = 0;
            for(size_t k = 0; k < m.x; k++)
                sum += matrix[i][k] * m.matrix[k][j];
            copy[i][j] = sum;
        }
    }
    this -> y = m.y;
    this -> matrix = copy;
}

void Matrix::dotProductMatrix(const Matrix &m)
{
    if(!checkSameDimensions(m))
    {
        std::cout << ">Error scalarProductMatrix function: Dimensions mismatch\n";
        std::cout << "m1 dimensions: " << x << " " << y<< std::endl;
        std::cout << "m2 dimensions: " << m.x << " " << m.y<< std::endl;
        exit(1);
    }
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] *= m.matrix[i][j];
}

void Matrix::scaleMatrix(double scalar)
{
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] *= scalar;
}

void Matrix::transposeMatrix()
{
    auto copy = std::vector<std::vector<double>>(y, std::vector<double>(x));
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            copy[j][i] = matrix[i][j];
    matrix = copy;
    size_t c = x;
    x = y;
    y = c;
    copy.clear();
}

void Matrix::addScalarMatrix(double scalar)
{
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] += scalar;
}

void Matrix::saveMatrix(const std::string& path)
{
    std::string directory = ".../MNIST/DaneWielo";
    std::ofstream file(directory + "/" + path + ".txt");
    if(!file.is_open())
    {
        std::cout << ">Error in function saveMatrix: file was not opened";
        exit(1);
    }
    file << x << "\n" << y << "\n";
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            file << matrix[i][j] << "\n";
    file.close();
}

void Matrix::loadMatrix(const std::string& path)
{
    std::string directory = ".../MNIST/DaneWielo";
    std::ifstream file(directory + "/" + path + ".txt");
    if(!file.is_open())
    {
        std::cout << ">Error in function loadMatrix: file was not opened";
        exit(1);
    }
    size_t tempX, tempY;
    file >> tempX >> tempY;
    std::vector<std::vector<double>> temp = std::vector<std::vector<double>>(tempX, std::vector<double>(tempY));
    for(size_t i = 0; i < tempX; i++)
        for(size_t j = 0; j < tempY; j++)
            file >> temp[i][j];
    x = tempX;
    y = tempY;
    matrix = temp;
    file.close();
}

void Matrix::flattenMatrixToRow()
{
    Matrix m(1, x * y);
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            m.matrix[0][i * y + j] = matrix[i][j];
    matrix = m.matrix;
    x = m.x;
    y = m.y;
}

void Matrix::flattenMatrixToColumn()
{
    Matrix m(x * y, 1 );
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            m.matrix[i * y + j][0] = matrix[i][j];
    matrix = m.matrix;
    x = m.x;
    y = m.y;
}

void Matrix::randomizeMatrix(size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0 / sqrt(n), 1.0 / sqrt(n));
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] = dist(gen);
}

void Matrix::applyFunctionMatrix(double (*func)(double))
{
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] = func(matrix[i][j]);
}

void Matrix::softmaxMatrix()
{
    double total = 0;
    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            total += exp(matrix[i][j]);

    for(size_t i = 0; i < x; i++)
        for(size_t j = 0; j < y; j++)
            matrix[i][j] = exp(matrix[i][j]) / total;
}

double Matrix::argMax()
{
    double result = 0;
    int index = 0;
    for(int i = 0; i < x; i++)
        if(matrix[i][0] > result)
        {
            result = matrix[i][0];
            index = i;
        }
    return index;
}

Matrix add(const Matrix &m1, const Matrix &m2)
{
    if(!m1.checkSameDimensions(m2))
    {
        std::cout << ">Error in add function: mismatched Dimensions";
        std::cout << "m1 dimensions: " << m1.x << " " << m1.y<< std::endl;
        std::cout << "m2 dimensions: " << m2.x << " " << m2.y<< std::endl;
        exit(1);
    }
    Matrix m(m1.x, m1.y);
    for(size_t i = 0; i < m1.x; i++)
        for(size_t j = 0; j < m1.y; j++)
            m.matrix[i][j] = m1.matrix[i][j] + m2.matrix[i][j];
    return m;
}

Matrix sub(const Matrix &m1, const Matrix &m2)
{
    if(!m1.checkSameDimensions(m2))
    {
        std::cout << ">Error in sub function: mismatched Dimensions";
        std::cout << "m1 dimensions: " << m1.x << " " << m1.y<< std::endl;
        std::cout << "m2 dimensions: " << m2.x << " " << m2.y<< std::endl;
        exit(1);
    }
    Matrix m(m1.x, m1.y);
    for(size_t i = 0; i < m1.x; i++)
        for(size_t j = 0; j < m1.y; j++)
            m.matrix[i][j] = m1.matrix[i][j] - m2.matrix[i][j];
    return m;
}

Matrix multiply(const Matrix &m1, const Matrix &m2)
{
    if(!m1.checkDimensions(m2))
    {
        std::cout << ">Error in multiply function: mismatched Dimensions: " << std::endl;
        std::cout << "m1 dimensions: " << m1.x << " " << m1.y<< std::endl;
        std::cout << "m2 dimensions: " << m2.x << " " << m2.y<< std::endl;
        exit(1);
    }
    Matrix m(m1.x, m2.y);
    for(size_t i = 0; i < m1.x; i++)
    {
        for(size_t j = 0; j < m2.y; j++)
        {
            double sum = 0;
            for(size_t k = 0; k < m2.x; k++)
                sum += m1.matrix[i][k] * m2.matrix[k][j];
            m.matrix[i][j] = sum;
        }
    }
    return m;
}

Matrix dotProduct(const Matrix &m1, const Matrix &m2)
{
    if(!m1.checkSameDimensions(m2))
    {
        std::cout << ">Error in dotProduct function: mismatched Dimensions";
        exit(1);
    }
    Matrix m(m1.x, m1.y);
    for(size_t i = 0; i < m1.x; i++)
        for(size_t j = 0; j < m1.y; j++)
            m.matrix[i][j] = m1.matrix[i][j] * m2.matrix[i][j];
    return m;
}

Matrix applyFunction(const Matrix &m, double(*func)(double))
{
    Matrix mat(m);
    for(size_t i = 0; i < mat.x; i++)
        for(size_t j = 0; j < mat.y; j++)
            mat.matrix[i][j] = func(mat.matrix[i][j]);
    return mat;
}

Matrix scale(const Matrix &m, double scalar)
{
    Matrix mat(m);
    for(size_t i = 0; i < mat.x; i++)
        for(size_t j = 0; j < mat.y; j++)
            mat.matrix[i][j] = mat.matrix[i][j] * scalar;
    return mat;
}

Matrix transpose(const Matrix &m)
{
    Matrix mat(m.y, m.x);

    for(size_t i = 0; i < m.x; i++)
        for(size_t j = 0; j < m.y; j++)
            mat.matrix[j][i] = m.matrix[i][j];
    return mat;
}

double sigmoid(double input)
{
    return 1.0 / (1 + exp(-1.0 * input));
}

Matrix sigmoidPrime(const Matrix &m)
{
    Matrix ones(m.x, m.y, 1);
    auto subtracted = sub(ones, m);
    auto multiplied= dotProduct(m, subtracted);
    return multiplied;
}

Matrix softmax(const Matrix &m)
{
    double total = 0;
    for(size_t i = 0; i < m.x; i++)
        for(size_t j = 0; j < m.y; j++)
            total += (double)exp(m.matrix[i][j]);

    Matrix mat(m.x, m.y);

    for(size_t i = 0; i < mat.x; i++)
        for(size_t j = 0; j < mat.y; j++)
            mat.matrix[i][j] = (double)exp(mat.matrix[i][j] / total);
    return mat;
}



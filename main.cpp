#include "MNIST/Matrix.cpp"
#include "MNIST/DataBase.cpp"
#include "MNIST/NeuralNetwork.cpp"
#include "opencv2/opencv.hpp"

Matrix convertedImage(const std::string& fileName)
{
    std::string path = "..." + fileName + ".jpg";
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) { std::cerr << "Failed to open or read the image file" << std::endl; exit(1);}
    cv::resize(image, image, cv::Size(28, 28));
    double grayscaleMatrix[28][28];
    for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j)
            grayscaleMatrix[i][j] = (abs(255 - static_cast<double>(image.at<uchar>(i, j))));
    std::ofstream file("MNIST/grayscale_matrix.txt");
    if (!file.is_open())
        std::cout << "Failed to open file";
    file << 28 << "\n";
    file << 28 << "\n";
    file.close();

    Matrix m(28,28);
    m.loadMatrix("grayscale_matrix");
    return m;
}

auto main() -> int
{
    MNISTDataBase dataBase;
    dataBase.loadTestImages();
    dataBase.loadTestLabels();
    dataBase.loadTrainImages();
    dataBase.loadTrainLabels();

    //Matrix mat(10,784);
    //auto *perceptron = new Perceptron(784,10,0.1,mat);
    //perceptron ->trainNeuralNetwork(dataBase, 10);
    //std::cout << perceptron ->calculatePredictions(dataBase, 1000);
    //perceptron -> generalMatrix.saveMatrix("MatrixPerceptron");

    //Matrix hidden(100, 784);
    //Matrix output(10,100);
    //auto *network = new NeuralNetwork(784,100,10,0.1,10,hidden,output);
    //network ->trainNeuralNetwork(dataBase);
    //std::cout << network ->calculatePredictions(dataBase, 1000);
    //network -> hiddenMatrix.saveMatrix("HiddenMatrix");
    //network -> outputMatrix.saveMatrix("OutputMatrix");

    NeuralNetwork network = loadNetwork();
    network.testImage(dataBase, 7483);
    Matrix m = convertedImage("piec");
    //network.test(m);

    return 0;
}
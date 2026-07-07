#ifndef NEU_NETWORK_NETWORK_H
#define NEU_NETWORK_NETWORK_H

#endif //NEU_NETWORK_NETWORK_H

#include <vector>
#include "Matrix.h"
#include "Layer.h"
#include <fstream>

struct Network {
    float trainingRate;
    Network(float trainingRate = 0.01) : trainingRate(trainingRate) {}

    std::vector<Layer> layers;

    void add_layer(Layer layer) {
        layers.push_back(layer);
    }

    Matrix forward(const Matrix& input) {
        Matrix out = input;
        for (int i = 0; i < layers.size(); i++) {
            out = layers[i].forward(out);
        }
        return out;
    }

    Matrix backward(const Matrix& prediction, const Matrix& expected, int currentEpoch) {
        Matrix err = prediction-expected;
        float currentRate = trainingRate * (1.0f /(1.0f+0.0001f*currentEpoch));

        for (int i = layers.size() - 1; i >= 0; i--) {

            err = err.hadamard(layers[i].preActivation.apply_Function(layers[i].activation_derivative));
            Matrix weightGradient = (-layers[i].prev) * err;

            for (auto& val : weightGradient.data)
                val = std::max(-1.0f, std::min(1.0f, val));

            Matrix newErr = err * (-layers[i].weights);

            for (auto& val : err.data)
                val = std::max(-1.0f, std::min(1.0f, val));

            layers[i].weights = layers[i].weights - weightGradient*currentRate;
            layers[i].biases = layers[i].biases - err*currentRate;

            err = newErr;
        }

        return err;
    }

    float loss(const Matrix& prediction, const Matrix& expected) {
        float loss = 0.0f;
        for (int i = 0; i < prediction.cols*prediction.rows; i++) {
            loss += expected.data[i] * log(std::max(prediction.data[i], 1e-7f));
        }
        return -loss;
    }

    void save(const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: creating file " << filename << std::endl;
            exit(-1);
        }

        for (int i = 0; i < layers.size(); i++) {
            //weights saving
            file<< layers[i].weights.cols<< " " << layers[i].weights.rows << "\n";
            for (int j = 0; j < layers[i].weights.data.size(); j++)
                file << layers[i].weights.data[j] << " ";
            file << "\n";

            //bias saving
            file << layers[i].biases.cols<< " " << layers[i].biases.rows << "\n";
            for (int j = 0; j < layers[i].biases.data.size(); j++)
                file << layers[i].biases.data[j] << " ";
            file << "\n";
        }
        std::cout << filename<< " is done saving." << std::endl;
    }

    void load(const std::string& filename, bool expansion) {
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: opening file " << filename << std::endl;
            exit(-1);
        }

        for (int i = 0; i < layers.size(); i++) {
            int lCols, lRows;
            file >> lCols >> lRows;

            if (lCols != layers[i].weights.cols || lRows != layers[i].weights.rows) {
                if (!expansion) {
                    std::cerr << "Error: weight size mismatch in given file." << filename << std::endl;
                    exit(-1);
                }else {
                    Matrix expandedWeights(layers[i].weights.cols, layers[i].weights.rows);
                    for (int r = 0; r < lRows; r++)
                        for (int c = 0; c < lCols; c++)
                            file>>expandedWeights.data[r * expandedWeights.cols + c];
                    expandedWeights.randomize();
                    layers[i].weights = expandedWeights;
                }
            }else {
                for (int j = 0; j < lCols*lRows; j++)
                    file >> layers[i].weights.data[j];
            }

            file >> lCols >> lRows;

            if (lCols != layers[i].biases.cols || lRows != layers[i].biases.rows) {
                if (!expansion) {
                    std::cerr << "Error: weight size mismatch in given file." << filename << std::endl;
                    exit(-1);
                }else {
                    Matrix expandedBias(layers[i].biases.cols, layers[i].biases.rows);
                    for (int r = 0; r < lRows; r++)
                        for (int c = 0; c < lCols; c++)
                            file>>expandedBias.data[r * expandedBias.cols + c];
                    expandedBias.randomize();
                    layers[i].biases = expandedBias;
                }
            }else {
                for (int j = 0; j < lCols*lRows; j++)
                    file >> layers[i].biases.data[j];
            }
        }
        std::cout << filename<< " is done loading." << std::endl;
    }

};
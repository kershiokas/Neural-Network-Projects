#ifndef NEU_NETWORK_MATRIX_H
#define NEU_NETWORK_MATRIX_H

#endif //NEU_NETWORK_MATRIX_H

#pragma once
#include <iostream>
#include <vector>
#include <random>

class Matrix {
    public:
    int cols, rows;
    std::vector<float> data;

    Matrix() : cols(0), rows(0), data(0) {}
    Matrix(int cols, int rows) : cols(cols), rows(rows), data(rows * cols, 0.0f){}
    Matrix(const std::vector<float>& dataInput, int cols, int rows) : cols(cols), rows(rows), data(dataInput) {}

    float& find(int c, int r) {
        return data[r * cols + c];
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            std::cerr << "Invalid input size." << std::endl;
            exit(0);
        }

        Matrix result(other.cols, rows);
        Matrix transponed = -other;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < other.cols; j++)
                for (int k = 0; k < cols; k++)
                    result.data[i * other.cols + j] += data[i * cols + k] * transponed.data[j * cols + k];

        return result;
    }

    Matrix hadamard(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows) {
            std::cerr << "Invalid input size." << std::endl;
            exit(0);
        }

        Matrix result = *this;
        for (int i = 0; i < rows * cols; i++)
            result.data[i] = data[i] * other.data[i];

        return result;
    }

    Matrix operator*(float scalar) const {
        Matrix result = *this;
        for (int i = 0; i < rows * cols; i++)
            result.data[i] *= scalar;
        return result;
    }

    Matrix operator -() const{
        Matrix result = *this;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[j * rows + i] = data[i * cols + j];

        result.rows = cols;
        result.cols = rows;

        return result;
    }

    Matrix operator+(const Matrix& other) const{
        if (cols != other.cols || rows != other.rows) {
            std::cerr << "Invalid input size." << std::endl;
            exit(0);
        }

        Matrix result = *this;
        for (int i = 0; i < cols*rows; i++)
                result.data[i] = data[i] + other.data[i];

        return result;
    }

    Matrix operator-(const Matrix& other) const{
        if (cols != other.cols || rows != other.rows) {
            std::cerr << "Invalid input size." << std::endl;
            exit(0);
        }

        Matrix result = *this;
        for (int i = 0; i < cols*rows; i++)
            result.data[i] = data[i] - other.data[i];

        return result;
    }

    Matrix apply_Function(float(*fn)(float)) {
        Matrix result = *this;
        for (int i = 0; i < rows * cols; i++)
            result.data[i] = fn(data[i]);
        return result;
    }

    void randomize() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < cols * rows; i++)
            data[i] = dist(rng);
    }

    void randomize(int inputSize) {
        std::mt19937 rng(std::random_device{}());
        float scale = sqrt(2.0f / inputSize);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (int i = 0; i < cols * rows; i++)
            data[i] = dist(rng);
    }

    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                std::cout << data[i * cols + j] << " ";
            std::cout << std::endl;
        }
    }

};
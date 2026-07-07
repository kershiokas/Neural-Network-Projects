#ifndef NEU_NETWORK_LAYER_H
#define NEU_NETWORK_LAYER_H

#endif //NEU_NETWORK_LAYER_H

#pragma once
#include <cmath>
#include "Matrix.h"

struct Layer {
    Matrix weights;
    Matrix biases;
    Matrix preActivation;
    Matrix prev;

    float(*activation_function)(float);
    float(*activation_derivative)(float);

    Layer(int inputSize, int outSize, float(*activation_function)(float), float(*activation_derivative)(float)) :
    weights(outSize, inputSize), biases(outSize, 1),
    activation_function(activation_function), activation_derivative(activation_derivative) {
        weights.randomize(inputSize);
        biases.randomize();
    }


    Matrix forward(const Matrix& input) {
        prev = input;
        preActivation = input * weights + biases;

        return preActivation.apply_Function(activation_function);
    }
};
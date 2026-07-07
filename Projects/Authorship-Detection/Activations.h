#ifndef NEU_NETWORK_ACTIVATIONS_H
#define NEU_NETWORK_ACTIVATIONS_H

#endif //NEU_NETWORK_ACTIVATIONS_H

#pragma once
#include <cmath>

inline float relu(float x) {
    if (x > 0.0f) {
        return x;
    }
    return 0.0f;
}

inline float relu_derivative(float x) {
    if (x > 0.0f) {
        return 1.0f;
    }
    return 0.0f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoid_derivative(float x) {
    return sigmoid(x) * (1.0f - sigmoid(x));
}

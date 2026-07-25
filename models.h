#pragma once
#include "config.h"

template <typename T>
class LinearRegression {
    public:
        std::vector<std::vector<T>> points;
        void loss_function(T w, T b);
        std::pair<T, T> gradient_descent(T w_now, T b_now, T L);
};

#include "linear_regression.cpp"


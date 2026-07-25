#pragma once
#include "config.h"

template <typename T>
class LinearRegression {
    public:
        std::vector<std::vector<T>> points;
        void loss_function(std::vector<std::vector<T>> w, T b);
        std::pair<std::vector<std::vector<T>>, T> gradient_descent(std::vector<std::vector<T>> w_now, T b_now, T L);
};

#include "linear_regression.cpp"

#pragma once
#include "config.h"

// This file contains Machine Learning models.
// you can use by including " #include "models.h" "
// and creating a object of a required model class 
// call it's data members/member functions
// to perform the required job.

// Linear Regression model.
template <typename T>
class LinearRegression {
    public:
        std::vector<std::vector<T>> points; // data member to store data.
        void loss_function(const std::vector<std::vector<T>>& w, T b); // function to calculate MSE loss.
        std::pair<std::vector<std::vector<T>>, T> gradient_descent(const std::vector<std::vector<T>>& w_now, T b_now, T L);
};

#include "linear_regression.cpp"

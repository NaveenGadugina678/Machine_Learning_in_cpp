// Linear Regression model.

#pragma once
#include "models.h"
#include "matrices.h"

// function to calculate MSE loss.
template <typename T>
void LinearRegression<T>::loss_function(const std::vector<std::vector<T>>& w, T b) {
    T total_error = 0;

    for (size_t i = 0 ; i < points.size() ; i++) { // traversing each data.
        std::vector<std::vector<T>> X;
        for (size_t j = 0 ; j < points[0].size() - 1 ; j++) { // features.
            X.push_back({points[i][j]});
        }

        T y = points[i].back(); // label.
        std::vector<std::vector<T>> result = matrix::multiplication(w, X);

        total_error += pow((y - (result[0][0] + b)), 2); // MSE calculation.
    }
    
    total_error /= points[0].size();
    std::cout <<"Loss :" <<  total_error;
}

// Gradient descent : 
// new_weight = old_weight - (small_amount * weight_slope)
// new_bias = old_bias - (small_amount * bias_slope). 
template <typename T>
std::pair<std::vector<std::vector<T>>, T> LinearRegression<T>::gradient_descent(const std::vector<std::vector<T>>&  weight_now, T bias_now, T L) {

    std::vector<std::vector<T>> weight_slope(weight_now.size(), std::vector<T> (weight_now[0].size()));
    T bias_slope = 0;
    T n = static_cast<T> (points.size());

    for (size_t i = 0 ; i < n ; i++) { // traversing all data points.

        std::vector<std::vector<T>> X; // features.
        for (size_t k = 0 ; k < points[0].size() - 1 ; k++) {
            X.push_back({points[i][k]});
        }

        T y = points[i].back(); // label.
        std::vector<std::vector<T>> result = matrix::multiplication(weight_now, X); // storing W by X-transpose.
        
        for (size_t j = 0 ; j < weight_now[0].size() ; j++) { // calculating weight slope by differentiating loss function wrt weight.
            weight_slope[0][j] += (static_cast<T>(2) / n) * X[j][0] * (result[0][0] + bias_now - y);
        }

        bias_slope += (static_cast<T>(2) / n) * (result[0][0] + bias_now - y); // calculating bias slope by differntiating loss fun wrt                                                                                // to bias.
    }
    
    std::vector<std::vector<T>> weight = matrix::subtraction(weight_now, matrix::scalar_multiplication(L, weight_slope));
    T bias = bias_now - (L * bias_slope);

    return {weight, bias};
}

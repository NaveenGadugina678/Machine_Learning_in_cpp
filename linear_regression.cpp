#include "models.h"
#include "matrices.h"

template <typename T>
void LinearRegression<T>::loss_function(std::vector<std::vector<T>> w, T b) {
    T total_error = 0;

    for (size_t i = 0 ; i < points.size() ; i++) {
        std::vector<std::vector<T>> X;
        for (size_t j = 0 ; j < points[0].size() - 1 ; j++) {
            X.push_back({points[i][j]});
        }

        T y = points[i].back();
        std::vector<std::vector<T>> result = matrix::multiplication(w, X);

        total_error += pow((y - (result[0][0] + b)), 2);
    }
    
    total_error /= points[0].size();
    std::cout <<"Loss :" <<  total_error;
}

template <typename T>
std::pair<std::vector<std::vector<T>>, T> LinearRegression<T>::gradient_descent(std::vector<std::vector<T>> weight_now, T bias_now, T L) {
    std::vector<std::vector<T>> weight_slope(weight_now.size(), std::vector<T> (weight_now[0].size()));

    T bias_slope = 0;
    T n = static_cast<T> (points.size());

    for (size_t i = 0 ; i < n ; i++) {
        std::vector<std::vector<T>> X;
        for (size_t k = 0 ; k < points[0].size() - 1 ; k++) {
            X.push_back({points[i][k]});
        }

        T y = points[i].back();
        std::vector<std::vector<T>> result = matrix::multiplication(weight_now, X);
        
        for (size_t j = 0 ; j < weight_now[0].size() ; j++) {
            weight_slope[0][j] += (static_cast<T>(2) / n) * X[j][0] * (result[0][0] + bias_now - y);
        }

        bias_slope += (static_cast<T>(2) / n) * (result[0][0] + bias_now - y);
    }
    
    std::vector<std::vector<T>> weight = matrix::subtraction(weight_now, matrix::scalar_multiplication(L, weight_slope));
    T bias = bias_now - (L * bias_slope);

    return {weight, bias};
}

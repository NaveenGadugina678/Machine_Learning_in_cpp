#include "models.h"

template <typename T>
void LinearRegression<T>::loss_function(T w, T b) {
    T total_error = 0;
    T x, y;

    for (size_t i = 0 ; i < points[0].size() ; i++) {
        x = points[0][i];
        y = points[1][i];
        
        total_error += pow((y - (w * x + b)), 2);
    }
    
    total_error /= points[0].size();
    std::cout <<"Loss :" <<  total_error;
}

template <typename T>
std::pair<T, T> LinearRegression<T>::gradient_descent(T weight_now, T bias_now, T L) {
    T weight_slope = 0;
    T bias_slope = 0;
    T n = static_cast<T> (points[0].size());

    for (size_t i = 0 ; i < n ; i++) {
        T x = points[0][i];
        T y = points[1][i];

        weight_slope += (static_cast<T>(2) / n) * x * ((weight_now * x + bias_now) - y);
        bias_slope += (static_cast<T>(2) / n) * ((weight_now * x + bias_now) - y);
    }

    T weight = weight_now - (L * weight_slope);
    T bias = bias_now - (L * bias_slope);

    return {weight, bias};
}

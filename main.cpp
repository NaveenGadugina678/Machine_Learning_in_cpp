#include "config.h"
#include "models.h"

int main() {
    LinearRegression<float> lr;
    int n;
    std::cout << "Enter the size : ";
    std::cin >> n;

    std::vector<float> X(n), y(n);

    std::cout << "Enter X values : ";
    for (auto& val : X) {
        std::cin >> val;
    }

    std::cout << "Enter y values : ";
    for (auto& val : y) {
        std::cin >> val;
    }

    lr.points.push_back(X);
    lr.points.push_back(y);

    int epoch;
    std::cout << "Enter epoch : ";
    std::cin >> epoch;

    float weight = 0;
    float bias = 0;
    float L;
    std::cout << "Enter learning rate(alpha) : ";
    std::cin >> L;
    std::pair<float, float> result;

    for (int i = 0 ; i < epoch ; i++) {
        result = lr.gradient_descent(weight, bias, L);
        weight = result.first;
        bias = result.second;
    }

    std::cout << "weight = " << weight << "\nbias = " << bias << "\n";
    
    char yes_or_no;
    std::cout << "Do you want to test [y/n]? : ";
    std::cin >> yes_or_no;

    while (yes_or_no == 'y') {
        float x_test;
        std::cout << "x : ";
        std::cin >> x_test;
        
        float y_test;
        y_test = (weight * x_test) + bias;
        std::cout << y_test << "\n";

        std::cout << "Do you want to test [y/n]? : ";
        std::cin >> yes_or_no;
    }

    return 0;
}

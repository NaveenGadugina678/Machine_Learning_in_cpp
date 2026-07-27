#include "config.h"
#include "models.h"
#include "matrices.h"

int main() {

    // creating an object of the class linear regression model.
    LinearRegression<float> lr;
    
    // collecting data from the user.
    int n; // number of data points.
    std::cout << "Enter the size : ";
    std::cin >> n;
    
    int no_of_var; // number of features.
    std::cout << "Enter the number of varibles/features : ";
    std::cin >> no_of_var;

    std::vector<std::vector<float>> data(n, std::vector<float> (no_of_var + 1));

    std::cout << "Enter features and target : ";
    for (int i = 0 ; i < n ; i++) {
        for (int j = 0 ; j < no_of_var + 1 ; j++) {
            std::cin >> data[i][j];
        }
    }

    lr.points = data; // storing the collected data in the data member of the class linear regression.

    int epoch; // a hyperparameter which tells how many times to iterate the entire data.
    std::cout << "Enter epoch : ";
    std::cin >> epoch;

    std::vector<std::vector<float>> weight(1, std::vector<float> (no_of_var));
    float bias = 0;
    
    float L; // a hyperparameter which tells speed of the weight and the bias change.
    std::cout << "Enter learning rate(alpha) : ";
    std::cin >> L;
    std::pair<std::vector<std::vector<float>>, float> result;
    
    for (int i = 0 ; i < epoch ; i++) { // traversing entire data epoch times.
        result = lr.gradient_descent(weight, bias, L);
        weight = result.first;
        bias = result.second;
    }
    
    // printing weights and the bias.
    std::cout << "weights : ";
    for (auto& ele : weight[0]) {
        std::cout << ele << " ";
    }
    std::cout << "\nbias : " << bias << "\n";

    char yes_or_no;
    std::cout << "Do you want to test [y/n]? : ";
    std::cin >> yes_or_no;

    while (yes_or_no == 'y') {
        std::vector<std::vector<float>> X_test;
        std::cout << "X : ";
        for (size_t i = 0 ; i < no_of_var ; i++) {
            float ele;
            std::cin >> ele;
            X_test.push_back({ele});
        }

        float y_test;
        
        std::vector<std::vector<float>> product = matrix::multiplication(weight, X_test);
        y_test = product[0][0] + bias;
        std::cout << y_test << "\n";

        std::cout << "Do you want to test [y/n]? : ";
        std::cin >> yes_or_no;
    }

    return 0;
}

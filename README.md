# 🤖 Machine Learning in C++

A lightweight, from-scratch C++ implementation of fundamental Machine Learning algorithms and mathematical utilities. Built with modern, templated C++ for high efficiency, flexibility, and zero external dependencies.

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-11%2F14%2F17%2F20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg?style=for-the-badge)](https://github.com/NaveenGadugina678/Machine_Learning_in_cpp)

---

## 📌 Overview

This repository provides an educational and modular foundation for Machine Learning implemented entirely from scratch in C++. It includes custom matrix linear algebra operations, model abstraction, loss function calculation, and iterative optimization via Gradient Descent.

Whether you're exploring how machine learning algorithms work under the hood or seeking a dependency-free C++ implementation, this project serves as a clear, extensible starting point.

---

## ✨ Features

- 🧮 **Custom Matrix Engine (`namespace matrix`)**:
  - Matrix multiplication, matrix subtraction, and scalar multiplication.
  - Template-based numerical operations supporting `float`, `double`, etc.
  - Built-in bounds and dimension compatibility checking with exception handling.

- 📈 **Multivariate Linear Regression (`LinearRegression<T>`)**:
  - Support for multi-feature datasets ($X_1, X_2, \dots, X_n$).
  - **Mean Squared Error (MSE)** loss computation.
  - Analytical **Gradient Descent** optimization for updating weights vector ($W$) and bias ($b$).

- 💻 **Interactive CLI Interface (`main.cpp`)**:
  - Interactive prompts to input custom dataset size, feature dimensions, training data, epoch count, and learning rate ($\alpha$).
  - Real-time training loop displaying optimized weights and bias.
  - Interactive prediction mode to test trained models on fresh input features ($X_{test}$).

---

## 📐 Mathematical Formulation

### 1. Hypothesis Function
For an input vector $X$ with $m$ features:
$$\hat{y} = W \cdot X + b$$
where $W = [w_1, w_2, \dots, w_m]$ is the weight vector and $b$ is the scalar bias.

### 2. Loss Function (Mean Squared Error)
$$J(W, b) = \frac{1}{N} \sum_{i=1}^{N} \left( y^{(i)} - (W \cdot X^{(i)} + b) \right)^2$$

### 3. Gradient Descent Update Rules
At each epoch, parameters are updated using the gradients of the loss function:
$$\frac{\partial J}{\partial W_j} = \frac{2}{N} \sum_{i=1}^{N} X_j^{(i)} \left( (W \cdot X^{(i)} + b) - y^{(i)} \right)$$

$$\frac{\partial J}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} \left( (W \cdot X^{(i)} + b) - y^{(i)} \right)$$

$$W_{new} = W_{old} - \alpha \cdot \frac{\partial J}{\partial W}$$
$$b_{new} = b_{old} - \alpha \cdot \frac{\partial J}{\partial b}$$

where $\alpha$ is the learning rate.

---

## 📁 Repository Structure

```
Machine_Learning_in_cpp/
├── CMakeLists.txt
├── examples
│   └── main.cpp
├── include
│   ├── config.h
│   ├── matrices.h
│   └── models.h
├── README.md
└── src
    └── linear_regression.cpp

```

---

## 🚀 Quick Start

### Prerequisites
- Any modern C++ compiler supporting C++11 or higher (`g++`, `clang++`, or MSVC).

### Building and Running

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NaveenGadugina678/Machine_Learning_in_cpp.git
   cd Machine_Learning_in_cpp
   ```

2. **Compile the Project**
   Using `g++`:
   ```bash
   g++ -std=c++17 main.cpp -o ml_app
   ```

3. **Execute**
   ```bash
   ./ml_app
   ```

---

## 💡 Example Interactive Session

```text
Enter the size : 4
Enter the number of variables/features : 1
Enter features and target : 
1 2
2 4
3 6
4 8
Enter epoch : 1000
Enter learning rate(alpha) : 0.01

weights : 1.999 
bias : 0.002

Do you want to test [y/n]? : y
X : 5
10.002
```

---

## 🛠️ Usage as a Library

You can easily integrate the `LinearRegression` model or `matrix` operations into your own C++ code:

```cpp
#include "config.h"
#include "models.h"
#include "matrices.h"

int main() {
    // Instantiate Linear Regression model with float
    LinearRegression<float> model;

    // Define dataset: {feature_1, feature_2, ..., target}
    model.points = {
        {1.0f, 2.0f},
        {2.0f, 4.0f},
        {3.0f, 6.0f}
    };

    std::vector<std::vector<float>> weight = {{0.0f}};
    float bias = 0.0f;
    float learning_rate = 0.01f;
    int epochs = 500;

    for (int i = 0; i < epochs; ++i) {
        auto [new_weight, new_bias] = model.gradient_descent(weight, bias, learning_rate);
        weight = new_weight;
        bias = new_bias;
    }

    std::cout << "Trained Weight: " << weight[0][0] << ", Bias: " << bias << std::endl;
    return 0;
}
```

---

## 🤝 Contributing

Contributions, feature suggestions (e.g., Logistic Regression, Neural Networks, Decision Trees), and optimization PRs are welcome! Feel free to open an issue or submit a Pull Request.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

# 🤖 Machine Learning in C++

A lightweight, from-scratch C++ implementation of fundamental Machine Learning algorithms and linear algebra utilities. Built with modern, templated C++ for high efficiency, flexibility, and zero external dependencies.

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-11%2F14%2F17%2F20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg?style=for-the-badge)](https://github.com/NaveenGadugina678/Machine_Learning_in_cpp)

---

## 📌 Overview

This repository provides an educational and modular foundation for Machine Learning implemented entirely from scratch in C++. It includes custom matrix linear algebra operations, model declarations, loss function calculation, and parameter optimization via Gradient Descent.

Whether you're exploring how machine learning algorithms work under the hood or seeking a dependency-free C++ implementation, this project serves as a clear, extensible starting point.

---

## ✨ Features

- 🧮 **Custom Matrix Engine (`namespace matrix`)**:
  - Matrix multiplication (`matrix::multiplication`)
  - Matrix subtraction (`matrix::subtraction`)
  - Scalar multiplication (`matrix::scalar_multiplication`)
  - Template-based numerical operations supporting `float`, `double`, etc.
  - Dimension compatibility checks with `std::invalid_argument` error throwing.

- 📈 **Multivariate Linear Regression (`LinearRegression<T>`)**:
  - Support for multi-feature datasets ($X_1, X_2, \dots, X_m$).
  - **Mean Squared Error (MSE)** loss computation (`loss_function`).
  - Analytical **Gradient Descent** optimization (`gradient_descent`) updating weight row vector ($W$) and bias scalar ($b$).

- 💻 **Interactive CLI Application (`main.cpp`)**:
  - Prompts for dataset size ($N$), feature count ($m$), training data (features + target), epochs, and learning rate ($L$).
  - Trains model using iterative gradient descent and outputs final weights and bias.
  - Includes interactive prediction loop to test trained models on fresh input features ($X_{\text{test}}$).

---

## 📐 Mathematical Formulation

### 1. Hypothesis Function
For a dataset sample with $m$ features represented as a column vector $X \in \mathbb{R}^{m \times 1}$ and weight row vector $W \in \mathbb{R}^{1 \times m}$:
$$\hat{y} = W \cdot X + b$$

### 2. Loss Function (Mean Squared Error)
$$J(W, b) = \frac{1}{N} \sum_{i=1}^{N} \left( y^{(i)} - (W \cdot X^{(i)} + b) \right)^2$$

### 3. Gradient Descent Update Rules
Gradients with respect to weight $W_j$ and bias $b$ over $N$ data samples:
$$\frac{\partial J}{\partial W_j} = \frac{2}{N} \sum_{i=1}^{N} X_j^{(i)} \left( W \cdot X^{(i)} + b - y^{(i)} \right)$$

$$\frac{\partial J}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} \left( W \cdot X^{(i)} + b - y^{(i)} \right)$$

Parameter updates using learning rate $L$:
$$W_{\text{new}} = W_{\text{old}} - L \cdot \frac{\partial J}{\partial W}$$
$$b_{\text{new}} = b_{\text{old}} - L \cdot \frac{\partial J}{\partial b}$$

---

## 📁 Repository Structure

```
Machine_Learning_in_cpp/
├── config.h               # Core standard header inclusions (<iostream>, <vector>, <cmath>, <utility>)
├── matrices.h             # Templated matrix operation functions (multiplication, subtraction, scalar ops)
├── models.h               # Class template declarations for ML models (LinearRegression<T>)
├── linear_regression.cpp  # Implementation of loss_function and gradient_descent
└── main.cpp               # Interactive main program for dataset input, training, and inference
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
   g++ -std=c++17 main.cpp -o main
   ```

3. **Execute**
   ```bash
   ./main
   ```

---

## 💡 Example Interactive Session

```text
Enter the size : 4
Enter the number of varibles/features : 1
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
Do you want to test [y/n]? : n
```

---

## 🛠️ Usage as a Library

You can include `models.h` directly in your C++ code:

```cpp
#include "config.h"
#include "models.h"
#include "matrices.h"
#include <iostream>

int main() {
    // Instantiate Linear Regression model with float
    LinearRegression<float> model;

    // Define dataset: {feature_1, feature_2, ..., target}
    model.points = {
        {1.0f, 2.0f},
        {2.0f, 4.0f},
        {3.0f, 6.0f},
        {4.0f, 8.0f}
    };

    std::vector<std::vector<float>> weight = {{0.0f}};
    float bias = 0.0f;
    float learning_rate = 0.01f;
    int epochs = 1000;

    for (int i = 0; i < epochs; ++i) {
        auto result = model.gradient_descent(weight, bias, learning_rate);
        weight = result.first;
        bias = result.second;
    }

    std::cout << "Optimized Weight: " << weight[0][0] << ", Bias: " << bias << std::endl;
    return 0;
}
```

---

## 🤝 Contributing

Contributions are welcome! Potential areas for contribution include:
- Adding new models (e.g., Logistic Regression, Neural Networks, Decision Trees).
- Expanding matrix utilities (transpose, determinant, matrix inversion).
- Adding CMake build configuration & automated unit tests (Catch2/GoogleTest).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

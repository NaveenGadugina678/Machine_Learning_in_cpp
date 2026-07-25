#pragma once
#include <vector>
#include <stdexcept>

namespace matrix {
    template <typename T>
    std::vector<std::vector<T>> multiplication(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        if (A.empty() || B.empty()) {
            throw std::invalid_argument("Matrices cannot be empty.");
        }

        if (A[0].size() != B.size()) {
            throw std::invalid_argument("Matrices are incompatible for multiplication.");
        }
        
        size_t m = A.size();
        size_t n = B[0].size();
        std::vector<std::vector<T>> result(m, std::vector<T> (n)); 

        for (size_t i = 0 ; i < m ; i++) {
            for (size_t j = 0 ; j < n ; j++) {
                for (size_t k = 0 ; k < B.size() ; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }

    template <typename T>
    std::vector<std::vector<T>> subtraction(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        if (A.empty() || B.empty()) {
            throw std::invalid_argument("Matrices cannot be empty.");
        }

        if (A.size() != B.size() || A[0].size() != B[0].size()) {
            throw std::invalid_argument("Matrices are incompatible for subtraction.");
        }

        size_t m = A.size();
        size_t n = A[0].size();
        std::vector<std::vector<T>> difference(m, std::vector<T> (n));
        
        for (size_t i = 0 ; i < m ; i++) {
            for (size_t j = 0 ; j < n ; j++) {
                difference[i][j] = A[i][j] - B[i][j];
            }
        }

        return difference;
    }

    template <typename T>
    std::vector<std::vector<T>> scalar_multiplication(const T a, const std::vector<std::vector<T>>& A) {
        if (A.empty()) {
            throw std::invalid_argument("Matrix cannot be empty.");
        }
            
        size_t m = A.size();
        size_t n = A[0].size();
        std::vector<std::vector<T>> B(m, std::vector<T> (n));

        for (size_t i = 0 ; i < m ; i++) {
            for (size_t j = 0 ; j < n ; j++) {
                B[i][j] = A[i][j] * a;
            }
        }

        return B;
    }
}

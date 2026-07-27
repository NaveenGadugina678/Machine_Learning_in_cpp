#pragma once
#include <vector>
#include <stdexcept>

// This file can be used by including " #include "matrices.h" "
// in your working file and call the functions needed according
// to the operation to be performed.

namespace matrix {

    //matrix multiplication.
    template <typename T>
    std::vector<std::vector<T>> multiplication(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        
        // error throwing.
        if (A.empty() || B.empty()) {
            throw std::invalid_argument("Matrices cannot be empty.");
        }

        if (A[0].size() != B.size()) {
            throw std::invalid_argument("Matrices are incompatible for multiplication.");
        }
        
        // product calculation.
        size_t m = A.size();
        size_t n = B[0].size();
        std::vector<std::vector<T>> result(m, std::vector<T> (n)); // product matrix. 
        
        // multiplying row to col.
        for (size_t i = 0 ; i < m ; i++) { // row traversal.
            for (size_t j = 0 ; j < n ; j++) { // col traversal.
                for (size_t k = 0 ; k < B.size() ; k++) { // elemental traversal.
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }

    // matrix subtraction.
    template <typename T>
    std::vector<std::vector<T>> subtraction(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        
        // error throwing.
        if (A.empty() || B.empty()) {
            throw std::invalid_argument("Matrices cannot be empty.");
        }

        if (A.size() != B.size() || A[0].size() != B[0].size()) {
            throw std::invalid_argument("Matrices are incompatible for subtraction.");
        }
        
        // difference calculation (basically we are doing A - B).
        size_t m = A.size();
        size_t n = A[0].size();
        std::vector<std::vector<T>> difference(m, std::vector<T> (n)); // difference matrix.
        
        for (size_t i = 0 ; i < m ; i++) { // row traversal.
            for (size_t j = 0 ; j < n ; j++) { // col traversal.
                difference[i][j] = A[i][j] - B[i][j];
            }
        }

        return difference;
    }

    // multiplying a matrix by a scalar value.
    template <typename T>
    std::vector<std::vector<T>> scalar_multiplication(const T a, const std::vector<std::vector<T>>& A) {

        // error throwing.
        if (A.empty()) {
            throw std::invalid_argument("Matrix cannot be empty.");
        }
            
        size_t m = A.size();
        size_t n = A[0].size();
        std::vector<std::vector<T>> B(m, std::vector<T> (n)); // resultant matrix.

        for (size_t i = 0 ; i < m ; i++) { // row traversal.
            for (size_t j = 0 ; j < n ; j++) { //col traversal.
                B[i][j] = A[i][j] * a;
            }
        }

        return B;
    }
}

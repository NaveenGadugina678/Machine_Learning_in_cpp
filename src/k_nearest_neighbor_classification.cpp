#pragma once
#include "models.h"
#include "matrices.h"
#include <map>
#include <vector>

template <typename T>
T KNearestNeighbor<T>::predict(const std::vector<T>& A, T k) {
    std::map<T, T> distances;

    for (size_t i = 0 ; i < points.size() ; i++) {
        for (auto& point : points[i]) {
            T distance = matrix::euclidean_distance(A, point);
            distances[distance] = i;
        }
    }
    
    T result = 0;
    std::unordered_map<T, int> freq;
    auto it = distances.begin();
    
    for (size_t i = 0 ; i < k && it != distances.end(); i++, it++) {
        freq[it->second]++;
    }
    
    for (auto it : freq) {
        if (result < it.second) result = it.first;
    }
    
    return result;
}
    

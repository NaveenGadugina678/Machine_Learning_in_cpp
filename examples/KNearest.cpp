#include "config.h"
#include "models.h"

int main() {
    KNearestNeighbor<float> m1; // creating a instance of a class K Nearest neighbor.
    
    int n;
    std::cout << "Enter number of categories : ";
    std::cin >> n;
    
    int m;
    std::cout << "Enter each number of points in each category : ";
    std::cin >> m;
    
    int d;
    std::cout << "Enter dimension of the point(vector) : ";
    std::cin >> d;
    
    std::vector<std::vector<std::vector<float>>> data(n, std::vector<std::vector<float>> (m));
    std::cout << "Enter data points category wise : ";
    
    for (int i = 0 ; i < n ; i++) {
        for (int j = 0 ; j < m ; j++) {
            for (int k = 0 ; k < d ; k++) {
                float num;
                std::cin >> num;

                data[i][j].push_back(num);
            }
        }
    }

    m1.points = data;
    
    char yes_or_no;
    std::cout << "Do you want to test(y/n) : ";
    std::cin >> yes_or_no;

    while (yes_or_no == 'y') {
        std::cout << "Enter the point : ";
        std::vector<float> test(d);
        
        for (auto& x : test) {
            std::cin >> x;
        }

        std::cout << "Belongs to category : " << 1 + m1.predict(test, 11) << "\n";

        std::cout << "Do you want to test(y/n) : ";
        std::cin >> yes_or_no;
    }

    return 0;
}

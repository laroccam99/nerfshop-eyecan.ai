#include <iostream>
#include <string>
#include <map>
#include <Eigen/Dense>

class selection_map {
private:
    static std::map<std::size_t, Eigen::Vector3f> privateMap; 

public:
    selection_map() {
        privateMap[0] = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
        privateMap[1] = Eigen::Vector3f(4.0f, 5.0f, 6.0f);
        privateMap[2] = Eigen::Vector3f(7.0f, 8.0f, 9.0f);
    }

    std::map<std::size_t, Eigen::Vector3f> getPrivateMap() const {
        return privateMap;
    }
};    
#include <iostream>
#include <string>
#include <map>
#include <Eigen/Dense>

class selection_map {
private:
    static std::map<std::size_t, Eigen::Vector3f> privateMap; 

public:
    selection_map() {}

    std::map<std::size_t, Eigen::Vector3f> getPrivateMap() const {
        return privateMap;
    }

    void updatePrivateMap(std::size_t id, Eigen::Vector3f coord) const {
        //std::cout << "id: " << id << ", coord: " << coord << "Added to map" << std::endl;
        privateMap.insert(std::make_pair(id, coord));            
    }
};    
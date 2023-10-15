#include <iostream>
#include <string>
#include <map>
#include <Eigen/Dense>

class selection_map {
private:
    static std::map<std::uint32_t, Eigen::Vector3f> privateMap; 

public:
    selection_map() {}

    std::map<std::uint32_t, Eigen::Vector3f> getPrivateMap() const {
        return privateMap;
    }

    void add_to_privateMap(std::uint32_t id, Eigen::Vector3f coord) {
        //std::cout << "id: " << id << ", coord: " << coord << "Added to map" << std::endl;
        privateMap.insert(std::make_pair(id, coord));            
    }

    bool remove_from_privateMap(std::uint32_t id) {
        auto iterator = privateMap.find(id);
        if (iterator != privateMap.end()) {         //controlla se quell'elemento è davvero contenuto nella mappa
            privateMap.erase(iterator);
            return true;
        }
        return false;
    }
};    
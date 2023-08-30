#include <Eigen/Dense>
#include <iostream>
#include <vector>

class helper_selected_pixels {
private:
    static std::vector<Eigen::Vector2i> selected_pixels; 

public:
    helper_selected_pixels() {}

    std::vector<Eigen::Vector2i> get_selected_pixels() const {
        return selected_pixels;
    }

    void set_selected_pixels(std::vector<Eigen::Vector2i> pixels) {
        selected_pixels = pixels; 
        std::cout << "helper selected_pixels size: " << selected_pixels.size() << std::endl;
    }

    void clear_selected_pixels() {
        selected_pixels.clear();
    }
};    
#include <neural-graphics-primitives/selection_map.h>
#include <Eigen/Dense>

std::map<std::uint32_t, Eigen::Vector3f> selection_map::privateMap; //permette di allocare lo spazio necessario per la map

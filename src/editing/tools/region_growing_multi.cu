#include <neural-graphics-primitives/editing/tools/region_growing_multi.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <neural-graphics-primitives/common_nerf.h>
#include <cmath>
#include <tiny-cuda-nn/common_device.h>
#include <functional>



NGP_NAMESPACE_BEGIN

//GUI Button "Clear Selection"
// Launched by GrowingSelection::reset_growing selection grid
void RegionGrowingMulti::reset_growing(const std::vector<uint32_t>& selected_cells, int growing_level) {
    // Copy the density grid
    multi_density_grid_host.resize(multi_density_grid.size());
    multi_density_grid.copy_to_host(multi_density_grid_host);

    // Reset the selection grid (0 empty, 1 selected)	
    multi_selection_grid_bitfield = std::vector<uint8_t>(multi_density_grid_bitfield.size(), 0);

    // Reset the growing queue
    multi_growing_queue = std::queue<uint32_t>();

    uint32_t n_rays = selected_cells.size();

    // Reset the points (used for visualization)
    multi_selection_points.clear();
    multi_selection_cell_idx.clear();
    multi_selection_points.reserve(n_rays);
    multi_selection_cell_idx.reserve(n_rays);

    for (int i = 0; i < n_rays; i++) {
        uint32_t cell_idx = selected_cells[i];
        uint32_t level = cell_idx / NERF_GRIDVOLUME();

        // If it's bigger than the requested level, discard it
        if (level > growing_level) {
            continue;
        }
        // If it is smaller then uplift!
        if (level < growing_level) {
            cell_idx = get_upper_cell_idx(cell_idx, growing_level);
        };
        level = cell_idx / NERF_GRIDVOLUME();

        // Add all pixels to their reprojected coordinate in the queue
        multi_growing_queue.push(cell_idx);

        // Add visualization points
        // Invert morton coordinates to get xyz
        uint32_t pos_idx = cell_idx % NERF_GRIDVOLUME();
        uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
        uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
        uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
        multi_selection_points.push_back(get_cell_pos(x, y, z, level));
        multi_selection_cell_idx.push_back(cell_idx);
    }
}

void RegionGrowingMulti::upscale_selection(int current_level) {
    // If the current level is already the maximum cascade, we can't upscale
    if (current_level == multi_max_cascade)
        return;
    // Otherwise, upscale everything
    multi_growing_level = current_level + 1;

    // Reset the bitfield grid
    std::fill(multi_selection_grid_bitfield.begin(), multi_selection_grid_bitfield.end(), 0);
    
    // Upscale the existing points
    std::vector<uint32_t> new_cell_indices;
    multi_selection_points = std::vector<Eigen::Vector3f>();
    for (const auto cell_idx: multi_selection_cell_idx) {
        uint32_t new_cell_idx = get_upper_cell_idx(cell_idx, multi_growing_level);
        uint32_t new_pos_idx = new_cell_idx % (NERF_GRIDVOLUME());
        uint32_t x = tcnn::morton3D_invert(new_pos_idx>>0);
        uint32_t y = tcnn::morton3D_invert(new_pos_idx>>1);
        uint32_t z = tcnn::morton3D_invert(new_pos_idx>>2);
        Eigen::Vector3f cell_pos = get_cell_pos(x, y, z, multi_growing_level);
        multi_selection_points.push_back(cell_pos);
        new_cell_indices.push_back(new_cell_idx);
        set_bitfield_at(new_pos_idx, multi_growing_level, true, multi_selection_grid_bitfield.data());
    }
    multi_selection_cell_idx = new_cell_indices;

    // Upscale the growing queue too
    std::queue<uint32_t> new_growing_queue;
    while (!multi_growing_queue.empty()) {
        uint32_t current_cell = multi_growing_queue.front();
        multi_growing_queue.pop();
        new_growing_queue.push(get_upper_cell_idx(current_cell, multi_growing_level));
    }
    multi_growing_queue = new_growing_queue;
}

//GUI Button "Grow region", second function
void RegionGrowingMulti::grow_region(float density_threshold, ERegionGrowingMode region_growing_mode, int growing_level, int growing_steps) {
    // Make sure we can actually grow!
    if (multi_growing_queue.empty()) {
        std::cout << "Growing queue is empty!" << std::endl;
        return;
    }
    multi_growing_level = growing_level; 

    int i = 1;

    if (region_growing_mode == ERegionGrowingMode::Manual) {
        while (!multi_growing_queue.empty() && i <= growing_steps) {
            uint32_t current_cell = multi_growing_queue.front();
            float current_density = multi_density_grid_host[current_cell];
            multi_growing_queue.pop();

            // Get position (with corresponding level) to fetch neighbours
            uint32_t level = current_cell / (NERF_GRIDVOLUME());
            uint32_t pos_idx = current_cell % (NERF_GRIDVOLUME());

            // Sample accepted only if at requested level, statisfying density threshold and not already selected!
            if (!get_bitfield_at(pos_idx, level, multi_selection_grid_bitfield.data()) && current_density >= density_threshold && level == multi_growing_level) {
                
                // Test whether the new sample touches the boundary, if yes then upscale!
                if (is_boundary(pos_idx)) {
                    std::cout << "UPSAMPLING" << std::endl;
                    upscale_selection(multi_growing_level);
                    // Don´t forget to also upscale the current cell!
                    current_cell = get_upper_cell_idx(current_cell, multi_growing_level);
                    level = current_cell / (NERF_GRIDVOLUME());
                    pos_idx = current_cell % (NERF_GRIDVOLUME());
                }

                // Invert morton coordinates to get xyz
                uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
                uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
                uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
                // Add possible neighbours
                add_neighbours(multi_growing_queue, x, y, z, level);

                // Mark the current cell
                Eigen::Vector3f cell_pos = get_cell_pos(x, y, z, level);
                multi_selection_points.push_back(cell_pos);
                multi_selection_cell_idx.push_back(current_cell);
                set_bitfield_at(pos_idx, level, true, multi_selection_grid_bitfield.data());
                //std::cout << "multi_selection_cell_idx: " << current_cell << std::endl;
            }
            i++;
        }     
    } 
}

/*
bool RegionGrowingMulti::not_zero_coordinate(Eigen::Vector3f point_to_check) {
    //std::cout << "point_to_check: " << point_to_check << std::endl;
    if (point_to_check == Eigen::Vector3f(0.0f, 0.0f, 0.0f)) {
        std::cout << " Zero Coordinate Point discarded------------------------------------------------------------------------------- " << std::endl;
        return false;
    }
    else {
        return true;
    }
}
*/
//#########forse da rimuovere tutto ###############################################

// Queue needs to be copied because we'll exhaust it
template <typename T>
inline void to_json_queue(nlohmann::json& j, std::queue<T> queue) {
	std::vector<T> tmp_vec;
    tmp_vec.reserve(queue.size());
    while (!queue.empty()) {
        tmp_vec.push_back(queue.front());
        queue.pop();
    }
	to_json(j, tmp_vec);
}

template <typename T>
inline void from_json_queue(const nlohmann::json& j, std::queue<T>& queue) {
	std::vector<T> tmp_vec = j.get<std::vector<T>>();
	for (auto item: tmp_vec) {
		queue.push(item);
	}
}

nlohmann::json RegionGrowingMulti::to_json() {
        nlohmann::json j;

        j["selection_grid_bitfield"] = multi_selection_grid_bitfield;
        j["selection_points"] = multi_selection_points;
        j["selection_cell_idx"] = multi_selection_cell_idx;
        j["density_grid_host"] = multi_density_grid_host;
        // TODO: support saving of queue
        // to_json_queue<uint32_t>(j["growing_queue"], multi_growing_queue);

        return j;
    }

void RegionGrowingMulti::load_json(nlohmann::json& j) {
    std::cout << "most" << std::endl;
    from_json(j["selection_grid_bitfield"], multi_selection_grid_bitfield);
    from_json(j["selection_points"], multi_selection_points);
    from_json(j["selection_cell_idx"], multi_selection_cell_idx);
    from_json(j["density_grid_host"], multi_density_grid_host);
    // TODO: support reloading of the queue
    // from_json_queue<uint32_t>(j["growing_queue"], multi_growing_queue);
}

NGP_NAMESPACE_END
#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/hello_world.h>
#include <neural-graphics-primitives/selection_map.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <vector>
#include <queue>

#include <json/json.hpp>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

enum class ERegionGrowingMode : int {
	Manual,
	AppearanceBased,
};

static constexpr const char* RegionGrowingModeStr = "Manual\0AppearanceBased\0\0";

class RegionGrowingMulti {
public:
    RegionGrowingMulti (
        const tcnn::GPUMemory<float>& density_grid, 
        const tcnn::GPUMemory<uint8_t>& density_grid_bitfield,
        const uint32_t max_cascade) : multi_density_grid(density_grid), multi_density_grid_bitfield(density_grid_bitfield), multi_max_cascade(max_cascade) {}

    // Reset the growing selection grid
    void reset_growing(const std::vector<uint32_t>& selected_cells, int growing_levels);

    const std::vector<Eigen::Vector3f>& selection_points() const {
        return multi_selection_points;
    }

    const std::vector<uint32_t>& selection_cell_idx() const {
        return multi_selection_cell_idx;
    }

    const std::vector<uint8_t>& selection_grid_bitfield() const {
        return multi_selection_grid_bitfield;
    }

    const uint32_t growing_level() const {
        return multi_growing_level;
    }

    void upscale_selection(int current_level);

    void grow_region(float density_threshold, ERegionGrowingMode region_growing_mode, int growing_level, int growing_steps);

    nlohmann::json to_json();

    void load_json(nlohmann::json& j);

private:
    // Max cascade as specified with the dataset
    const uint32_t multi_max_cascade;
    uint32_t multi_growing_level;

    // Const data from the NeRF model
    const tcnn::GPUMemory<float>& multi_density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
    const tcnn::GPUMemory<uint8_t>& multi_density_grid_bitfield;

    // Selection data
    std::vector<uint8_t> multi_selection_grid_bitfield;
    std::vector<Eigen::Vector3f> multi_selection_points;
    std::vector<uint32_t> multi_selection_cell_idx;
    std::vector<float> multi_density_grid_host;
    std::queue<uint32_t> multi_growing_queue;

};

NGP_NAMESPACE_END
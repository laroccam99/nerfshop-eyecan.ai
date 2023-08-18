#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/json_binding.h>
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

class RegionGrowing {
public:
    RegionGrowing (
        const tcnn::GPUMemory<float>& density_grid, 
        const tcnn::GPUMemory<uint8_t>& density_grid_bitfield,
        const uint32_t max_cascade) : m_density_grid(density_grid), m_density_grid_bitfield(density_grid_bitfield), m_max_cascade(max_cascade) {}

    // Reset the growing selection grid
    void reset_growing(const std::vector<uint32_t>& selected_cells, int growing_levels);

    const std::vector<Eigen::Vector3f>& selection_points() const {
        return m_selection_points;
    }

    const std::vector<uint32_t>& selection_cell_idx() const {
        return m_selection_cell_idx;
    }

    const std::vector<uint8_t>& selection_grid_bitfield() const {
        return m_selection_grid_bitfield;
    }

    uint32_t growing_level() const {
        return m_growing_level;
    }
    //ANCORA NON UTILIZZATI, PER ORA INUTILI----------------------------------
    int get_min_ed_points_threshold() {
        return min_ed_points_threshold;
    }

    void set_min_ed_points_threshold(int value) {
        min_ed_points_threshold = value;
    }

    int get_max_ed_points_limit() {
        return max_ed_points_limit;
    }

    void set_max_ed_points_limit(int value) {
        max_ed_points_limit = value;
    }
    //----------------------------------------------------------------------
    void upscale_selection(int current_level);

    void grow_region(bool ed_flag, float density_threshold, ERegionGrowingMode region_growing_mode, int growing_level, int growing_steps);
    
    void equidistant_points(int min_ud_points_threshold);
    
    void equidistant_points(int min_ud_points_threshold, int interval);

    nlohmann::json to_json();

    void load_json(nlohmann::json& j);

private:
    // Max cascade as specified with the dataset
    const uint32_t m_max_cascade;
    uint32_t m_growing_level;
    //I valori di min_ed_points_threshold e max_ed_points_limit impostati qui vengono sovrascritti dal Button START in testbed.h in modo che abbiano valore == a max_num_operators di testbed.m_nerf.tracer
    int min_ed_points_threshold = 10;                   //soglia minima di punti equidistanti da ottenere
    int max_ed_points_limit = 10;                       //soglia massima di punti eq da ottenere

    // Const data from the NeRF model
    const tcnn::GPUMemory<float>& m_density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
    const tcnn::GPUMemory<uint8_t>& m_density_grid_bitfield;

    // Selection data
    std::vector<uint8_t> m_selection_grid_bitfield;
    std::vector<Eigen::Vector3f> m_selection_points;
    std::vector<uint32_t> m_selection_cell_idx;
    std::vector<float> m_density_grid_host;
    std::queue<uint32_t> m_growing_queue;

};

NGP_NAMESPACE_END
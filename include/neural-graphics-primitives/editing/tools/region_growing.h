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

    //Utilizzato nel LilSplit Button    -- TEMPORANEO
    void set_growing_level(uint32_t value) {
        m_growing_level = value;
    }
    
    int get_min_ed_points_threshold() {             //utile solo per stampa debug
        return min_ed_points_threshold;
    }

    //Permette di sincronizzare il numero di operatori con il numero di punti output del Grow Far
    void set_min_ed_points_threshold(int value) {
        min_ed_points_threshold = value;
    }

    int get_max_ed_points_limit() {                 //utile solo per stampa debug
        return max_ed_points_limit;
    }

    //Permette di sincronizzare il numero di operatori con il numero di punti output del Grow Far
    void set_max_ed_points_limit(int value) {
        max_ed_points_limit = value;
    }

    std::queue<uint32_t> get_m_growing_queue() {             //utile solo per stampa debug
        return m_growing_queue;
    }

    //Uilizzato dallo Split Button          
    void reset_push_m_growing_queue(int value) {
        std::queue<uint32_t> empty_queue;
        m_growing_queue = empty_queue;
        m_growing_queue.push(value);
    }

    //Utilizzato dallo Split Button 
    //Permette di effettuare il growing negli operatori secondari
    void set_m_density_grid_host() {
        m_density_grid_host.resize(m_density_grid.size());
        m_density_grid.copy_to_host(m_density_grid_host);
    }

    void upscale_selection(int current_level);

    //Se ed_flag==true fa GROW FAR, altrimenti utilizzato nel GROW&CAGE Button
    void grow_region(bool ed_flag, float density_threshold, ERegionGrowingMode region_growing_mode, int growing_level, int growing_steps);
    
    //Seleziona in modo uniforme solo alcuni punti superficiali distanti; si ferma al raggiungimento della soglia minima
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
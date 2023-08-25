#include <neural-graphics-primitives/editing/tools/region_growing.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <neural-graphics-primitives/common_nerf.h>
#include <cmath>
#include <tiny-cuda-nn/common_device.h>
#include <functional>

NGP_NAMESPACE_BEGIN

//GUI Button "Clear Selection"
// Launched by GrowingSelection::reset_growing selection grid
void RegionGrowing::reset_growing(const std::vector<uint32_t>& selected_cells, int growing_level) {
    // Copy the density grid
    std::cout << "reset_growing() " << std::endl;
    m_density_grid_host.resize(m_density_grid.size());
    m_density_grid.copy_to_host(m_density_grid_host);

    // Reset the selection grid (0 empty, 1 selected)	
    m_selection_grid_bitfield = std::vector<uint8_t>(m_density_grid_bitfield.size(), 0);

    // Reset the growing queue
    m_growing_queue = std::queue<uint32_t>();

    uint32_t n_rays = selected_cells.size();

    // Reset the points (used for visualization)
    m_selection_points.clear();
    m_selection_cell_idx.clear();
    m_selection_points.reserve(n_rays);
    m_selection_cell_idx.reserve(n_rays);

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
        m_growing_queue.push(cell_idx);

        // Add visualization points
        // Invert morton coordinates to get xyz
        uint32_t pos_idx = cell_idx % NERF_GRIDVOLUME();
        uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
        uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
        uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
        m_selection_points.push_back(get_cell_pos(x, y, z, level));
        m_selection_cell_idx.push_back(cell_idx);
    }
}

void RegionGrowing::upscale_selection(int current_level) {
    // If the current level is already the maximum cascade, we can't upscale
    if (current_level == m_max_cascade)
        return;
    // Otherwise, upscale everything
    m_growing_level = current_level + 1;

    // Reset the bitfield grid
    std::fill(m_selection_grid_bitfield.begin(), m_selection_grid_bitfield.end(), 0);
    
    // Upscale the existing points
    std::vector<uint32_t> new_cell_indices;
    m_selection_points = std::vector<Eigen::Vector3f>();
    for (const auto cell_idx: m_selection_cell_idx) {
        uint32_t new_cell_idx = get_upper_cell_idx(cell_idx, m_growing_level);
        uint32_t new_pos_idx = new_cell_idx % (NERF_GRIDVOLUME());
        uint32_t x = tcnn::morton3D_invert(new_pos_idx>>0);
        uint32_t y = tcnn::morton3D_invert(new_pos_idx>>1);
        uint32_t z = tcnn::morton3D_invert(new_pos_idx>>2);
        Eigen::Vector3f cell_pos = get_cell_pos(x, y, z, m_growing_level);
        m_selection_points.push_back(cell_pos);
        new_cell_indices.push_back(new_cell_idx);
        set_bitfield_at(new_pos_idx, m_growing_level, true, m_selection_grid_bitfield.data());
    }
    m_selection_cell_idx = new_cell_indices;

    // Upscale the growing queue too
    std::queue<uint32_t> new_growing_queue;
    while (!m_growing_queue.empty()) {
        uint32_t current_cell = m_growing_queue.front();
        m_growing_queue.pop();
        new_growing_queue.push(get_upper_cell_idx(current_cell, m_growing_level));
    }
    m_growing_queue = new_growing_queue;
}

//GUI Button "Grow region" and "Grow Far", second function
void RegionGrowing::grow_region(bool ed_flag, float density_threshold, ERegionGrowingMode region_growing_mode, int growing_level, int growing_steps) {
    // Make sure we can actually grow!
    if (m_growing_queue.empty()) {
        std::cout << "Growing queue is empty!" << std::endl;
        return;
    }
    m_growing_level = growing_level;            //attenzione a growing_level che sta a 0 

    int i = 1;

    if (region_growing_mode == ERegionGrowingMode::Manual) {
        while (!m_growing_queue.empty() && i <= growing_steps) {
            uint32_t current_cell = m_growing_queue.front();                //current_cell = m_selection_cell_idx 
            float current_density = m_density_grid_host[current_cell];      //con operatori secondari, di base m_density_grid_host è vuoto 
            m_growing_queue.pop();

            // Get position (with corresponding level) to fetch neighbours
            uint32_t level = current_cell / (NERF_GRIDVOLUME());
            uint32_t pos_idx = current_cell % (NERF_GRIDVOLUME());

            // Sample accepted only if at requested level, statisfying density threshold and not already selected!
            if (!get_bitfield_at(pos_idx, level, m_selection_grid_bitfield.data())) {                           //ERRORE Access violation reading location 0x0000000000063E23
                if(current_density >= density_threshold){                   //serve current_density + alta
                if (level == m_growing_level) {
                // Test whether the new sample touches the boundary, if yes then upscale!
                if (is_boundary(pos_idx)) {
                    std::cout << "UPSAMPLING" << std::endl;
                    upscale_selection(m_growing_level);
                    // Don´t forget to also upscale the current cell!
                    current_cell = get_upper_cell_idx(current_cell, m_growing_level);
                    level = current_cell / (NERF_GRIDVOLUME());
                    pos_idx = current_cell % (NERF_GRIDVOLUME());
                }

                // Invert morton coordinates to get xyz
                uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
                uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
                uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
                // Add possible neighbours
                add_neighbours(m_growing_queue, x, y, z, level);

                // Mark the current cell
                Eigen::Vector3f cell_pos = get_cell_pos(x, y, z, level);
                m_selection_points.push_back(cell_pos);
                m_selection_cell_idx.push_back(current_cell);
                set_bitfield_at(pos_idx, level, true, m_selection_grid_bitfield.data());
                //std::cout << "m_selection_cell_idx: " << current_cell << std::endl;
            }
                }
            }
            i++;
        }
 
        //SI POTREBBE AGGIUNGERE QUI UN CONTROLLO SUI PUNTI DUPLICATI POST GROWING
		//aggiungere tutto ad un set e poi assegnare il contenuto a m_selection_points, 
        //ma bisognerebbe anche sistemare m_selection_cell_idx e m_selection_grid_bitfield

        //Si prosegue solo con il Grow Far Button
        if(ed_flag){
            equidistant_points(min_ed_points_threshold);
        }    
    }        
    // std::cout << "Selected " << m_selection_points.size() << " points overall" << std::endl;
}

bool not_zero_coordinate(Eigen::Vector3f point_to_check) {
    //std::cout << "point_to_check: " << point_to_check << std::endl;
    if (point_to_check == Eigen::Vector3f(0.0f, 0.0f, 0.0f)) {
        std::cout << " Zero Coordinate Point discarded------------------------------------------------------------------------------- " << std::endl;
        return false;
    }
    else {
        return true;
    }
}

//DA FIXARE, NON PRENDE SEMPRE PUNTI DISTANTI IN TERMINI DI COORDINATE######################################################################
//Seleziona in modo uniforme solo alcuni punti superficiali distanti; si ferma al raggiungimento della soglia minima
void RegionGrowing::equidistant_points(int min_ed_points_threshold) {
    std::cout << "PRE m_selection_points size: "<< m_selection_points.size() << std::endl;
    //Vettori temporanei 
    std::vector<Eigen::Vector3f> m_temp_points;
    std::vector<uint32_t> m_temp_idx;

    //Ogni quanti punti bisogna salvarne 1 (per prendere punti distanti in modo uniforme)
    int interval = static_cast<int>(std::round(static_cast<double>(m_selection_points.size()) /  min_ed_points_threshold));
    int count = 0;                                                                                  //counter per scorrere l'array
    if (interval == 0){
         std::cout << "RegionGrowing::equidistant_points() failed: Not enough superficial points selected. Try with a higher growing level."<< std::endl;
        return;
    }

    selection_map selection_mapObj;

    for (int i = 0; i < m_selection_points.size() && m_temp_points.size() < max_ed_points_limit; i++) {
        if ((count % interval == 0) && (not_zero_coordinate(m_selection_points[i]))) {
            m_temp_points.push_back(m_selection_points[i]);
            m_temp_idx.push_back(m_selection_cell_idx[i]);
            //Aggiornamento Mappa utilizzata dallo SPLIT Button
            selection_mapObj.add_to_privateMap(m_selection_cell_idx[i], m_selection_points[i]);        
            //vstd::cout << "Growing point added: "<< i << " with id: " << id << std::endl;
        }
        count++;
    }

    // Sostituisce i vecchi vettori con quelli aggiornati
    m_selection_points = m_temp_points;
    m_selection_cell_idx = m_temp_idx;
    std::cout << "POST m_selection_points size: "<< m_selection_points.size() << std::endl;
}  

//DA FIXARE, NON PRENDE SEMPRE PUNTI DISTANTI IN TERMINI DI COORDINATE######################################################################
//Seleziona in modo uniforme solo alcuni punti superficiali distanti; intervallo scelto dall'utente; continua finchè non supera la soglia minima
void RegionGrowing::equidistant_points(int min_ed_points_threshold, int interval) {
    std::cout << "PRE m_selection_points size: "<< m_selection_points.size() << std::endl;
    //Vettori temporanei 
    std::vector<Eigen::Vector3f> m_temp_points;
    std::vector<uint32_t> m_temp_idx;
    int count = 0;                                                                              //counter per scorrere l'array
    selection_map selection_mapObj;

    for (int i = 0; i < m_selection_points.size() && m_temp_points.size() < max_ed_points_limit; i++) {
        if (count % interval == 0) {
            m_temp_points.push_back(m_selection_points[i]);
            m_temp_idx.push_back(m_selection_cell_idx[i]);
            selection_mapObj.add_to_privateMap(m_selection_cell_idx[i], m_selection_points[i]);
            //std::cout << "Growing point added A: "<< i << std::endl;
        }
        count++;
    }
  
    int interval2 = 0;
    int remaining_ud_points = min_ed_points_threshold - m_temp_points.size();
    if ( remaining_ud_points > 0) {
        interval2 = static_cast<int>(m_selection_points.size() / remaining_ud_points);
        if (interval == 0){
            std::cout << "RegionGrowing::equidistant_points() failed: Not enough superficial points selected. Try with a higher growing level."<< std::endl;
            return;
        }
        for (int i = 0; i < m_selection_points.size() && remaining_ud_points > 0 && m_temp_points.size() < max_ed_points_limit; i++) {
            if (count % interval2 == 0) {
                auto it = std::find(m_temp_points.begin(), m_temp_points.end(), m_selection_points[i]); //restituisce puntatore a ultimo elemento, se non trova l'oggetto
                if (it == m_temp_points.end()) {                            //se l'oggetto non è presente, viene aggiunto 
                    m_temp_points.push_back(m_selection_points[i]);
                    m_temp_idx.push_back(m_selection_cell_idx[i]);
                    //INSERIMENTO NELLA MAPPA NON TESTATO###########
                    selection_mapObj.add_to_privateMap(m_selection_cell_idx[i], m_selection_points[i]); 
                    remaining_ud_points--;
                    //std::cout << "Growing point added B: "<< i << std::endl;
                }
                
            }
            count++;
        }
    }
    
    // Sostituisci i vecchi vettori con quelli aggiornati
    m_selection_points = m_temp_points;
    m_selection_cell_idx = m_temp_idx;
    std::cout << "POST m_selection_points size: "<< m_selection_points.size() << std::endl;
}      

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

nlohmann::json RegionGrowing::to_json() {
        nlohmann::json j;

        j["selection_grid_bitfield"] = m_selection_grid_bitfield;
        j["selection_points"] = m_selection_points;
        j["selection_cell_idx"] = m_selection_cell_idx;
        j["density_grid_host"] = m_density_grid_host;
        // TODO: support saving of queue
        // to_json_queue<uint32_t>(j["growing_queue"], m_growing_queue);

        return j;
    }

void RegionGrowing::load_json(nlohmann::json& j) {
    std::cout << "most" << std::endl;
    from_json(j["selection_grid_bitfield"], m_selection_grid_bitfield);
    from_json(j["selection_points"], m_selection_points);
    from_json(j["selection_cell_idx"], m_selection_cell_idx);
    from_json(j["density_grid_host"], m_density_grid_host);
    // TODO: support reloading of the queue
    // from_json_queue<uint32_t>(j["growing_queue"], m_growing_queue);
}

NGP_NAMESPACE_END
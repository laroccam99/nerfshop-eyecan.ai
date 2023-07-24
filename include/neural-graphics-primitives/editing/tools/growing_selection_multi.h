#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/editing/datastructures/cage.h>
#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/editing/tools/fast_quadric.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/editing/datastructures/tet_mesh.h>
#include <neural-graphics-primitives/editing/tools/progressive_hulls.h>
#include <neural-graphics-primitives/editing/tools/region_growing_multi.h>
#include <neural-graphics-primitives/editing/tools/mm_operations.h>

#include <tiny-cuda-nn/common.h>

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>

#include <vector>
#include <queue>

#include <json/json.hpp>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

static constexpr const char SCREEN_SELECTION_KEY = 'B';
static constexpr const uint32_t DEBUG_CUBEMAP_WIDTH = 16;
static constexpr const uint32_t DEBUG_ENVMAP_WIDTH = 256;
static constexpr const uint32_t DEBUG_ENVMAP_HEIGHT = 128;

enum class ESelectionMode : int {
    PixelWise,
    Scribble
};

static constexpr const char* SelectionModeStr = "Pixelwise\0 Scribble\0\0";

enum class EPcRenderMode : int {
	Off,
	UniformColor,
	Labels,
};
static constexpr const char* PcRenderModeStr = "Off\0UniformColor\0Labels\0\0";

enum class EManipulationTarget : int {
	CageVerts,
	CursorCoords
};
static constexpr const char* targetStrings = "Cage Vertices\0Cursor Coordinate System";

enum class ESelectionRenderMode : int {
	Off,
	ScreenSelection,
    Projection,
	RegionGrowing,
	SelectionMesh,
	ProxyMesh,
	TetMesh,
};
static constexpr const char* SelectionRenderModeStr = "Off\0Screen Scribbles\0Projected Scribbles\0Grown Region\0Fine Cage\0Coarse Cage\0Tetrahedral Mesh\0\0";

enum class EDecimationAlgorithm : int {
	ShortestEdge,
    ProgressiveHullsQuadratic,
    ProgressiveHullsLinear,
};

static constexpr const char* DecimationAlgorithmStr = "Shortest Edge\0Progressive Hulls Quadratic\0Progressive Hulls Linear\0\0";

enum class ERadianceAlgorithm : int {
    ColorOnly,
    SH,
};

static constexpr const char* RadianceAlgorithmStr = "Color Only\0Spherical Harmonics\0\0";

enum class EProjectionThresholds : int {
    Low,
    Intermediate,
    High,
};
static constexpr const char* ProjectionThresholdsStr[3] = { "Low", "Intermediate", "High"};
static constexpr const float ProjectionThresholdsVal[3] = {1e-3f, 1e-1f, 1.f};

typedef float float_t;
typedef Eigen::Vector3f point_t;
typedef Eigen::Matrix3f matrix3_t;
typedef Eigen::Matrix4f matrix4_t;

struct CageEdition {
    std::vector<uint32_t> selected_vertices;
    point_t selection_barycenter;
    matrix3_t selection_rotation;
	point_t selection_scaling;
};

static Eigen::Vector3f DEBUG_COLORS_LUT[6] = {
    Eigen::Vector3f(1.0f, 0.0f, 0.0f),
    Eigen::Vector3f(0.0f, 1.0f, 0.0f),
    Eigen::Vector3f(0.0f, 0.0f, 1.0f),
    Eigen::Vector3f(1.0f, 0.75f, 0.0f),
    Eigen::Vector3f(0.9f, 0.55f, 0.9f),
    Eigen::Vector3f(0.9f, 0.9f, 0.9f),
};

struct GrowingSelectionMulti {

    // Fine mesh extracted from the region-grown points
    Mesh<float_t, point_t> selection_mesh;

    // Proxy cage obtained with decimation
    Cage<float_t, point_t> proxy_cage;
    bool display_in_tet = false;
    bool preserve_surface_mesh = true;
    float ideal_tet_edge_length;
    std::shared_ptr<TetMesh<float_t, point_t>> tet_interpolation_mesh;

	bool multi_copy = false;
	bool multi_bypass = false;

	EManipulationTarget multi_target = EManipulationTarget::CageVerts;

    float transmittance_threshold = 1e-1f;
    float off_surface_projection = 0.01f;

	Eigen::Vector3f multi_plane_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f multi_plane_dir = Eigen::Vector3f::Zero();
	Eigen::Vector3f multi_plane_dir1;
	Eigen::Vector3f multi_plane_dir2;

    int proxy_size = 100;

    ESelectionRenderMode render_mode = ESelectionRenderMode::ScreenSelection;

    GrowingSelectionMulti(
        BoundingBox aabb,
        cudaStream_t stream, 
        const std::shared_ptr<NerfNetwork<precision_t>> nerf_network, 
        const tcnn::GPUMemory<float>& density_grid, 
        const tcnn::GPUMemory<uint8_t>& density_grid_bitfield,
        const float cone_angle_constant,
        const ENerfActivation rgb_activation,
        const ENerfActivation density_activation,
        const Eigen::Vector3f light_dir,
        const std::string default_envmap_path,
        const uint32_t max_cascade
    );

    GrowingSelectionMulti(
        nlohmann::json operator_json,
        BoundingBox aabb,
        cudaStream_t stream, 
        const std::shared_ptr<NerfNetwork<precision_t>> nerf_network, 
        const tcnn::GPUMemory<float>& density_grid, 
        const tcnn::GPUMemory<uint8_t>& density_grid_bitfield,
        const float cone_angle_constant,
        const ENerfActivation rgb_activation,
        const ENerfActivation density_activation,
        const Eigen::Vector3f light_dir,
        const std::string default_envmap_path,
        const uint32_t max_cascade
    );

	void multi_find_plane();

	bool multi_imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center, bool& auto_clean);

    bool multi_visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time);

    void multi_draw_gl(
        const Eigen::Vector2i& resolution, 
        const Eigen::Vector2f& focal_length, 
        const Eigen::Matrix<float, 3, 4>& camera_matrix, 
        const Eigen::Vector2f& screen_center
    );

    void multi_to_json(nlohmann::json& j);

	float multi_plane_offset = 0.0f;

	bool multi_use_morphological = true;

	void multi_deform_proxy_from_file(std::string deformed_file);

	void multi_proxy_mesh_from_file(std::string orig_file);

	bool multi_refine_cage = false;

	void multi_set_proxy_mesh(std::vector<point_t>& points, std::vector<uint32_t>& indices);

private:

    // Selection specifics
    ESelectionMode multi_selection_mode = ESelectionMode::Scribble;
    std::vector<Eigen::Vector2i> multi_selected_pixels;
    std::vector<ImVec2> multi_selected_pixels_imgui;
    Eigen::Vector2i multi_last_selected_pixel = Eigen::Vector2i(-1, -1);

    // Necessary for the kernel parts
    const BoundingBox multi_aabb;
    const std::shared_ptr<NerfNetwork<precision_t>> multi_nerf_network;
    const tcnn::GPUMemory<float>& multi_density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
    const tcnn::GPUMemory<uint8_t>& multi_density_grid_bitfield;
    const float multi_cone_angle_constant = 1.f/256.f;
    const ENerfActivation multi_rgb_activation;
    const ENerfActivation multi_density_activation;
    const Eigen::Vector3f multi_light_dir;

    cudaStream_t multi_stream;

    // Cage rect-based screen-space selection
    bool selected_cage = false;
    bool currently_selecting_cage = false;
    ImVec2 mouse_clicked_selecting_cage;
    ImVec2 mouse_released_selecting_cage;
    ImGuizmo::MODE multi_gizmo_mode = ImGuizmo::LOCAL;
	ImGuizmo::OPERATION multi_gizmo_op = ImGuizmo::TRANSLATE;


    CageEdition cage_edition = {};

    EPcRenderMode multi_pc_render_mode = EPcRenderMode::Labels;
    int multi_pc_render_max_level = NERF_CASCADES();
    bool multi_visualize_max_level_cube = false;
    bool multi_automatic_max_level = true;
    uint32_t multi_max_cascade;

    // Projected pixels
    std::vector<Eigen::Vector3f> multi_projected_pixels;
    std::vector<uint8_t> multi_projected_labels;
    std::vector<uint32_t> multi_projected_cell_idx;
    // std::vector<FeatureVector> multi_projected_features;

	float multi_select_radius = 8;
	bool multi_rigid_editing = false;
 
    // Region-grown points (+ MM operators)
    std::vector<Eigen::Vector3f> multi_selection_points;
    std::vector<uint8_t> multi_selection_labels;
    std::vector<uint32_t> multi_selection_cell_idx;
    std::vector<uint8_t> multi_selection_grid_bitfield;

    // Region-growing
    int multi_growing_steps = 10000;
    int multi_growing_level = 0;
    float multi_density_threshold = 0.01f;
    ERegionGrowingMode multi_region_growing_mode = ERegionGrowingMode::Manual;
    RegionGrowingMulti multi_region_growing;

    // Morphological operations
    std::shared_ptr<MMOperations> multi_MM_operations;
    bool multi_performed_closing = false;

    EDecimationAlgorithm multi_decimation_algorithm = EDecimationAlgorithm::ProgressiveHullsQuadratic;
    
    ProgressiveHullsParams multi_progressive_hulls_params;

    ERadianceAlgorithm multi_radiance_algorithm = ERadianceAlgorithm::SH;
    int multi_n_hemisphere_samples = 10;
    int multi_hemisphere_width = 10;
    int multi_projection_threshold_simple = (int)EProjectionThresholds::Intermediate;
    float multi_transmittance_threshold_boundary = 1e-3f;
    float multi_brush_color[3] = {0.0f, 1.0f, 0.0};
    float multi_cage_color[3] = {0.9f, 0.9f, 0.98f};
    GLuint multi_debug_cubemap_textures[6] = {0, 0, 0, 0, 0, 0};
    bool multi_rotate_debug_cubemap = true;
    bool multi_initial_debug_cubemap = false;
    int multi_debug_ray_idx = 0;
    Eigen::Vector3f multi_debug_rotation_normal = Eigen::Vector3f(1.f, 0.f, 0.f);
    float multi_t_min_boundary = 0.05f; // This ensures that we don't get interferences from within the boundary
    tcnn::GPUMemory<float> multi_envmap;
    Eigen::Vector2i multi_envmap_resolution = Eigen::Vector2i::Constant(0.0f);
    std::string multi_default_envmap_path = "";
    GLuint multi_debug_envmap_texture = 0;

	float multi_plane_radius = 0.1f;

    // Automatically update the tet when a manipulation is performed
    bool multi_update_tet_manipulation = true;

    std::vector<Eigen::Vector3f> multi_debug_points;
    std::vector<Eigen::Vector3f> multi_debug_colors;

    // Poisson editing
    struct PoissonEditing {
        int sh_sampling_width = 10;
        GLuint sh_cubemap_textures[6] = {0, 0, 0, 0, 0, 0};
        float sh_sum_weights_threshold = 1e-4f;
        float inside_contribution = 1.f;
        float mvc_gamma = 1.0f;
    } multi_poisson_editing;

    bool multi_correct_direction = true;

    void multi_clear();

    // ------------------------
    // Screen-space selection
    // ------------------------
	inline bool multi_is_near_mouse(const ImVec2& p);

    inline bool multi_is_inside_rect(const ImVec2& p);

	void multi_select_scribbling(const Eigen::Matrix<float, 4, 4>& world2proj);

    void multi_select_cage_rect(const Eigen::Matrix<float, 4, 4>& world2proj);

    void multi_reset_cage_selection();

    void multi_delete_selected_projection();

    void multi_delete_selected_growing();

    void multi_color_selection();

    // ------------------------
    // Region growing
    // ------------------------

    // Initialize region growing
    void multi_reset_growing();
    
    void multi_upscale_growing();

    // Grow region (by user-selected steps)
    void multi_grow_region();

    // ------------------------
    // Morphological Operators
    // ------------------------
	
    // MM dilation
    void multi_dilate();
	
    // MM_erosion
    void multi_erode();

    // ------------------------
    // Proxy mesh processing
    // ------------------------

    // Extract the fine mesh from the voxelized region selection (using marching cubes)
    void multi_extract_fine_mesh();

    // Decimate fine mesh with linear bounding constraint
    void multi_compute_proxy_mesh();

    // Decimate all fine meshes  
    void multi_compute_all_proxy_mesh();
    
    // Not used in practice
    void multi_fix_fine_mesh();

    // Fix proxy mesh with MeshFix
    void multi_fix_proxy_mesh();

    // Fix all proxy meshes with MeshFix
    void multi_big_fix_proxy_mesh();

    // DEBUG: export the proxy mesh as a file
    void multi_export_proxy_mesh();

    // ------------------------
    // Proxy mesh processing
    // ------------------------

    // Extract mesh with TetGen
    void multi_extract_tet_mesh();

	void multi_force_cage();

    // Initialize the tet mvc coordinates with the pre-computed cage
    void multi_initialize_mvc();

    // Update the new position with MVC coordinates and update tet_lut
    void multi_update_tet_mesh();

    // ------------------------
    // Poisson correction
    // ------------------------

    // Store view-dependent color of NeRF at the boundary proxies of the cage as SH
    void multi_compute_poisson_boundary(const bool is_inside);

    // DEBUG: display incoming radiance SH as cube maps
    void multi_generate_poisson_cube_map();

    // TODO: handle rotation
    // Interpolate poisson values at the boundary using MVC coordinates
    void multi_interpolate_poisson_boundary();
};

void draw_selection_gl(const std::vector<Eigen::Vector3f>& points, const std::vector<uint8_t>& labels, const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center, const int pc_render_mode, const int max_label);

void draw_debug_gl(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector3f>& colors, const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center);

NGP_NAMESPACE_END


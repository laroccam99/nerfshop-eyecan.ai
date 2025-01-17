#include <neural-graphics-primitives/common_gl.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common_nerf.h>
#include <neural-graphics-primitives/editing/tools/growing_selection.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <neural-graphics-primitives/editing/tools/default_mm_operations.h>
#include <neural-graphics-primitives/editing/tools/correct_mm_operations.h>
#include <neural-graphics-primitives/editing/tools/sh_utils.h>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/editing/tools/visualization_utils.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/helper_selected_pixels.h>


#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/decimate.h>
#include <igl/writeOFF.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <igl/per_vertex_normals.h>
#include <random>

#include "meshfix.h"

#include <cstdlib>

#ifdef NGP_GUI
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GL/glu.h>
#  include <GLFW/glfw3.h>
#endif

#include <imguizmo/ImGuizmo.h>

NGP_NAMESPACE_BEGIN
const int max_number_of_iterations = 1;
extern int num_of_iterations = 0;
//std::set<GrowingSelection> GS_Set;		//inutilizzato


GrowingSelection::GrowingSelection(
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
    ) : 
        m_aabb{aabb},
        m_stream{stream}, 
        m_nerf_network{nerf_network},
        m_density_grid{density_grid},
        m_density_grid_bitfield{density_grid_bitfield},
        m_cone_angle_constant{cone_angle_constant},
        m_rgb_activation{rgb_activation},
        m_density_activation{density_activation},
        m_light_dir{light_dir},
		m_default_envmap_path{default_envmap_path},
		m_region_growing(density_grid, density_grid_bitfield, max_cascade),
		m_MM_operations{new CorrectMMOperations()},
		m_max_cascade{max_cascade} {

	// For each face:
	for (int f = 0; f < 6; f++) {
		// Create a OpenGL texture identifier
		glGenTextures(1, &m_debug_cubemap_textures[f]);
		glGenTextures(1, &m_poisson_editing.sh_cubemap_textures[f]);
	}
	// For the envmap:
	glGenTextures(1, &m_debug_envmap_texture);

	// Initialize mm_operator
	// std::cout << (bool)m_MM_operations << std::endl;
}

GrowingSelection::GrowingSelection(
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
) : GrowingSelection(aabb, stream, nerf_network, density_grid, density_grid_bitfield, cone_angle_constant, rgb_activation, density_activation, light_dir, default_envmap_path, max_cascade) {

	from_json(operator_json["projected_pixels"], m_projected_pixels);
	from_json(operator_json["projected_labels"], m_projected_labels);
	from_json(operator_json["projected_cell_idx"], m_projected_cell_idx);
	// from_json(operator_json["projected_features"], m_projected_features);

	from_json(operator_json["selection_points"], m_selection_points);
	from_json(operator_json["selection_labels"], m_selection_labels);
	from_json(operator_json["selection_cell_idx"], m_selection_cell_idx);
	from_json(operator_json["m_selection_grid_bitfield"], m_selection_grid_bitfield);

	m_growing_level = operator_json["growing_level"];
	m_region_growing.load_json(operator_json["region_growing"]);

	from_json(operator_json["selection_mesh"], selection_mesh);
	from_json(operator_json["proxy_cage"], proxy_cage);

	if (operator_json.contains("interpolation_mesh")) {
		tet_interpolation_mesh = std::make_shared<TetMesh<float_t, point_t>>(operator_json["interpolation_mesh"]);
		tet_interpolation_mesh->build_tet_grid(m_stream);
	}
	
}

bool GrowingSelection::imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center, bool& auto_clean) {
    bool grid_edit = false;

	if (ImGui::Button("A1-PROJECT")) {
		project_selection_pixels(m_selected_pixels, resolution, focal_length, camera_matrix, screen_center, m_stream);
		if (m_projected_cell_idx.size() > 0)
			render_mode = ESelectionRenderMode::Projection;
	}
	ImGui::SameLine(); 
	//Funziona solo post-scribbling: lascia solo 1 punto tra quelli proiettati, permette growing, costruzione cage e spostamento
	if(ImGui::Button("RemoveBut1")) {
		std::cout << "############################## RemoveBut1 Button " << std::endl;
		int random_index = random_index_in_selected_pixels();
		remove_but_one(random_index);		
	}
	bool growing_allowed = m_projected_cell_idx.size() > 0 || m_selection_points.size() > 0;
	if (growing_allowed) {
		//ImGui::SameLine(); 
		if(ImGui::Button("GROW_REGION")) {
			grow_region(false, get_m_growing_steps());
			render_mode = ESelectionRenderMode::RegionGrowing;
		}
		ImGui::SameLine(); 

		if(ImGui::Button("A1_GROW FAR")) {
			grow_region(true, get_m_grow_far_steps());
			render_mode = ESelectionRenderMode::RegionGrowing;
		}
		ImGui::SameLine(); //SI POTREBBE CONTROLLARE LA PRESENZA DI UN FLAG PER PASSARE A QUESTA MODALITà
		if(ImGui::Button("GROW&CAGE")) {
			grow_and_cage();
		}
	}	

	bool proxy_allowed = true;// m_selection_points.size() > 0;
	if (proxy_allowed) {
		if (m_refine_cage) {
			if (m_selection_grid_bitfield.size() > 0)
			{
				if (ImGui::Button("EXTRACT CAGE")) {
					ImGui::Text("Please wait, extracting cage...");
					compute_proxy_mesh();
					fix_proxy_mesh();
				}
			}
			if (proxy_cage.vertices.size() > 0) {
				ImGui::SameLine();
				if (ImGui::Button("COMPUTE PROXY")) {
					ImGui::Text("Please wait, computing proxy...");
					fix_proxy_mesh();
					update_tet_mesh();
					interpolate_poisson_boundary();
				}
			}
		} else {
			if (m_selection_grid_bitfield.size() > 0)
			{
				ImGui::SameLine();
				// Will extract and clean the proxy directly, then compute the tet mesh and extract it as well

				if (ImGui::Button("COMPUTE1")) {
					ImGui::Text("Please wait, computing proxy...");
					fix_proxy_mesh();
					update_tet_mesh();
					interpolate_poisson_boundary();
				}
			}
		}
	}

	bool clear_selection_allowed = m_selected_pixels.size() > 0 || m_selection_cell_idx.size() > 0 || proxy_cage.vertices.size() > 0;
	if (clear_selection_allowed && imgui_colored_button2("Clear selection", 0.f)) {
		clear();
		//set_apply_all_edits_flag(false);						//aggiunto per testing
		apply_all_edits_flag = false;
		do_it_once = false;
	}
	bool reset_growing_allowed = m_selection_cell_idx.size() > 0;
	if (reset_growing_allowed) {
		ImGui::SameLine();
		if (imgui_colored_button2("Reset growing", 0.1f)) {
			reset_growing();
			//set_apply_all_edits_flag(false);						//aggiunto per testing
			apply_all_edits_flag = false;
			do_it_once = false;
			render_mode = ESelectionRenderMode::RegionGrowing;
		}
	}
	//DA RIMUOVERE: Button UTILE SOLO PER TESTING#############################################à
	ImGui::SameLine();
	if (m_selection_grid_bitfield.size() > 0) {
		if (imgui_colored_button2("Apply 1 edit", 0.2f)) {
			set_apply_all_edits_flag(true);
			//manca la parte che rende set_m_apply_poisson(true)
		}
	}
	if (tet_interpolation_mesh) {
		if (imgui_colored_button2("Vanish!", 0.4f)) {
			tet_interpolation_mesh->vanish(m_density_grid.data(), m_density_grid_bitfield.data(), m_stream);
			//for (auto& v : proxy_cage.vertices) {
			//	// Then, translate
			//	v.y() -= 1000;
			//}
			//cage_edition.selection_barycenter.y() -= 1000;
			//update_tet_mesh();
			auto_clean = false;
		}

		// Guizmo control
		ImGui::Checkbox("Copy", &m_copy);

		ImGui::Combo("Target", (int*)&m_target, targetStrings);

		if (ImGui::RadioButton("Translate", m_gizmo_op == ImGuizmo::TRANSLATE))
			m_gizmo_op = ImGuizmo::TRANSLATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", m_gizmo_op == ImGuizmo::ROTATE))
			m_gizmo_op = ImGuizmo::ROTATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Scale", m_gizmo_op == ImGuizmo::SCALE))
			m_gizmo_op = ImGuizmo::SCALE;
		//ImGui::SameLine();
		//if (ImGui::RadioButton("Local", m_gizmo_mode == ImGuizmo::LOCAL))
		//    m_gizmo_mode = ImGuizmo::LOCAL;
		//ImGui::SameLine();
		//if (ImGui::RadioButton("World", m_gizmo_mode == ImGuizmo::WORLD))
		//    m_gizmo_mode = ImGuizmo::WORLD;

	}

	ImGui::Separator();

	if (render_mode == ESelectionRenderMode::ScreenSelection) {
		const char* elem_name = ProjectionThresholdsStr[m_projection_threshold_simple];
		if (ImGui::SliderInt("Projection Threshold", &m_projection_threshold_simple, 0, 2, elem_name)) {
			transmittance_threshold = ProjectionThresholdsVal[m_projection_threshold_simple];
		}
	}

	if (proxy_allowed) {
		if (render_mode == ESelectionRenderMode::RegionGrowing || render_mode == ESelectionRenderMode::Projection)
			ImGui::SliderInt("Growing steps", &m_growing_steps, 0, 100000, "%d", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderInt("Cage size", &proxy_size, 0, 10000, "%d", ImGuiSliderFlags_Logarithmic);
		ImGui::Checkbox("Cage refinement", &m_refine_cage);
	}

	ImGui::SetNextItemOpen(false, ImGuiCond_Once);
	if (ImGui::TreeNode("Advanced")) {

		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Selection")) {
			if(ImGui::Button("Clear selection")) {
				clear();
			}
			ImGui::Combo("Selection Mode", (int*)&(m_selection_mode), SelectionModeStr);
			ImGui::Checkbox("Automatic max level", &m_automatic_max_level);
			ImGui::Checkbox("Visualize max level cube", &m_visualize_max_level_cube);
			if (!m_automatic_max_level) {
				ImGui::SliderInt("Growing levels", &m_growing_level, 0, NERF_CASCADES() - 1);
			}
			ImGui::TreePop();
		}

		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Projection")) {
			ImGui::SliderFloat("Transmittance threshold", &transmittance_threshold, 0.0f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
			if (ImGui::Button("Project selection")) {
				project_selection_pixels(m_selected_pixels, resolution, focal_length, camera_matrix, screen_center, m_stream);
				render_mode = ESelectionRenderMode::Projection;
			}
			ImGui::TreePop();
		}

		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Region growing")) {
			ImGui::SliderFloat("Density threshold", &m_density_threshold, 0.0f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
			ImGui::SliderInt("Growing steps", &m_growing_steps, 0, 100000, "%d", ImGuiSliderFlags_Logarithmic);
			if (ImGui::Button("Reset growing")) {
				reset_growing();
				render_mode = ESelectionRenderMode::RegionGrowing;
			}
			ImGui::SameLine();
			if (ImGui::Button("Grow region")) {
				grow_region(false, get_m_growing_steps());
				render_mode = ESelectionRenderMode::RegionGrowing;
			}
			ImGui::Combo("Growing Mode", (int*)&(m_region_growing_mode), RegionGrowingModeStr);
			if (ImGui::Button("Upscale")) {
				upscale_growing();
			}
			ImGui::TreePop();
		}

		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Morphological operations")) {
			if (ImGui::Button("Dilate")) {
				dilate();
				render_mode = ESelectionRenderMode::RegionGrowing;
			}
			ImGui::SameLine();
			if (ImGui::Button("Erode")) {
				erode();
				render_mode = ESelectionRenderMode::RegionGrowing;
			}
			ImGui::SameLine();
			if (ImGui::Button("Dilate & Erode")) {
				dilate();
				erode();
				m_performed_closing = true;
				render_mode = ESelectionRenderMode::RegionGrowing;
			}
			m_MM_operations->imgui(resolution, focal_length, camera_matrix, screen_center);
			ImGui::TreePop();
		}
		
		//ImGui::SliderFloat("Off surface projection", &off_surface_projection, 1e-4, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Proxy Generation")) {
			ImGui::Checkbox("Use Morphological Ops", &m_use_morphological);
			if (ImGui::Button("Extract initial mesh")) {
				extract_fine_mesh();
				render_mode = ESelectionRenderMode::SelectionMesh;
			}
			ImGui::SameLine();
			if (ImGui::Button("Fix initial mesh")) {
				fix_fine_mesh();
			}
			ImGui::SliderInt("Proxy target size", &proxy_size, 0, 10000, "%d", ImGuiSliderFlags_Logarithmic);
			ImGui::Combo("DecimationAlgorithm", (int*)&m_decimation_algorithm, DecimationAlgorithmStr);
			if (m_decimation_algorithm == EDecimationAlgorithm::ProgressiveHullsQuadratic || m_decimation_algorithm == EDecimationAlgorithm::ProgressiveHullsLinear) {
				ImGui::SliderFloat("QEM contribution", &m_progressive_hulls_params.w, 0.0, 1.0, "%.4f", ImGuiSliderFlags_Logarithmic);	
				ImGui::Checkbox("Compactness", &m_progressive_hulls_params.compactness_test);
				ImGui::SameLine();
				ImGui::Checkbox("Normal (NO)", &m_progressive_hulls_params.normal_test);
				ImGui::SameLine();
				ImGui::Checkbox("Valence", &m_progressive_hulls_params.valence_test);
				if (m_progressive_hulls_params.compactness_test) {
					ImGui::SliderFloat("Compactness Threshold", &m_progressive_hulls_params.compactness_threshold, 0.0, 1.0);	
				}
				if (m_progressive_hulls_params.normal_test) {
					ImGui::SliderFloat("Normal Threshold", &m_progressive_hulls_params.normal_threshold, 0.0, M_PI/2);	
				}
				if (m_progressive_hulls_params.valence_test) {
					ImGui::SliderInt("Max valence", &m_progressive_hulls_params.max_valence, 6, 20);	
				}
				ImGui::Checkbox("Presimplify", &m_progressive_hulls_params.presimplify);
				if (m_progressive_hulls_params.presimplify) {
					ImGui::SliderFloat("Presimplification Ratio", &m_progressive_hulls_params.presimplification_ratio, 0, 1, "%.4f", ImGuiSliderFlags_Logarithmic);
				}
			}
			if (ImGui::Button("Compute proxy")) {
				compute_proxy_mesh();
			}
			ImGui::SameLine();
			if (ImGui::Button("Fix proxy")) {
				fix_proxy_mesh();
			}
			ImGui::SameLine();
			if (ImGui::Button("Export Proxy Mesh")) {
				export_proxy_mesh();
			}
			ImGui::TreePop();
		}

		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Tet extraction")) {
			ImGui::Checkbox("Preserve surface mesh", &preserve_surface_mesh);
			ImGui::SameLine();
			ImGui::Checkbox("Display inside", &display_in_tet);
			ImGui::Checkbox("Auto update", &m_update_tet_manipulation);
			ImGui::SliderFloat("Ideal tet length", &ideal_tet_edge_length, 0.0f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
			if (ImGui::Button("Force Cage")) {
				force_cage();
			}
			if (ImGui::Button("Extract tet")) {
				extract_tet_mesh();
				initialize_mvc();
			}
			ImGui::SameLine();
			if (ImGui::Button("Update TetMesh")) {
				update_tet_mesh();
				grid_edit = true;
			}
			if (tet_interpolation_mesh) {
				ImGui::Text("Max tet lookup: %d", tet_interpolation_mesh->max_tet_lookup);
			}
			if (ImGui::Checkbox("Correct direction", &m_correct_direction)) {
				if (!m_correct_direction && tet_interpolation_mesh) {
					tet_interpolation_mesh->local_rotations_gpu.resize(0);
				}
				update_tet_mesh();
			}
			ImGui::TreePop();
		}

		ImGui::SetNextItemOpen(false, ImGuiCond_Once);
		if (ImGui::TreeNode("Poisson editing")) {
			ImGui::SliderFloat("Plane Radius", &m_plane_radius, 0.1f, 1.0f);
			ImGui::SliderFloat("Plane Offset", &m_plane_offset, -1.0f, 1.0f);
			if(ImGui::SliderFloat("MVC gamma", &m_poisson_editing.mvc_gamma, 1.f, 2.f)) {
				if (!tet_interpolation_mesh && tet_interpolation_mesh->vertices.size() > 0) {
					// std::cout << "Updating MVC to gamma = " << m_poisson_editing.mvc_gamma << std::endl;
					proxy_cage.compute_mvc(tet_interpolation_mesh->original_vertices, tet_interpolation_mesh->gamma_coordinates, tet_interpolation_mesh->labels, true, m_poisson_editing.mvc_gamma);
					interpolate_poisson_boundary();
				}
			}
			ImGui::SliderInt("SH sampling width", &m_poisson_editing.sh_sampling_width, 1, 20);
			if (ImGui::SliderFloat("weights threshold", &m_poisson_editing.sh_sum_weights_threshold, 1e-6f, 1e-2f, "%.6f", ImGuiSliderFlags_Logarithmic)) {
				interpolate_poisson_boundary();
			}
			if (ImGui::SliderFloat("inside contribution", &m_poisson_editing.inside_contribution, 0.f, 1.f)) {
				interpolate_poisson_boundary();
			}
			// if (ImGui::Button("Compute boundary values")) {
			// 	compute_poisson_boundary();
			// }
			ImGui::SameLine();
			if (ImGui::Button("Compute cubemap")) {
				generate_poisson_cube_map();
			}
			if (ImGui::Button("Update MVC")) {
				interpolate_poisson_boundary();
			}
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
			for (int i = 0; i < 6; i++)
			{
				ImGui::PushID(i);
				if (i < 1 || i >3 ) {
					ImGui::Indent(4*DEBUG_CUBEMAP_WIDTH);
				}  
				ImGui::Image((void*)(intptr_t)m_poisson_editing.sh_cubemap_textures[i], ImVec2(4*DEBUG_CUBEMAP_WIDTH, 4*DEBUG_CUBEMAP_WIDTH));
				if (i < 1 || i >3 ) {
					ImGui::Unindent(4*DEBUG_CUBEMAP_WIDTH);
				}  
				ImGui::PopID();
				if (i >= 1 && i < 3) {
					ImGui::SameLine();
				}
			}
			ImGui::PopStyleVar();
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}
    
    
    ImGui::Separator();
    ImGui::Combo("Operator Visualization", (int*)&render_mode, SelectionRenderModeStr);			//da controllare
    // if (render_mode == ESelectionRenderMode::TetMesh) {
    //     ImGui::Combo("TetMeshRenderMode", (int*)&(tet_interpolation_mesh->render_mode), TetMeshRenderModeStr);
    // } else if (render_mode == ESelectionRenderMode::ProxyMesh) {
    //     ImGui::Combo("Cage Render Mode", (int*)&proxy_cage.render_mode, CageRenderModeStr);
    // } else if (render_mode == ESelectionRenderMode::RegionGrowing) {
    //     ImGui::Combo("PcRenderMode", (int*)&m_pc_render_mode, PcRenderModeStr);
	// 	ImGui::SliderInt("Max level", &m_pc_render_max_level, 0, NERF_CASCADES());
    // }
    ImGui::Separator();

    // ImGui::LabelText(std::to_string(m_selected_pixels.size()).c_str(), "Nb selected pixels");
    
    // ImGui::Separator();

	//ImGui::Checkbox("Rigid Editing", &m_rigid_editing);
	//ImGui::SameLine();
	
	//ImGui::Checkbox("Bypass verts", &m_bypass);
	//ImGui::Separator();

	return grid_edit;
}

//Edit GUI
// The function returns a boolean value indicating whether the gizmo was edited or not.
bool GrowingSelection::visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) {
	
	//Record the current time to measure performance.
	auto before = std::chrono::system_clock::now();
	//Get the foreground draw list from ImGui, which will be used for rendering.	
    ImDrawList* list = ImGui::GetForegroundDrawList();

	/*Initialize variables and matrices related to Guizmo visualization and editing, 
	such as the focal length, aspect ratio, and view-to-projection matrix.*/
    // Guizmo visualization and editing
    bool edited_guizmo = false;
    float flx = focal.x();
    float fly = focal.y();
    Eigen::Matrix<float, 4, 4> view2proj_guizmo;
    float zfar = 100.f;
    float znear = 0.1f;
    view2proj_guizmo <<
        fly*2.f/aspect, 0, 0, 0,
        0, -fly*2.f, 0, 0,
        0, 0, (zfar+znear)/(zfar-znear), -(2.f*zfar*znear) / (zfar-znear),
        0, 0, 1, 0;

    ImGuiIO& io = ImGui::GetIO();

	//Set the ImGuizmo rectangle to cover the entire display area.
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

	//Questa porzione di codice non viene attivata, non ho trovato nessun punto in cui viene modificato m_rigid_editing
	if (m_rigid_editing && cage_edition.selected_vertices.size() == 0)
	{
		for (int i = 0; i < proxy_cage.vertices.size(); i++) cage_edition.selected_vertices.push_back(i);

		//Initialize the vertex selection
		cage_edition.selection_barycenter = point_t::Zero();
		if (!m_plane_dir.isZero())
		{
			matrix3_t rot;
			rot.col(0) = m_plane_dir1;
			rot.col(1) = m_plane_dir;
			rot.col(2) = m_plane_dir2;
			cage_edition.selection_rotation = rot;
		}
		else
		{
			cage_edition.selection_rotation = matrix3_t::Identity();
		}
		cage_edition.selection_scaling = point_t::Ones();

		//Compute the barycenter and rotation matrix of the selection
		std::cout << "cage_edition.selected_vertices size: " << cage_edition.selected_vertices.size() << std::endl;
		std::cout << "proxy_cage.vertices size: " << proxy_cage.vertices.size() << std::endl;

		for (const auto selected_vertex: cage_edition.selected_vertices) {
			cage_edition.selection_barycenter += proxy_cage.vertices[selected_vertex];
		}
		cage_edition.selection_barycenter /= cage_edition.selected_vertices.size();

		std::cout << "cage_edition.selection_barycenter m_rigid_editing: " << cage_edition.selection_barycenter << std::endl;

/*
		for (const auto selected_point : m_selection_points) {
			cage_edition.selection_barycenter += selected_point;
		}
		cage_edition.selection_barycenter /= m_selection_points.size();

		std::cout << "cage_edition.selection_barycenter: " << cage_edition.selection_barycenter << std::endl;
*/
	}


	//Create an edit_matrix to store the transformation matrix for the selected vertices.
    Eigen::Matrix4f edit_matrix;
	//point_t guizmo_scale = point_t(1.0f, 1.0f, 1.0f);
	compose_imguizmo_matrix<matrix3_t, point_t, float_t>(edit_matrix, cage_edition.selection_rotation, cage_edition.selection_barycenter, cage_edition.selection_scaling);
	
	/* Check if the render mode is one of ProxyMesh, TetMesh, or Off. 
	If so, use ImGuizmo's Manipulate function to update the edit_matrix based on user input.*/
	if(num_of_iterations < max_number_of_iterations){			//Limitatore di modifiche automatiche (altrimenti va in loop)	
		if ((render_mode == ESelectionRenderMode::ProxyMesh 	//Modifica avviata solo dopo la costruzione della Cage
		|| render_mode == ESelectionRenderMode::TetMesh 
		|| render_mode == ESelectionRenderMode::Off) 
/*		&&  ImGuizmo::Manipulate((const float*)&world2view, 	//fa apparire l'interfaccia di ImGuizmo per manipolare manualmente la deformazione

		(const float*)&view2proj_guizmo, 
		(ImGuizmo::OPERATION)m_gizmo_op, 
		(ImGuizmo::MODE)m_gizmo_mode, 
		(float*)&edit_matrix, NULL, NULL)*/
		) {
			//Siccome la modifica automatica viene avviata solo se si passa dalla funzione select_cage_rect() oppure select_scribbling()
			//Simula l'utilizzo della funzione select_cage_rect() se do_it_once==false
			if (render_mode == ESelectionRenderMode::ProxyMesh && do_it_once==false) {	//CHECK DA SEMPLIFICARE ##############################################
				for (int i = 0; i < proxy_cage.vertices.size(); i++) { 		//sembra ridondante, eppure non funziona togliendo questo for
					proxy_cage.labels[i] = 1;
							/*proxy_cage.colors[i] = Eigen::Vector3f(m_brush_color[0], m_brush_color[1], m_brush_color[2]);*/	//inutile, ma fa capire cosa è stato selezionato
				}
				Eigen::Matrix<float, 4, 4> screen_selection;			//inizializzata con valori che sembrano comprendere l'intera schermata
				screen_selection << 1030.0f, 0.0f, -900.0f, 1750.0f,	//Bisognerebbe testare questi valori anche su schermi grandi 4k
									15.0f, -1000.0f, -475.0f, 1880.0f,
									0.0f, 0.0f, 11.0f, 2.0f,
									0.0f, 0.0f, 0.0f, 2.0f;
				select_cage_rect(screen_selection);						//con select_scribbling non funziona siccome è legato alla posizione del mouse 
				reset_cage_selection();									//resetto tutto per non lasciare tracce, l'importante è passare per questa parte di codice
				
				// Inizializza un generatore di numeri casuali
				std::random_device rd;
				std::mt19937 gen(rd());
				// Definisci il range di numeri casuali
				float min_value = -0.03;
				float max_value = 0.03;
				// Crea una distribuzione uniforme tra min_value e max_value
				std::uniform_real_distribution<float> dis(min_value, max_value);
				// Genera un numero casuale all'interno del range
				float random_float_x = dis(gen);
				float random_float_y = dis(gen);
				float random_float_z = dis(gen);
				random_translation = point_t(random_float_x, random_float_y, random_float_z);
				std::cout << "random_translation:      X=" << random_float_x << "      Y=" << random_float_y << "      Z=" << random_float_z << std::endl;

				do_it_once = true;										//il flag si potrebbe evitare con un do-while, ma almeno così è resettabile
				//std::cout << "do_it_once: " << do_it_once << std::endl;
			}
			if (apply_all_edits_flag == true) {			//Il flag diventa True cliccando sul Button Apply_all_edits
				std::cout << "Number of edits of current Operator: from " << num_of_iterations << std::endl;
				edited_guizmo = true;
				matrix3_t guizmo_rotation;
				point_t guizmo_translation;
				point_t guizmo_scale = point_t(1.0f, 1.0f, 1.0f);
				decompose_imguizmo_matrix<matrix3_t, point_t, float_t>(edit_matrix, guizmo_rotation, guizmo_translation, guizmo_scale);

				//Vettore TRASLATION: scegli direzione e intensità dello spostamento del volume interno alla porzione di cage selezionata
				//Sottrazione tra Coordinate punto guizmo spostato e Coordinate punto guizmo iniziale (baricentro) 
				point_t translation = guizmo_translation - cage_edition.selection_barycenter;
		/*		std::cout << "cage_edition.selection_barycenter vector: " << cage_edition.selection_barycenter << std::endl; 	
				std::cout << "cage_edition.selection_rotation.transpose(): " << cage_edition.selection_rotation.transpose() << std::endl;  
		*/      matrix3_t rotation = guizmo_rotation * cage_edition.selection_rotation.transpose();
				point_t scaling = guizmo_scale.cwiseQuotient(cage_edition.selection_scaling);
		/*		std::cout << "Initial Translation vector: " << translation << std::endl;  
				std::cout << "Initial rotation matrix: " << rotation << std::endl;  
		*/
		/*		float theta = M_PI/48;
				rotation << 1, 0, 0, 
							0, cos(theta), -sin(theta), 
							0, sin(theta), cos(theta);										//rotazione forzata rispetto all'asse x verso sinistra
				guizmo_rotation << rotation * cage_edition.selection_rotation;				//Per rendere il valore Guizmo rotation coerente con la rotazione imposta
				std::cout << "Final rotation matrix: " << rotation << std::endl;  
		*/
				//translation = random_translation; 											//modifica al vettore spostamento (modifica arbitraria, verso l'alto)
				//translation << 0.000f, 0.03f, 0.000f;										//Spostamento verso l'alto
				guizmo_translation << translation + cage_edition.selection_barycenter; 		//Rende il punto iniziale Guizmo coerente con lo spostamento, per la prossima modifica
		//		std::cout << "Final Translation vector: " << translation << std::endl;  

				// std::cout << m_rotation_matrix.determinat() << std::endl;
				// m_scale = guizmo_scale.cwiseQuotient(m_selection_box.scale);

				//Apply the transformation to the selected vertices, either rigidly or non-rigidly, depending on the value of  m_rigid_editing.
				if (m_target == EManipulationTarget::CageVerts)
				{
					if (m_rigid_editing)				//non so come attivare questa parte di codice
					{
						for (auto& selected_vertex : proxy_cage.vertices) {
							// Rotate (w.r.t. barycenter)
							selected_vertex = rotation * (selected_vertex - proxy_cage.bbox.center()) + proxy_cage.bbox.center();
							selected_vertex = scaling.cwiseProduct(selected_vertex - proxy_cage.bbox.center()) + proxy_cage.bbox.center();
							// Then, translate
							selected_vertex += translation;
						}
						proxy_cage.bbox.set_center(proxy_cage.bbox.center() + translation);
					}
					else
					{
						//CODICE PER RICERCARE IL VERTICE PIU' DISTANTE DAL PUNTO SUPERFICIALE(only_point) LASCIATO CON remove_but_one() nel m_selection_points originale
						//Una volta individuato, si ricercano anche un gruppo di vertici più vicini ad esso, in modo da applicare la deformazione ad una piccola zona  				
						Eigen::Vector3f diff;
						float euclidean_distance;
						std::vector<std::pair<std::uint32_t, float>> id_distance_vec; 
						 
						for (uint32_t id = 0; id < proxy_cage.vertices.size(); id++) {
							//Calcola la differenza tra le coordinate dei punti
							diff = only_point - proxy_cage.vertices.at(id);							//only_point viene inizializzato da remove_but_one()
							//Calcola la distanza euclidea
							euclidean_distance = diff.norm();							
							id_distance_vec.push_back(std::make_pair(id, euclidean_distance));      //viene riempito un vettore di distanze      
							//std::cout << "id: " << id << ", distance: " << euclidean_distance << " added to id_distance_vec" << std::endl;
						}
						// Ordina il vettore in ordine decrescente in base alla distanza
						std::sort(id_distance_vec.begin(), id_distance_vec.end(), [](const auto& a, const auto& b) {
							return a.second > b.second; // Ordine decrescente
						});

						std::vector<std::uint32_t> vertices_to_move_vec;							//Conterrà tutti i vertici che subiranno la deformazione
						vertices_to_move_vec.push_back(id_distance_vec[0].first);					//Il primo vertice nel vettore è il più lontano dall'only_point
						/*//Stampa vettore debug, DA RIMUOVERE
						for(int i=0 ; i<id_distance_vec.size(); i++) {
							std::cout << "id: " << id_distance_vec[i].first << ", distance: " << id_distance_vec[i].second << std::endl;
						}*/

						//Calcola ed aggiunge ad un vettore tutti i vertici della cage assieme alla loro distanza dal vertice più lontano
						std::vector<std::pair<std::uint32_t, float>> id_distance_from_furthest_point; 						 
						for (uint32_t id = 0; id < proxy_cage.vertices.size(); id++) {//per ogni vertice
							// Calcola la differenza tra le coordinate del punto pi+ lontano e il vertice in considerazione
							diff = proxy_cage.vertices.at(id_distance_vec[0].first) - proxy_cage.vertices.at(id);
							// Calcola la distanza euclidea
							euclidean_distance = diff.norm();							
							id_distance_from_furthest_point.push_back(std::make_pair(id, euclidean_distance));            
							//std::cout << "id: " << id << ", distance from vertex " << id_distance_vec[0].first << ": " << euclidean_distance << " added to id_distance_from_furthest_point" << std::endl;
						}
						// Ordina il vettore in ordine crescente (servono i vertici attorno a quello più lontano) in base alla distanza
						std::sort(id_distance_from_furthest_point.begin(), id_distance_from_furthest_point.end(), [](const auto& a, const auto& b) {
							return a.second < b.second; // Ordine crescente
						});

						// Rimuovere il primo vertice: il vertice più vicino al vertice più lontano è il vertice stesso (che è già stato aggiunto)
						if (!id_distance_from_furthest_point.empty()) {
							id_distance_from_furthest_point.erase(id_distance_from_furthest_point.begin());
						}
						/*//Stampa vettore debug, DA RIMUOVERE
						for(int i=0 ; i<id_distance_from_furthest_point.size(); i++) {
							std::cout << "id: " << id_distance_from_furthest_point[i].first << ", distance from vertex: " << id_distance_from_furthest_point[i].second << std::endl;
						}*/
						// Riempie il vettore finale con i vertici da spostare						
						int num_vertices_around = 9;								//quanti vertici spostare attorno al più lontano
						for (int i=0; i<num_vertices_around; i++) {	
							vertices_to_move_vec.push_back(id_distance_from_furthest_point[i].first);
						}
						/*//Stampa vettore debug, DA RIMUOVERE
						for(int i=0 ; i<vertices_to_move_vec.size(); i++) {
							std::cout << "TO MOVE id: " << vertices_to_move_vec[i] << std::endl;
						}*/
						std::cout << "TO MOVE: " << vertices_to_move_vec.size() << " cage vertices" << std::endl;
						//Selezione vertici da traslare
						for (uint32_t i=0; i<vertices_to_move_vec.size(); i++) {
							//proxy_cage.labels[vertices_to_move_vec[i]] = 1;
							cage_edition.selected_vertices.push_back(vertices_to_move_vec[i]);
						}	

						Eigen::Vector3f from_center_vector;
						Eigen::Vector3f sphere_center(0.5025475, 0.62293, 0.500645);			//trovato empiricamente, vale solo per il modello earth
						float distance_from_center;
						Eigen::Vector3f normal;

						//Codice originale per applicare traslazione e rotazione ad ogni vertice
						for (const auto selected_vertex : cage_edition.selected_vertices) {
							// Rotate (w.r.t. barycenter)
							proxy_cage.vertices[selected_vertex] = rotation * (proxy_cage.vertices[selected_vertex] - cage_edition.selection_barycenter) + cage_edition.selection_barycenter;
							// Scale (by rotating back, scaling and then rotation again)
							proxy_cage.vertices[selected_vertex] = cage_edition.selection_rotation * scaling.cwiseProduct(cage_edition.selection_rotation.transpose() * (proxy_cage.vertices[selected_vertex] - cage_edition.selection_barycenter)) + cage_edition.selection_barycenter;
							// Then, translate
							
							//Calcolo del vettore traslazione in modo che sia parallelo alla normale
							from_center_vector = proxy_cage.vertices[selected_vertex] - sphere_center;	//Calcolo del vettore che va dal centro della sfera al punto sulla sfera
							distance_from_center = from_center_vector.norm();						//Calcolo la norma (lunghezza) del vettore
							normal = from_center_vector / distance_from_center;						//Calcolo della normale normalizzando il vettore
							
							translation = normal * 0.03;											//per ridurre l'impatto della traslazione
							proxy_cage.vertices[selected_vertex] += translation;					//applica la traslazione
						}			
				
/*						//Codice sostituito per applicare l'editing a tutti i vertici della cage e non solo a quelli selezionati in verde 		
						for (auto& selected_vertex : proxy_cage.vertices) {
							// Rotate (w.r.t. barycenter)
							selected_vertex = rotation * (selected_vertex - cage_edition.selection_barycenter) + cage_edition.selection_barycenter;
							std::cout << "selected_vertex " << i << ": " << selected_vertex << std::endl;
							// Scale (by rotating back, scaling and then rotation again)
							selected_vertex = cage_edition.selection_rotation * scaling.cwiseProduct(cage_edition.selection_rotation.transpose() * (selected_vertex - cage_edition.selection_barycenter)) + cage_edition.selection_barycenter;
							std::cout << "selected_vertex " << i << ": " << selected_vertex << std::endl;
							// Then, translate
							selected_vertex += translation;
							std::cout << "Final selected_vertex " << i << ": " << selected_vertex << std::endl;
						}*/
					}
				}
				//Update the barycenter, rotation, and scaling of the selection based on the new transformation.
				cage_edition.selection_barycenter = guizmo_translation;
				cage_edition.selection_rotation = guizmo_rotation;
				cage_edition.selection_scaling = guizmo_scale;

				//If the m_update_tet_manipulation flag is enabled, update the tetrahedral mesh by interpolating the Poisson boundary and updating the mesh.
				// If auto update is activated, perform it
				if (m_update_tet_manipulation) {						//senza il codice sottostante, le modifiche non si propagano ai colori e al volume interno alla proxy_cage
					// Only update if the tet mesh already exists!
					if (tet_interpolation_mesh) {							
						interpolate_poisson_boundary();						
						update_tet_mesh();
						num_of_iterations++;
						std::cout << "Number of edits of current Operator: to " << num_of_iterations << std::endl;
					}
				}

				//SI POTREBBE ATTIVARE UN FLAG QUI PER SWITCHARE OPERATORE DA testbed.cu

			}
			
		}
	}

    if (render_mode == ESelectionRenderMode::ScreenSelection) {
        if (ImGui::IsKeyDown(SCREEN_SELECTION_KEY) && io.MouseDown[0]) {
			Eigen::Vector2i selected_pixel = Eigen::Vector2i(io.MousePos.x, io.MousePos.y);
			if (selected_pixel != m_last_selected_pixel) {
				if (m_selection_mode == ESelectionMode::PixelWise) {
					if (selected_pixel != m_last_selected_pixel) {
						// std::cout << "Selected new pixel: " << selected_pixel << std::endl;
						m_selected_pixels_imgui.push_back(ImVec2(io.MousePos.x, io.MousePos.y));
						m_selected_pixels.push_back(selected_pixel);
						m_last_selected_pixel = selected_pixel;
					}
				} else if (m_selection_mode == ESelectionMode::Scribble) {
					// If there is a last selected pixel, connect both (with naive line drawing algorithm)
					if (m_last_selected_pixel.x() >= 0 && m_last_selected_pixel.y() >= 0) {
						Eigen::Vector2i min_pos = m_last_selected_pixel.cwiseMin(selected_pixel);
						Eigen::Vector2i max_pos = m_last_selected_pixel.cwiseMax(selected_pixel);
						Eigen::Vector2i diff = max_pos - min_pos;
						for (int x = min_pos.x() + 1; x < max_pos.x(); x++) {
							Eigen::Vector2i new_pixel (x, min_pos.y() + diff.y() * (x - min_pos.x()) / diff.x());
							m_selected_pixels_imgui.push_back(ImVec2(new_pixel.x(), new_pixel.y()));
							m_selected_pixels.push_back(new_pixel);
						}
					}
					// In any case add the new pixel if not similar to the previous one
					m_selected_pixels_imgui.push_back(ImVec2(io.MousePos.x, io.MousePos.y));
					m_selected_pixels.push_back(selected_pixel);
					m_last_selected_pixel = selected_pixel;
				}
			}
			          
        }
		
		// Make sure to reset the last selected pixel if the user release the key
		if (ImGui::IsKeyReleased(SCREEN_SELECTION_KEY)) {
			m_last_selected_pixel = Eigen::Vector2i(-1, -1);
		}

        for (auto& selected_pixel: m_selected_pixels_imgui) {
            list->AddCircleFilled(ImVec2(selected_pixel.x, selected_pixel.y), 4.0f, IM_COL32(255, 0, 0, 255));
        }
    }

	// ----------------------
	// Handle projection deletion
	if (ImGui::IsKeyDown(ImGuiKey_Delete)) {
		if (render_mode == ESelectionRenderMode::Projection) {
			delete_selected_projection();
		} else if (render_mode == ESelectionRenderMode::RegionGrowing) {
			delete_selected_growing();
		}
	}

	// ----------------------
	// Handle scribble selection
	if(ImGui::IsKeyPressed(ImGuiKey_LeftCtrl))
	{
		if (io.MouseWheel > 0)
			m_select_radius *= 1.5f;
		if (io.MouseWheel < 0)
			m_select_radius /= 1.5f;
		select_scribbling(world2proj);
		list->AddCircle(io.MousePos, m_select_radius+0.5f, IM_COL32(0, 255, 255, 255));
	}

    // ----------------------
	// Handle cage selection
	if (ImGui::IsKeyPressed(ImGuiKey_LeftShift) && !currently_selecting_cage) {
		mouse_clicked_selecting_cage = io.MousePos;
		currently_selecting_cage = true;
		selected_cage = false;
	}
	if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
		list->AddRect(mouse_clicked_selecting_cage, io.MousePos, 0xffffffff, ImDrawFlags_Closed);
	}
	if (ImGui::IsKeyReleased(ImGuiKey_LeftShift)) {
		mouse_released_selecting_cage = io.MousePos;
		currently_selecting_cage = false;
		selected_cage = true;
		list->AddRect(mouse_clicked_selecting_cage, mouse_released_selecting_cage, 0xff40ff40, ImDrawFlags_Closed);
		select_cage_rect(world2proj);
	}
	if (ImGui::IsKeyPressed(ImGuiKey_LeftAlt)) {
		reset_cage_selection();
	}

	if (m_visualize_max_level_cube) {
		visualize_level_cube(world2proj, m_growing_level);
	}

	if (!m_plane_dir.isZero())
	{
		auto dim = tet_interpolation_mesh->original_bbox.max - tet_interpolation_mesh->original_bbox.min;

		visualize_quad(world2proj, m_plane_pos, m_plane_dir1.normalized() * dim.x() * 0.5f, m_plane_dir2.normalized() * dim.y() * 0.5f);
	}

	return edited_guizmo;
}

void GrowingSelection::color_selection() {
	if (proxy_cage.outside_colors.size() != proxy_cage.vertices.size()) {
		proxy_cage.outside_colors.resize(proxy_cage.vertices.size());
	}

	for (const auto selected_vertex: cage_edition.selected_vertices) {
		// Rotate (w.r.t. barycenter)
		proxy_cage.outside_colors[selected_vertex] = Eigen::Vector3f{m_brush_color[0], m_brush_color[1], m_brush_color[2]};
		proxy_cage.colors[selected_vertex] = Eigen::Vector3f{m_brush_color[0], m_brush_color[1], m_brush_color[2]};
	}
}

//Launched by GrowingSelection::select_scribbling  (ctrl+scroll wheel)
inline bool GrowingSelection::is_near_mouse(const ImVec2& p)
{
	ImGuiIO& io = ImGui::GetIO();
	Eigen::Vector2i selected_pixel = Eigen::Vector2i(io.MousePos.x, io.MousePos.y);
	Eigen::Vector2i candidate_pixel = Eigen::Vector2i(p.x, p.y);
	if ((selected_pixel - candidate_pixel).norm() < m_select_radius)
	{
		return true;
	}
	return false;
}

inline bool GrowingSelection::is_inside_rect(const ImVec2& p) {

	if (p.x >= std::min(mouse_clicked_selecting_cage.x, mouse_released_selecting_cage.x) &&
		p.x <= std::max(mouse_clicked_selecting_cage.x, mouse_released_selecting_cage.x) &&
		p.y >= std::min(mouse_clicked_selecting_cage.y, mouse_released_selecting_cage.y) &&
		p.y <= std::max(mouse_clicked_selecting_cage.y, mouse_released_selecting_cage.y)) {
		return true;
	}
	else {
		return false;
	}
}

//Launched by clicking on ALT Button
void GrowingSelection::reset_cage_selection() {
	// In case of proxy mesh
    for (uint32_t i = 0; i < proxy_cage.labels.size(); i++) {
        proxy_cage.labels[i] = 0;
		proxy_cage.colors[i] = Eigen::Vector3f(
            m_cage_color[0],
            m_cage_color[1],
            m_cage_color[2]);
    }
    cage_edition.selected_vertices.clear();
	// In case of projected pixels
	for (int i = 0; i < m_projected_labels.size(); i++) {
		m_projected_labels[i] = 0;
	}
	// In case of selected points
	for (int i = 0; i < m_selection_labels.size(); i++) {
		m_selection_labels[i] = 0;
	}
}

//Launched by canc button
template<typename T>
inline std::vector<T> erase_indices(const std::vector<T>& data, std::vector<size_t>& indicesToDelete/* can't assume copy elision, don't pass-by-value */)
{
    if(indicesToDelete.empty())
        return data;

    std::vector<T> ret;
    ret.reserve(data.size() - indicesToDelete.size());

    std::sort(indicesToDelete.begin(), indicesToDelete.end());

    // new we can assume there is at least 1 element to delete. copy blocks at a time.
    typename std::vector<T>::const_iterator itBlockBegin = data.begin();
    for(std::vector<size_t>::const_iterator it = indicesToDelete.begin(); it != indicesToDelete.end(); ++ it)
    {
        typename std::vector<T>::const_iterator itBlockEnd = data.begin() + *it;
        if(itBlockBegin != itBlockEnd)
        {
            std::copy(itBlockBegin, itBlockEnd, std::back_inserter(ret));
        }
        itBlockBegin = itBlockEnd + 1;
    }

    // copy last block.
    if(itBlockBegin != data.end())
    {
        std::copy(itBlockBegin, data.end(), std::back_inserter(ret));
    }

    return ret;
}

//Launched by canc button
void GrowingSelection::delete_selected_projection() {
	std::vector<size_t> pixels_to_delete;
	for (int i = 0; i < m_projected_pixels.size(); i++) {
		if (m_projected_labels[i] == 1) {
			pixels_to_delete.push_back(i);
		}
	}
	if (pixels_to_delete.size() > 0) {
		m_projected_pixels = erase_indices<Eigen::Vector3f>(m_projected_pixels, pixels_to_delete);
		m_projected_labels = erase_indices<uint8_t>(m_projected_labels, pixels_to_delete);
		m_projected_cell_idx = erase_indices<uint32_t>(m_projected_cell_idx, pixels_to_delete);
		// m_projected_features = erase_indices<FeatureVector>(m_projected_features, pixels_to_delete);
	}
	// Don't forget to reset growing!
	reset_growing();
}

//Launched by canc button
void GrowingSelection::delete_selected_growing() {
	std::vector<size_t> points_to_delete;
	for (int i = 0; i < m_selection_points.size(); i++) {
		if (m_selection_labels[i] == 1) {
			points_to_delete.push_back(i);
		}
	}
	if (points_to_delete.size() > 0) {
		// Update the bitfield accordingly
		for (auto idx: points_to_delete) {
			uint32_t level = m_selection_cell_idx[idx] / NERF_GRIDVOLUME();
			uint32_t pos_idx = m_selection_cell_idx[idx] % NERF_GRIDVOLUME();
			set_bitfield_at(pos_idx, level, false, m_selection_grid_bitfield.data());
		}
		m_selection_points = erase_indices<Eigen::Vector3f>(m_selection_points, points_to_delete);
		m_selection_labels = erase_indices<uint8_t>(m_selection_labels, points_to_delete);
		m_selection_cell_idx = erase_indices<uint32_t>(m_selection_cell_idx, points_to_delete);
	}
}

//Launched by ctrl+scroll wheel
void GrowingSelection::select_scribbling(const Eigen::Matrix<float, 4, 4>& world2proj) {
	//std::cout << "world2proj: " << world2proj << std::endl;			//per testare, da rimuovere
	if (render_mode == ESelectionRenderMode::Projection) {
		// Take every projected pixel and reproject it in screen space
		uint32_t n_selected = 0;
		std::vector<int> pixel_to_delete;
		for (uint32_t i = 0; i < m_projected_pixels.size(); i++) {
			const Eigen::Vector3f& p = m_projected_pixels[i];
			Eigen::Vector4f ph; ph << p, 1.f;
			Eigen::Vector4f pa = world2proj * ph;
			ImVec2 o;
			if (pa.w() <= 0.f) continue;
			o.x = pa.x() / pa.w();
			o.y = pa.y() / pa.w();
			if (is_near_mouse(o)) {
				n_selected++;
				m_projected_labels[i] = 1;
			}
		}
	}
	else if (render_mode == ESelectionRenderMode::RegionGrowing) {
		// Take every selected pixel and reproject it in screen space
		uint32_t n_selected = 0;
		std::vector<int> pixel_to_delete;
		for (uint32_t i = 0; i < m_selection_points.size(); i++) {
			const Eigen::Vector3f& p = m_selection_points[i];
			Eigen::Vector4f ph; ph << p, 1.f;
			Eigen::Vector4f pa = world2proj * ph;
			ImVec2 o;
			if (pa.w() <= 0.f) continue;
			o.x = pa.x() / pa.w();
			o.y = pa.y() / pa.w();
			if (is_near_mouse(o)) {
				n_selected++;
				m_selection_labels[i] = 1;
			}
		}
	}
	else if (render_mode == ESelectionRenderMode::ProxyMesh) {
		// Take every vertex of the cage and reproject it in screen space
		uint32_t n_selected = 0;
		for (uint32_t i = 0; i < proxy_cage.vertices.size(); i++) {
			const point_t& p = proxy_cage.vertices[i];
			Eigen::Vector4f ph; ph << p.cast<float>(), 1.f;
			Eigen::Vector4f pa = world2proj * ph;
			ImVec2 o;	
			//If the projected vertex is behind the camera, it is skipped. 
			if (pa.w() <= 0.f) continue;
			o.x = pa.x() / pa.w();
			o.y = pa.y() / pa.w();
			//If the projected vertex is near the mouse cursor, it is marked as selected and its label and color are updated. 
			if (is_near_mouse(o)) {
				n_selected++;
				proxy_cage.labels[i] = 1;
				proxy_cage.colors[i] = Eigen::Vector3f(
					m_brush_color[0],
					m_brush_color[1],
					m_brush_color[2]
				);
			}
		}
		// Update the cage edition structure accordingly 
		cage_edition.selected_vertices.clear();
		for (uint32_t i = 0; i < proxy_cage.vertices.size(); i++) {
			if (proxy_cage.labels[i] == 1) {
				cage_edition.selected_vertices.push_back(i);
			}
		}
		cage_edition.selection_barycenter = point_t::Zero();
		if (!m_plane_dir.isZero())
		{
			matrix3_t rot;
			rot.col(0) = m_plane_dir1;
			rot.col(1) = m_plane_dir;
			rot.col(2) = m_plane_dir2;
			cage_edition.selection_rotation = rot;
		}
		else
		{
			cage_edition.selection_rotation = matrix3_t::Identity();
		}
		cage_edition.selection_scaling = point_t::Ones();
		
		//Sostituito cage_edition.selection_barycenter IN MODO CHE NON DIPENDA DA UNA CAGE
		//Sostituito cage_edition.selected_vertices con i punti rossi std::vector<Eigen::Vector3f> m_selection_points
		//Compute the barycenter and rotation matrix of the selection
		//std::cout << "cage_edition.selected_vertices size: " << cage_edition.selected_vertices.size() << std::endl;
		//std::cout << "proxy_cage.vertices size: " << proxy_cage.vertices.size() << std::endl;

		for (const auto selected_vertex : cage_edition.selected_vertices) {
			cage_edition.selection_barycenter += proxy_cage.vertices[selected_vertex];
		}
		cage_edition.selection_barycenter /= cage_edition.selected_vertices.size();

		//std::cout << "Selected " << n_selected <<  " out of " << proxy_cage.vertices.size() << " vertices" << std::endl;
		//std::cout << "cage_edition.selection_barycenter ORIGINAL: " << cage_edition.selection_barycenter << std::endl;


		for (const auto selected_point : m_selection_points) {
			cage_edition.selection_barycenter += selected_point;
		}
		cage_edition.selection_barycenter /= m_selection_points.size();

		//std::cout << "cage_edition.selection_barycenter MODIFICATA: " << cage_edition.selection_barycenter << std::endl;

	}
}

//Voxel selection with Shift+Mouse Left Click
void GrowingSelection::select_cage_rect(const Eigen::Matrix<float, 4, 4>& world2proj) {
	//std::cout << "world2proj: " << world2proj << std::endl;				//per testare, da rimuovere
	if (render_mode == ESelectionRenderMode::Projection) {
		// Take every projected pixel and reproject it in screen space
		uint32_t n_selected = 0;
		std::vector<int> pixel_to_delete;
		for (uint32_t i = 0; i < m_projected_pixels.size(); i++) {
			const Eigen::Vector3f& p = m_projected_pixels[i];
			Eigen::Vector4f ph; ph << p, 1.f;
			Eigen::Vector4f pa = world2proj * ph;
			ImVec2 o;
			if (pa.w() <= 0.f) continue;
			o.x = pa.x() / pa.w();
			o.y = pa.y() / pa.w();
			if (is_inside_rect(o)) {
				n_selected++;
				m_projected_labels[i] = 1;
			}
		}
	} else if (render_mode == ESelectionRenderMode::RegionGrowing) {
		// Take every selected pixel and reproject it in screen space
		uint32_t n_selected = 0;
		std::vector<int> pixel_to_delete;
		for (uint32_t i = 0; i < m_selection_points.size(); i++) {
			const Eigen::Vector3f& p = m_selection_points[i];
			Eigen::Vector4f ph; ph << p, 1.f;
			Eigen::Vector4f pa = world2proj * ph;
			ImVec2 o;
			if (pa.w() <= 0.f) continue;
			o.x = pa.x() / pa.w();
			o.y = pa.y() / pa.w();
			if (is_inside_rect(o)) {
				n_selected++;
				m_selection_labels[i] = 1;
			}
		}
	} else if (render_mode == ESelectionRenderMode::ProxyMesh) {
		// Take every vertex of the cage and reproject it in screen space
		uint32_t n_selected = 0;
		for (uint32_t i = 0; i < proxy_cage.vertices.size(); i++) {
			const point_t& p = proxy_cage.vertices[i];
			Eigen::Vector4f ph; ph << p.cast<float>(), 1.f;
			Eigen::Vector4f pa = world2proj * ph;
			ImVec2 o;
			if (pa.w() <= 0.f) continue;
			o.x = pa.x() / pa.w();
			o.y = pa.y() / pa.w();
			if (is_inside_rect(o)) {
				n_selected++;
				proxy_cage.labels[i] = 1;
				proxy_cage.colors[i] = Eigen::Vector3f(
					m_brush_color[0],
					m_brush_color[1],
					m_brush_color[2]
				);
			}
		}
		// Updates the cage edition structure by clearing the selected vertices and adding the newly selected vertices
		cage_edition.selected_vertices.clear();
		for (uint32_t i = 0; i < proxy_cage.vertices.size(); i++) {
			if (proxy_cage.labels[i] == 1) {
				cage_edition.selected_vertices.push_back(i);
			}
		}
		cage_edition.selection_barycenter = point_t::Zero();
		if (!m_plane_dir.isZero())
		{
			matrix3_t rot;
			rot.col(0) = m_plane_dir1;
			rot.col(1) = m_plane_dir;
			rot.col(2) = m_plane_dir2;
			cage_edition.selection_rotation = rot;
		}
		else
		{
			cage_edition.selection_rotation = matrix3_t::Identity();
		}
		cage_edition.selection_scaling = point_t::Ones();

		//Sostituito cage_edition.selection_barycenter IN MODO CHE NON DIPENDA DA UNA CAGE
		//Sostituito cage_edition.selected_vertices con i punti rossi std::vector<Eigen::Vector3f> m_selection_points
		//Compute the barycenter and rotation matrix of the selection
		//std::cout << "cage_edition.selected_vertices size: " << cage_edition.selected_vertices.size() << std::endl;
		//std::cout << "proxy_cage.vertices size: " << proxy_cage.vertices.size() << std::endl;

		for (const auto selected_vertex: cage_edition.selected_vertices) {
			cage_edition.selection_barycenter += proxy_cage.vertices[selected_vertex];
		}
		cage_edition.selection_barycenter /= cage_edition.selected_vertices.size();

		//std::cout << "Selected " << n_selected <<  " out of " << proxy_cage.vertices.size() << " vertices" << std::endl;
		//std::cout << "cage_edition.selection_barycenter ORIGINAL: " << cage_edition.selection_barycenter << std::endl;


		for (const auto selected_point : m_selection_points) {
			cage_edition.selection_barycenter += selected_point;
		}
		cage_edition.selection_barycenter /= m_selection_points.size();

		//std::cout << "cage_edition.selection_barycenter MODIFICATA: " << cage_edition.selection_barycenter << std::endl;

	}
}

//Launched wih compute 
void GrowingSelection::set_proxy_mesh(std::vector<point_t>& points, std::vector<uint32_t>& indices)
{
	//std::cout << "GrowingSelection::set_proxy_mesh()" << std::endl;

	render_mode = ESelectionRenderMode::ProxyMesh;

	// Create the associated cage!
	proxy_cage = Cage<float_t, point_t>(points, indices);

	// DEBUG
	for (int i = 0; i < proxy_cage.colors.size(); i++) {
		proxy_cage.colors[i] = Eigen::Vector3f(
			m_cage_color[0],
			m_cage_color[1],
			m_cage_color[2]);
	}

	// TODO: refine this!
	// Compute the ideal edge length
	float bbox_diag_length = proxy_cage.bbox.diag().norm();
	ideal_tet_edge_length = bbox_diag_length * 1 / 20.0f;

	std::cout << "Computed proxy with " << points.size() << " vertices and " << indices.size() / 3 << " triangles" << std::endl;
}

//Launched by GrowingSelection::fix_proxy_mesh()
void GrowingSelection::compute_proxy_mesh() {
	//std::cout << "GrowingSelection::compute_proxy_mesh()" << std::endl;

	// If there is no selection mesh, extract it!
	if (selection_mesh.vertices.size() == 0) {
		extract_fine_mesh();
	}

	// Clear selected vertices
    cage_edition.selected_vertices.clear();

    // Copy to CPU
    uint32_t n_verts = selection_mesh.vertices.size();
    uint32_t n_indices = selection_mesh.indices.size();

	std::vector<point_t> new_vertices_proxy;
	std::vector<uint32_t> new_indices_proxy;

	Eigen::MatrixXd input_V(n_verts, 3);
	Eigen::MatrixXi input_F(n_indices / 3, 3);
	for (int i = 0; i < n_verts; i++) {
		input_V.row(i) = selection_mesh.vertices[i].cast<double>();
	}	
	for (int i = 0; i < n_indices / 3; i++) {
		input_F.row(i) << selection_mesh.indices[3*i], selection_mesh.indices[3*i+2], selection_mesh.indices[3*i+1];
	}
	Eigen::MatrixXd output_V;
	Eigen::MatrixXi output_F;
	Eigen::VectorXi output_J;
	if (m_decimation_algorithm == EDecimationAlgorithm::ShortestEdge) {
		igl::decimate(input_V, input_F, proxy_size, output_V, output_F, output_J);
	} else if (m_decimation_algorithm == EDecimationAlgorithm::ProgressiveHullsQuadratic) {
		if (m_progressive_hulls_params.presimplify) {
			igl::decimate(input_V, input_F, std::max(proxy_size, int(m_progressive_hulls_params.presimplification_ratio * input_F.rows())), output_V, output_F, output_J);
			input_F = output_F;
			input_V = output_V;
		}
		bool success = progressive_hulls_quadratic(input_V, input_F, proxy_size, output_V, output_F, output_J, m_progressive_hulls_params);
		if (!success) {
			std::cout << "Failed to compute progressive hulls, please try again!" << std::endl;
			return;
		}
	} else if (m_decimation_algorithm == EDecimationAlgorithm::ProgressiveHullsLinear) {
		if (m_progressive_hulls_params.presimplify) {
			igl::decimate(input_V, input_F, std::max(proxy_size, int(m_progressive_hulls_params.presimplification_ratio * input_F.rows())), output_V, output_F, output_J);
			input_F = output_F;
			input_V = output_V;
		}
		bool success = progressive_hulls_linear(input_V, input_F, proxy_size, output_V, output_F, output_J, m_progressive_hulls_params);
		if (!success) {
			std::cout << "Failed to compute progressive hulls, please try again!" << std::endl;
			return;
		}
	}

	n_verts = output_V.rows();
	n_indices = output_F.rows()*3;
	new_vertices_proxy.resize(n_verts);
	new_indices_proxy.resize(n_indices);

	for (int i = 0; i < n_verts; i++) {
		new_vertices_proxy[i] = output_V.row(i).cast<float_t>();
	}
	for (int i = 0; i < output_F.rows(); i++) {
		new_indices_proxy[3*i] = output_F.row(i)(0);
		new_indices_proxy[3*i+1] = output_F.row(i)(1);
		new_indices_proxy[3*i+2] = output_F.row(i)(2);
	}
	
	set_proxy_mesh(new_vertices_proxy, new_indices_proxy);
}

//ADATTARE IN MODO DA FAR CREARE UNA PROXY MESH PER OGNI PUNTO NEL m_selection_points
void GrowingSelection::compute_all_proxy_mesh() {
	std::cout << "GrowingSelection::compute_all_proxy_mesh()" << std::endl;
	std::cout << "m_selection_points.size(): " << m_selection_points.size() << std::endl;


	for (int i=0; i<m_selection_points.size(); i++){
		//AGGIUNGERE CONTROLLO SUI DUPLICATI
		std::cout << "Selection point" << i << " : " << m_selection_points[i] << std::endl;

		// If there is no selection mesh, extract it!
		if (selection_mesh.vertices.size() == 0) {
			extract_fine_mesh();
		}

		// Clear selected vertices
		cage_edition.selected_vertices.clear();

		// Copy to CPU
		uint32_t n_verts = selection_mesh.vertices.size();
		uint32_t n_indices = selection_mesh.indices.size();

		std::vector<point_t> new_vertices_proxy;
		std::vector<uint32_t> new_indices_proxy;

		Eigen::MatrixXd input_V(n_verts, 3);
		Eigen::MatrixXi input_F(n_indices / 3, 3);
		for (int i = 0; i < n_verts; i++) {
			input_V.row(i) = selection_mesh.vertices[i].cast<double>();
		}	
		for (int i = 0; i < n_indices / 3; i++) {
			input_F.row(i) << selection_mesh.indices[3*i], selection_mesh.indices[3*i+2], selection_mesh.indices[3*i+1];
		}
		Eigen::MatrixXd output_V;
		Eigen::MatrixXi output_F;
		Eigen::VectorXi output_J;
		if (m_decimation_algorithm == EDecimationAlgorithm::ShortestEdge) {
			igl::decimate(input_V, input_F, proxy_size, output_V, output_F, output_J);
		} else if (m_decimation_algorithm == EDecimationAlgorithm::ProgressiveHullsQuadratic) {
			if (m_progressive_hulls_params.presimplify) {
				igl::decimate(input_V, input_F, std::max(proxy_size, int(m_progressive_hulls_params.presimplification_ratio * input_F.rows())), output_V, output_F, output_J);
				input_F = output_F;
				input_V = output_V;
			}
			bool success = progressive_hulls_quadratic(input_V, input_F, proxy_size, output_V, output_F, output_J, m_progressive_hulls_params);
			if (!success) {
				std::cout << "Failed to compute progressive hulls, please try again!" << std::endl;
				return;
			}
		} else if (m_decimation_algorithm == EDecimationAlgorithm::ProgressiveHullsLinear) {
			if (m_progressive_hulls_params.presimplify) {
				igl::decimate(input_V, input_F, std::max(proxy_size, int(m_progressive_hulls_params.presimplification_ratio * input_F.rows())), output_V, output_F, output_J);
				input_F = output_F;
				input_V = output_V;
			}
			bool success = progressive_hulls_linear(input_V, input_F, proxy_size, output_V, output_F, output_J, m_progressive_hulls_params);
			if (!success) {
				std::cout << "Failed to compute progressive hulls, please try again!" << std::endl;
				return;
			}
		}

		n_verts = output_V.rows();
		n_indices = output_F.rows()*3;
		new_vertices_proxy.resize(n_verts);
		new_indices_proxy.resize(n_indices);

		for (int i = 0; i < n_verts; i++) {
			new_vertices_proxy[i] = output_V.row(i).cast<float_t>();
		}
		for (int i = 0; i < output_F.rows(); i++) {
			new_indices_proxy[3*i] = output_F.row(i)(0);
			new_indices_proxy[3*i+1] = output_F.row(i)(1);
			new_indices_proxy[3*i+2] = output_F.row(i)(2);
		}
		
		set_proxy_mesh(new_vertices_proxy, new_indices_proxy);
	}	
}

void GrowingSelection::fix_fine_mesh() {
	// If there is no fine mesh, skip!
	if (selection_mesh.vertices.size() == 0) {
		std::cout << "Can't fix selection mesh because it is not defined..." << std::endl;
		return;
	}

	uint32_t n_verts = selection_mesh.vertices.size();
    uint32_t n_indices = selection_mesh.indices.size();

	Eigen::MatrixXd input_V(n_verts, 3);
	Eigen::MatrixXi input_F(n_indices / 3, 3);
	for (int i = 0; i < n_verts; i++) {
		input_V.row(i) = selection_mesh.vertices[i].cast<double>();
	}
	for (int i = 0; i < n_indices / 3; i++) {
		input_F.row(i) << selection_mesh.indices[3*i], selection_mesh.indices[3*i+1], selection_mesh.indices[3*i+2];
	}
	Eigen::MatrixXd output_V;
	Eigen::MatrixXi output_F;
	meshfix(input_V, input_F, output_V, output_F);
	n_verts = output_V.rows();
	n_indices = output_F.rows()*3;

	std::vector<point_t> new_vertices;
	std::vector<uint32_t> new_indices;
	new_vertices.resize(n_verts);
	new_indices.resize(n_indices);
	for (int i = 0; i < n_verts; i++) {
		new_vertices[i] = output_V.row(i).cast<float_t>();
	}
	for (int i = 0; i < output_F.rows(); i++) {
		new_indices[3*i] = output_F.row(i)(0);
		new_indices[3*i+1] = output_F.row(i)(1);
		new_indices[3*i+2] = output_F.row(i)(2);
	}

	// Create the selection mesh
	selection_mesh.vertices = new_vertices;
	selection_mesh.indices = new_indices;
    selection_mesh.normals.resize(new_vertices.size());
}

void miniOBJ(std::string file, std::vector<point_t>& vertices_proxy, std::vector<uint32_t>& indices_proxy)
{
	std::ifstream infile(file);
	std::string line;
	while (std::getline(infile, line))
	{
		std::stringstream ss(line);
		char dummy;
		if (line.length() > 1 && line[0] == 'v' && line[1] == ' ')
		{
			Eigen::Vector3f vert;
			ss >> dummy;
			ss >> vert.z();
			ss >> vert.y();
			ss >> vert.x();
			vert *= 0.3333f;
			vert.x() *= -1;
			vert.x() += 0.5f;
			vert.y() += 0.5f;
			vert.z() += 0.5f;
			vertices_proxy.push_back(vert);
		}
		if (line.length() > 1 && line[0] == 'f' && line[1] == ' ')
		{
			std::vector<int> indices;
			int index, waste;

			ss >> dummy;
			while (ss >> index) // >> dummy >> waste >> dummy >> waste)
				indices.push_back(index - 1);

			if (indices.size() == 3)
			{
				indices_proxy.push_back(indices[0]);
				indices_proxy.push_back(indices[1]);
				indices_proxy.push_back(indices[2]);
			}
			else if (indices.size() == 4)
			{
				indices_proxy.push_back(indices[0]);
				indices_proxy.push_back(indices[1]);
				indices_proxy.push_back(indices[3]);
				indices_proxy.push_back(indices[3]);
				indices_proxy.push_back(indices[1]);
				indices_proxy.push_back(indices[2]);
			}
			else
			{
				throw std::runtime_error("No support for other faces!");
			}
		}
	}
}

void GrowingSelection::deform_proxy_from_file(std::string deformed_file)
{
	std::vector<point_t> vertices_proxy;
	std::vector<uint32_t> indices_proxy;
	miniOBJ(deformed_file, vertices_proxy, indices_proxy);

	if (vertices_proxy.size() != proxy_cage.original_vertices.size() ||
		indices_proxy.size() != proxy_cage.indices.size())
		throw std::runtime_error("Cages don't match!");

	proxy_cage.vertices = vertices_proxy;
}

void GrowingSelection::proxy_mesh_from_file(std::string orig_file)
{
	std::vector<point_t> vertices_proxy;
	std::vector<uint32_t> indices_proxy;
	miniOBJ(orig_file, vertices_proxy, indices_proxy);

	render_mode = ESelectionRenderMode::ProxyMesh;

	// Create the associated cage!
	proxy_cage = Cage<float_t, point_t>(vertices_proxy, indices_proxy);

	// TODO: refine this!
	// Compute the ideal edge length
	float bbox_diag_length = proxy_cage.bbox.diag().norm();
	ideal_tet_edge_length = bbox_diag_length * 1 / 20.0f;

	for (int i = 0; i < proxy_cage.colors.size(); i++) {
		proxy_cage.colors[i] = Eigen::Vector3f(
            m_cage_color[0],
            m_cage_color[1],
            m_cage_color[2]);
	}

	// std::cout << "Fixed proxy cage with meshfix" << std::endl;
}

//GUI Button "Compute Proxy" First function
void GrowingSelection::fix_proxy_mesh() {
	// If there is no proxy, extract one!
	if (proxy_cage.vertices.size() == 0) {
		compute_proxy_mesh();
	}

	// Reset vertices to original_vertices! (in cage the cage was edited inbetween)
	proxy_cage.reset_original_vertices();

	uint32_t n_verts = proxy_cage.vertices.size();
    uint32_t n_indices = proxy_cage.indices.size();

	if (n_verts == 0 || n_indices == 0)
	{
		std::cout << "Meshfix had nothing to fix" << std::endl;
		return;
	}
	//To store the vertices and indices of the proxy_cage, 
	//It loops through the vertices and indices of the "proxy_cage" 
	//and assigns their values to the corresponding rows of the Eigen matrices
	Eigen::MatrixXd input_V(n_verts, 3);
	Eigen::MatrixXi input_F(n_indices / 3, 3);
	for (int i = 0; i < n_verts; i++) {
		input_V.row(i) = proxy_cage.original_vertices[i].cast<double>();
	}
	for (int i = 0; i < n_indices / 3; i++) {
		input_F.row(i) << proxy_cage.indices[3*i], proxy_cage.indices[3*i+1], proxy_cage.indices[3*i+2];
	}

	Eigen::MatrixXd output_V;
	Eigen::MatrixXi output_F;
	meshfix(input_V, input_F, output_V, output_F);
	n_verts = output_V.rows();
	n_indices = output_F.rows()*3;

	std::vector<point_t> new_vertices_proxy;
	std::vector<uint32_t> new_indices_proxy;
	new_vertices_proxy.resize(n_verts);
	new_indices_proxy.resize(n_indices);
	for (int i = 0; i < n_verts; i++) {
		new_vertices_proxy[i] = output_V.row(i).cast<float_t>();
	}
	for (int i = 0; i < output_F.rows(); i++) {
		new_indices_proxy[3*i] = output_F.row(i)(0);
		new_indices_proxy[3*i+1] = output_F.row(i)(1);
		new_indices_proxy[3*i+2] = output_F.row(i)(2);
	}

	// Create the associated cage!
    proxy_cage = Cage<float_t, point_t>(new_vertices_proxy, new_indices_proxy);

	//std::cout << "Proxy cage Created!" << std::endl;
	//Set the proxy mesh using the new vertices and indices
    for (int i = 0; i < proxy_cage.colors.size(); i++) {
        proxy_cage.colors[i] = Eigen::Vector3f(
            m_cage_color[0],
            m_cage_color[1],
            m_cage_color[2]);
    }
	//std::cout << "Fixed proxy cage with meshfix" << std::endl;
}

void GrowingSelection::clear() {
    m_selected_pixels.clear();
    m_selected_pixels_imgui.clear();
    m_projected_pixels.clear();
	m_projected_cell_idx.clear();
	// m_projected_features.clear();
	m_projected_labels.clear();
    m_selection_points.clear();
	m_selection_labels.clear();
    m_selection_cell_idx.clear();
    m_selection_grid_bitfield.clear();
	

	// Reset growing
	// NOTE: this may be redundant too
	reset_growing();
}

void draw_debug_gl(
	const std::vector<Eigen::Vector3f>& points, 
	const std::vector<Eigen::Vector3f>& colors, 
	const Eigen::Vector2i& resolution, 
	const Eigen::Vector2f& focal_length, 
	const Eigen::Matrix<float, 3, 4>& camera_matrix, 
	const Eigen::Vector2f& screen_center) {
	
	if (points.size() == 0) {
        return;
    }

    static GLuint VAO = 0, VBO[4] = {}, vbosize = 0, program = 0, vs = 0, ps = 0;
    if (!VAO) {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
	}
    if (vbosize != points.size()) {
        // VBO for positions
        glGenBuffers(1, &VBO[0]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glBufferData(GL_ARRAY_BUFFER, points.size()*sizeof(Eigen::Vector3f), &points[0], GL_STATIC_DRAW);
        // VBO for boundary
        glGenBuffers(1, &VBO[1]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
        glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(Eigen::Vector3f), &colors[0], GL_STATIC_DRAW);
    }
    if (!program) {
        // vertex shader
        const char * vertexShaderSource = R"foo(
        layout (location = 0) in vec3 pos;
		layout (location = 1) in vec3 col;
        uniform mat4 camera;
        uniform vec2 f;
        uniform ivec2 res;
        uniform vec2 cen;
		out vec3 vtxcol;
        void main()
        {
            gl_PointSize = 2.0f;
            vec4 p = camera * vec4(pos, 1.0);
            p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
            p.w = p.z;
            p.z = p.z - 0.1;
            p.xy += cen * p.w;
            gl_Position = p;
			vtxcol = col;
        })foo";

        vs = compile_shader(false, vertexShaderSource);

        // fragment shader
        const char * fragmentShaderSource = R"foo(
		in vec3 vtxcol;
        out vec4 FragColor;
        void main() {
			FragColor = vec4(vtxcol, 1.0f);
        })foo";

        ps = compile_shader(true, fragmentShaderSource);

        program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, ps);
        glLinkProgram(program);
        if (!check_shader(program, "shader program", true)) {
			glDeleteProgram(program);
			program = 0;
		}

        glDeleteShader(vs);
        glDeleteShader(ps);
    }

    glBindVertexArray(VAO);

    glUseProgram(program);

    Eigen::Matrix4f view2world=Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();
    glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x(), focal_length.y());
	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x()*2.f-1.f, screen_center.y()*-2.f+1.f);
	glUniform2i(glGetUniformLocation(program, "res"), resolution.x(), resolution.y());

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glDrawArrays(GL_POINTS, 0, points.size());

    glDisableVertexAttribArray(0);
    // glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void draw_selection_gl(
    const std::vector<Eigen::Vector3f>& points, 
    const std::vector<uint8_t>& labels,
    const Eigen::Vector2i& resolution, 
    const Eigen::Vector2f& focal_length, 
    const Eigen::Matrix<float, 3, 4>& camera_matrix, 
    const Eigen::Vector2f& screen_center,
    const int pc_render_mode,
	const int max_label) {
    
    if (points.size() == 0) {
        return;
    }

    static GLuint VAO = 0, VBO[4] = {}, vbosize = 0, program = 0, vs = 0, ps = 0;
    if (!VAO) {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
	}
    if (vbosize != points.size()) {
        // VBO for positions
        glGenBuffers(1, &VBO[0]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glBufferData(GL_ARRAY_BUFFER, points.size()*sizeof(Eigen::Vector3f), &points[0], GL_STATIC_DRAW);
        // VBO for boundary
        glGenBuffers(1, &VBO[1]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
        glBufferData(GL_ARRAY_BUFFER, labels.size()*sizeof(uint8_t), &labels[0], GL_STATIC_DRAW);
    }
    if (!program) {
        // vertex shader
        const char * vertexShaderSource = R"foo(
        layout (location = 0) in vec3 pos;
		layout (location = 1) in int label;
        uniform mat4 camera;
        uniform vec2 f;
        uniform ivec2 res;
        uniform vec2 cen;
        uniform int mode;
		flat out int fLabel;
        void main()
        {
            gl_PointSize = 2.0f;
            vec4 p = camera * vec4(pos, 1.0);
            p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
            p.w = p.z;
            p.z = p.z - 0.1;
            p.xy += cen * p.w;
            gl_Position = p;
			fLabel = label;
        })foo";

        vs = compile_shader(false, vertexShaderSource);

        // fragment shader
        const char * fragmentShaderSource = R"foo(
        flat in int fLabel;
        out vec4 FragColor;
        uniform int mode;
		uniform int max_label;
        void main() {
			if (mode == 2) {
				if (fLabel == 0 && max_label > 0) {
					FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
				} else if (fLabel == 1 && max_label > 1) {
					FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
				} else if (fLabel == 2 && max_label > 2) {
					FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
				} else if (fLabel == 3 && max_label > 3) {
					FragColor = vec4(1.0f, 0.75f, 0.0f, 1.0f);
				} else if (fLabel == 4 && max_label > 4) {
					FragColor = vec4(0.9f, 0.55f, 0.9f, 1.0f);
				} else if (fLabel == 5 && max_label > 5) {
					FragColor = vec4(0.9f, 0.9f, 0.9f, 1.0f);
				}
				
                
            } else if (mode == 1) {
                FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }
        })foo";

        ps = compile_shader(true, fragmentShaderSource);

        program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, ps);
        glLinkProgram(program);
        if (!check_shader(program, "shader program", true)) {
			glDeleteProgram(program);
			program = 0;
		}

        glDeleteShader(vs);
        glDeleteShader(ps);
    }

    glBindVertexArray(VAO);

    glUseProgram(program);

    Eigen::Matrix4f view2world=Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();
    glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x(), focal_length.y());
	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x()*2.f-1.f, screen_center.y()*-2.f+1.f);
	glUniform2i(glGetUniformLocation(program, "res"), resolution.x(), resolution.y());
    glUniform1i(glGetUniformLocation(program, "mode"), pc_render_mode);
	glUniform1i(glGetUniformLocation(program, "max_label"), max_label);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribIPointer(1, 1, GL_UNSIGNED_BYTE, sizeof(uint8_t), 0);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glDrawArrays(GL_POINTS, 0, points.size());

    glDisableVertexAttribArray(0);
    // glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// NOTE: we have duplicates in here!
void inner_faces_from_tet(Eigen::MatrixXi& tet_tets, std::vector<uint32_t>& tet_indices) {
    tet_indices.resize(tet_tets.rows()*3*4);
    for (int i = 0; i < tet_tets.rows(); i++) {
        for (int j = 0; j < 4; j++) {
            tet_indices[3*4*i + 3*j] = tet_tets.row(i)(j);
            tet_indices[3*4*i + 3*j + 1] = tet_tets.row(i)((j+1)%4);
            tet_indices[3*4*i + 3*j + 2] = tet_tets.row(i)((j+2)%4);
        }
    }
}

void GrowingSelection::force_cage() {
	proxy_cage.original_vertices = proxy_cage.vertices;
}

void GrowingSelection::extract_tet_mesh() {
    // Make sure the proxy mesh is set
	if (proxy_cage.vertices.size() == 0) {
		compute_proxy_mesh();
		fix_proxy_mesh();
	}
    if (proxy_cage.vertices.size() == 0) {
        return;
    }

	// Reset vertices to original_vertices! (in cage the cage was edited inbetween)
	proxy_cage.reset_original_vertices();

    uint32_t n_verts = proxy_cage.original_vertices.size();
    uint32_t n_faces = proxy_cage.indices.size() / 3;
    Eigen::MatrixXd tetgen_pts(proxy_cage.original_vertices.size(), 3);
    Eigen::MatrixXi tetgen_faces(n_faces, 3);

	// std::cout << "Beginning " << n_verts << std::endl;
	
	// Note: use proxy_cage original weights!
    for (size_t i = 0; i < n_verts; ++i) {
        tetgen_pts.row(i) = proxy_cage.original_vertices[i].cast<double>();
    }

    for (size_t i = 0; i < n_faces; ++i) {
        tetgen_faces.row(i) = Eigen::Vector3i(int(proxy_cage.indices[3*i]), int(proxy_cage.indices[3*i+1]), int(proxy_cage.indices[3*i+2]));
    }

    Eigen::MatrixXi tetgen_generated_tets;
    Eigen::MatrixXd tetgen_generated_points;
    Eigen::MatrixXi tetgen_generated_faces;

    float max_volume = ideal_tet_edge_length * ideal_tet_edge_length * ideal_tet_edge_length *
                sqrt(2.) / 12.;

    std::stringstream buf;
    buf.precision(100);
    buf.setf(std::ios::fixed, std::ios::floatfield);
    buf << "Qpq2.0a"
        << max_volume;
    if (preserve_surface_mesh) {
        buf << "Y";
    }

    igl::copyleft::tetgen::tetrahedralize(tetgen_pts,
                                        tetgen_faces,
                                        buf.str(),
                                        tetgen_generated_points,
                                        tetgen_generated_tets,
                                        tetgen_generated_faces);

    n_verts = tetgen_generated_points.rows();
    n_faces = tetgen_generated_faces.rows();
    uint32_t n_tets = tetgen_generated_tets.rows();
    std::vector<point_t> output_vertices_host;
    std::vector<uint32_t> output_faces_host;
    std::vector<uint32_t> output_tets_host;
    output_vertices_host.resize(n_verts);
    output_faces_host.resize(n_faces*3);
    output_tets_host.resize(n_tets*4);
    for (int i = 0; i < n_verts; i++) {
        output_vertices_host[i] = tetgen_generated_points.row(i).cast<float_t>();
    }
    for (int i = 0; i < n_faces; i++) {
        output_faces_host[3*i] = tetgen_generated_faces.row(i)(0);
        output_faces_host[3*i+1] = tetgen_generated_faces.row(i)(1);
        output_faces_host[3*i+2] = tetgen_generated_faces.row(i)(2);
    }
    for (int i = 0; i < n_tets; i++) {
        output_tets_host[4*i] = tetgen_generated_tets.row(i)(0);
        output_tets_host[4*i+1] = tetgen_generated_tets.row(i)(1);
        output_tets_host[4*i+2] = tetgen_generated_tets.row(i)(2);
        output_tets_host[4*i+3] = tetgen_generated_tets.row(i)(3);
    }

    tet_interpolation_mesh = std::make_shared<TetMesh<float_t, point_t>>(output_vertices_host, output_faces_host, output_tets_host, m_aabb);

    std::vector<uint32_t> tet_faces_host;
    inner_faces_from_tet(tetgen_generated_tets, tet_faces_host);

	// std::cout << "RELEVANT: " << proxy_cage.vertices.size() << std::endl;
    std::cout << "Computed tet mesh with " << n_verts << " vertices, " << n_faces << " triangles and " << n_tets << " tets" << std::endl;
}   

void GrowingSelection::initialize_mvc() {
    if (proxy_cage.vertices.size() == 0) {
        std::cout << "Proxy cage is not defined..." << std::endl;
        return;
    }

    if (tet_interpolation_mesh->vertices.size() == 0) {
        std::cout << "Tet mesh is not defined..." << std::endl;
        return;
    }

    proxy_cage.compute_mvc(tet_interpolation_mesh->original_vertices, tet_interpolation_mesh->mvc_coordinates, tet_interpolation_mesh->labels, true);
	proxy_cage.compute_mvc(tet_interpolation_mesh->original_vertices, tet_interpolation_mesh->gamma_coordinates, tet_interpolation_mesh->labels, true, m_poisson_editing.mvc_gamma);

}

void GrowingSelection::update_tet_mesh() {
	// Make sure to compute the interpolation mesh before updating it
	if (!tet_interpolation_mesh) {
		extract_tet_mesh();
		initialize_mvc();
	}

	if (!tet_interpolation_mesh || tet_interpolation_mesh->vertices.size() == 0) {
		std::cout << "No or invalid tet interpolation mesh" << std::endl;
		return;
	}

	auto ab = std::chrono::system_clock::now();
	auto abc = std::chrono::system_clock::now();
    proxy_cage.interpolate_with_mvc(tet_interpolation_mesh);
	auto abcd = std::chrono::system_clock::now();
	if (m_correct_direction) {
		tet_interpolation_mesh->update_local_rotations(m_stream);
	}
	tet_interpolation_mesh->build_tet_grid(m_stream);
	//std::cout << (std::chrono::system_clock::now() - ab).count() << " " << (std::chrono::system_clock::now() - abc).count() << " " << (std::chrono::system_clock::now() - abcd).count() << std::endl;
}

// DEBUG
void GrowingSelection::export_proxy_mesh() {
    Eigen::MatrixXd V(proxy_cage.vertices.size(), 3);
    Eigen::MatrixXi F(proxy_cage.indices.size() / 3, 3);
    for (size_t i = 0; i < proxy_cage.vertices.size(); ++i) {
        V.row(i) = proxy_cage.vertices[i].cast<double>();
    }

    for (size_t i = 0; i < proxy_cage.indices.size() / 3; ++i) {
        F.row(i) = Eigen::Vector3i(int(proxy_cage.indices[3*i]), int(proxy_cage.indices[3*i+1]), int(proxy_cage.indices[3*i+2]));
    }
    igl::writePLY("proxy.ply",V,F);
}

void GrowingSelection::draw_gl(
        const Eigen::Vector2i& resolution, 
        const Eigen::Vector2f& focal_length, 
        const Eigen::Matrix<float, 3, 4>& camera_matrix, 
        const Eigen::Vector2f& screen_center) 
    {
    
	if (render_mode == ESelectionRenderMode::Projection) {
		draw_selection_gl(m_projected_pixels, m_projected_labels, resolution, focal_length, camera_matrix, screen_center, (int)m_pc_render_mode, m_pc_render_max_level);
	} else if (render_mode == ESelectionRenderMode::RegionGrowing) {
		draw_selection_gl(m_selection_points, m_selection_labels, resolution, focal_length, camera_matrix, screen_center, (int)m_pc_render_mode, m_pc_render_max_level);
	} else if (render_mode == ESelectionRenderMode::SelectionMesh) {
		selection_mesh.draw_gl(resolution, focal_length, camera_matrix, screen_center);
	} else if (render_mode == ESelectionRenderMode::ProxyMesh) {
		proxy_cage.draw_gl(resolution, focal_length, camera_matrix, screen_center);
	} else if (tet_interpolation_mesh != nullptr && render_mode == ESelectionRenderMode::TetMesh) {
		tet_interpolation_mesh->draw_gl(resolution, focal_length, camera_matrix, screen_center, display_in_tet);
	}
	draw_debug_gl(m_debug_points, m_debug_colors, resolution, focal_length, camera_matrix, screen_center);
}

__global__ void shoot_selection_rays_kernel(
	const uint32_t n_rays, 
	Vector2i* __restrict__ ray_pixels,
	Vector2i resolution,
	Vector2f focal_length,
	Matrix<float, 3, 4> camera_matrix,
	Vector2f screen_center,
	BoundingBox aabb,
	const uint32_t max_samples,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint8_t* __restrict__ density_grid,
	float cone_angle_constant,
	Eigen::Vector3f light_dir)  {

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	// Initialize the rays with the provided pixels
	Ray ray = pixel_to_ray(
		0,
		ray_pixels[i],
		resolution,
		focal_length,
		camera_matrix,
		screen_center
	);

	Vector2f tminmax = aabb.ray_intersect(ray.o, ray.d);
	float cone_angle = calc_cone_angle(ray.d.dot(camera_matrix.col(2)), focal_length, cone_angle_constant);
	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x() = fmaxf(tminmax.x(), 0.0f);
	float startt = tminmax.x();
	Vector3f idir = ray.d.cwiseInverse();

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t=startt;
	Vector3f pos;

	while (aabb.contains(pos = ray.o + t * ray.d) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, res);
		}
	}

	// No steps
	if (j == 0) {
		return;
	}

	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		// printf("Reached max samples!\n");
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	ray_indices_out[ray_idx] = i;
	rays_out[ray_idx] = ray;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	Vector3f light_dir_warped = warp_direction(light_dir);
	Vector3f warped_dir = warp_direction(ray.d);
	t=startt;
	j=0;
	while (aabb.contains(pos = ray.o + t * ray.d) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out(j)->set_with_optional_light_dir(warp_position(pos, aabb), warped_dir, warp_dt(dt), light_dir_warped, coords_out.stride_in_bytes);
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, res);
		}
	}
}

__global__ void composite_shot_rays(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	int padded_density_output_width,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const Ray* __restrict__ rays_in,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	Vector3f* coords_out,
	uint32_t* grid_indices,
	float transmittance_threshold
	// FeatureVector* __restrict__ accumulated_features
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { 
		// printf("Out of ray counter: %d (%d) \n", i, *rays_counter);
		return; 
	}

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];
	coords_in += base;
	network_output += base * padded_density_output_width;

	float T = 1.f;

	FeatureVector feature_ray = FeatureVector::Zero();

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T <= transmittance_threshold) {
			// Warped position!
			const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
			coords_out[i] = pos;
			const uint32_t level = mip_from_pos(pos);
			grid_indices[i] = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(pos, level);
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 16> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 16>*)network_output;
		const FeatureVector feature = network_to_feature(local_network_output);
		const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);

		float density = network_to_density(float(local_network_output[0]), density_activation);


		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		feature_ray += weight * feature;
		T *= (1.f - alpha);

		network_output += padded_density_output_width;
		coords_in += 1;
	}
	// If transmittance threshold wasn't reached
	if (T > transmittance_threshold) {
		coords_out[i] = aabb.min + Eigen::Vector3f(-1.f, -1.f, -1.f);
	}
	// accumulated_features[i] = feature_ray;
}

//GUI Button "Project" in Cage Deformation B B B B FONDAMENTALE PER SCRIBBLING
//Project the selected pixels using rays and then update the growing selection based on the projection results. 
void GrowingSelection::project_selection_pixels(const std::vector<Vector2i>& ray_pixels, const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center, cudaStream_t stream) {
	uint32_t n_rays = ray_pixels.size();
	if (n_rays == 0) {
		return;
	}
	

	//std::cout << "GrowingSelection::project_selection_pixels()" << std::endl;

	const uint32_t padded_output_width = m_nerf_network->padded_output_width();
	const uint32_t padded_density_output_width = m_nerf_network->padded_density_output_width();
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct
	const uint32_t max_samples = n_rays * NERF_STEPS();

	//Allocate GPU memory for various data structures
	tcnn::GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		uint32_t, // ray_indices
		Ray, // rays
		uint32_t, // numsteps
		float, // coords
		float, // max_level
		network_precision_t, // mlp_out
		uint32_t, // ray_counter
		uint32_t, // counter
		Vector2i, // pixels
		Vector3f, // coords_projected
		uint32_t,  // grid_indices
		FeatureVector // accumulated_features
	>(
		stream, &alloc,
		n_rays,
		n_rays,
		n_rays * 2,
		max_samples * floats_per_coord,
		max_samples,
		max_samples * padded_density_output_width,
		1,
		1,
		n_rays,
		n_rays,
		n_rays,
		n_rays
	);

	//Copy the input ray_pixels data to the GPU memory and initialize ray_counter and numsteps_counter with zeros
	uint32_t* ray_indices = std::get<0>(scratch);
	Ray* rays = std::get<1>(scratch);
	uint32_t* numsteps = std::get<2>(scratch);
	float* coords = std::get<3>(scratch);
	float* max_level = std::get<4>(scratch);
	// NOTE: this is the output of the density network!
	network_precision_t* mlp_out = std::get<5>(scratch);
	uint32_t* ray_counter = std::get<6>(scratch);
	uint32_t* numsteps_counter = std::get<7>(scratch);
	Vector2i* pixels = std::get<8>(scratch);
	Vector3f* coords_projected = std::get<9>(scratch);
	uint32_t* grid_indices = std::get<10>(scratch);
	FeatureVector* accumulated_features = std::get<11>(scratch);

	CUDA_CHECK_THROW(cudaMemcpyAsync(pixels, ray_pixels.data(), n_rays * sizeof(Vector2i), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter, 0, sizeof(uint32_t), stream));

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	//Compute the rays, number of steps, and coordinates for each ray
	linear_kernel(shoot_selection_rays_kernel, 0, stream,
		n_rays,
		pixels,
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		m_aabb,
		max_samples,
		ray_counter,
		numsteps_counter,
		ray_indices,
		rays,
		numsteps,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		m_density_grid_bitfield.data(),
		m_cone_angle_constant,
		m_light_dir.normalized());

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	//Copy the results from the GPU memory back to the host memory
	// TODO: pass NerfPosition rather than NerfCoordinate
	std::vector<NerfCoordinate> coords_host(max_samples, NerfCoordinate(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 0.f));

	std::vector<Ray> rays_host;
	rays_host.resize(n_rays);
	uint32_t numsteps_counter_host;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&numsteps_counter_host, numsteps_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaMemcpyAsync(rays_host.data(), rays, n_rays * sizeof(Ray), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaMemcpyAsync(coords_host.data(), coords, max_samples * floats_per_coord * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	//Check if any rays were found
	if (numsteps_counter_host == 0) {
		std::cout << "Couldn't find surface when shooting rays..." << std::endl;
		return;
	}

	tcnn::GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_samples);
	tcnn::GPUMatrix<network_precision_t> sigmafeature_matrix(mlp_out, padded_output_width, max_samples);

	//Compute the density values for the coordinates
	m_nerf_network->density(stream, coords_matrix, sigmafeature_matrix, false);

	//Compute the projected coordinates, grid indices and accumulated features
	linear_kernel(composite_shot_rays, 0, stream,
		n_rays,
		m_aabb,
		ray_counter,
		padded_density_output_width,
		mlp_out,
		numsteps_counter,
		rays,
		numsteps,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		m_rgb_activation,
		m_density_activation,
		coords_projected,
		grid_indices,
		transmittance_threshold
		// accumulated_features
	);

	//Copy the results back to the host memory
	std::vector<uint32_t> grid_indices_host_tmp;
	std::vector<uint32_t> grid_mips_host_tmp;
	std::vector<Eigen::Vector3f> m_projected_pixels_tmp;
	uint32_t ray_counter_host;
	// std::vector<FeatureVector> m_projected_features_tmp;
	m_projected_pixels_tmp.resize(n_rays);
	grid_indices_host_tmp.resize(n_rays);
	// m_projected_features_tmp.resize(n_rays);
	CUDA_CHECK_THROW(cudaMemcpyAsync(m_projected_pixels_tmp.data(), coords_projected, n_rays * sizeof(Vector3f), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaMemcpyAsync(grid_indices_host_tmp.data(), grid_indices, n_rays * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaMemcpyAsync(&ray_counter_host, ray_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	// CUDA_CHECK_THROW(cudaMemcpyAsync(m_projected_features_tmp.data(), accumulated_features, n_rays * sizeof(FeatureVector), cudaMemcpyDeviceToHost, stream));

	// printf("Shot %u rays but counted %u only.\n", n_rays, ray_counter_host);

	//Update the m_growing_level based on the maximum level found in the grid indices
	// If automatic level selection is on, we need to find the maximum level and set the growing level accordingly
	if (m_automatic_max_level) {
		m_growing_level = 0;
		for (int i = 0; i< ray_counter_host; i++) {
			if (m_aabb.contains(m_projected_pixels_tmp[i])) {
				uint32_t level = grid_indices_host_tmp[i] / NERF_GRIDVOLUME();

				// If it's bigger than the requested level, discard it
				if (level > m_growing_level) {
					m_growing_level = level;
				}
			}
		}
	}
	//std::cout << "ray_counter_host: " << ray_counter_host << std::endl;
	//std::cout << "m_projected_pixels_tmp size: " << m_projected_pixels_tmp.size() << std::endl;
	
	// Set to avoid duplicate cell_idx
	std::set<uint32_t> cell_idx_set;

	//Loop through the rays and update the projected cell indices, projected pixels, projected labels, and the cell index set
	// Check for rays that did not reach transmittance and discards them if they are outside the requested level
	// TODO: do something cleaner here...
	for (int i = 0; i< ray_counter_host; i++) {

			//Check if the current pixel is within the AABB
			if (m_aabb.contains(m_projected_pixels_tmp[i])) {
				//Get the level of the current grid index
				uint32_t level = grid_indices_host_tmp[i] / NERF_GRIDVOLUME();
//				std::cout << "Pixel " << i << "is within the AABB" << std::endl;
				//If the level is greater than the growing level, discard it
				// NOTE: should not happen with automatic level selection
				if (level > m_growing_level) {
//				std::cout << "Level is greater than the growing level: skipped: " << (level > m_growing_level) << std::endl;
					continue;
				}

				//Get the cell index
				uint32_t cell_idx = grid_indices_host_tmp[i];
				//If the level is less than the growing level, uplift the cell index
				if (level < m_growing_level) {
					cell_idx = get_upper_cell_idx(cell_idx, m_growing_level);
//					std::cout << "Updated cell_idx: " << std::endl;

				};
				//Get the level of the updated cell index
				level = cell_idx / NERF_GRIDVOLUME();
				//If the cell index is already in the set, discard it
				if (cell_idx_set.count(cell_idx) > 0) {
//					std::cout << "Skipped, already in the set " << std::endl;
					continue;
				}

				//printf("Index ray: %u (%u)\n", cell_idx, NERF_GRIDVOLUME());

				//Add the cell index, pixel, and label to their respective vectors
				m_projected_cell_idx.push_back(cell_idx);
				m_projected_pixels.push_back(m_projected_pixels_tmp[i]);
				//m_projected_features.push_back(m_projected_features_tmp[i]);
				m_projected_labels.push_back(0);
				//Add the cell index to the set
				cell_idx_set.insert(cell_idx);
//				std::cout << "Cell " << i << "added" << std::endl;
			}
	}

	//Update the n_rays variable based on the size of the m_projected_cell_idx vector
	n_rays = m_projected_cell_idx.size(); // Update n_rays accordingly
	if (n_rays == 0) {
		std::cout << "Couldn't find surface when shooting rays..." << std::endl;
		return;
	}
	//std::cout << "Reprojected " << n_rays << " rays" << std::endl;

	//Salva m_selected_pixels in una variabile apposita prima di eliminare il suo contenuto
	helper_selected_pixels helperObj;
	helperObj.set_selected_pixels(m_selected_pixels);
	
	//Clear the selected pixels and reset the growing selection
	m_selected_pixels.clear();
    m_selected_pixels_imgui.clear();

	// Reset growing if it had already been done
	reset_growing();
}

//Launched by GrowingSelection::project_selection_pixels()
void GrowingSelection::reset_growing() {

	cage_edition.selected_vertices.clear();

	render_mode = ESelectionRenderMode::ScreenSelection;
	m_performed_closing = false;

	// Make sure to reset the gizmo to TRANSLATE
	m_gizmo_op = ImGuizmo::TRANSLATE;

	// Clear the proxy and tet
	proxy_cage = Cage<float_t, point_t>();
	tet_interpolation_mesh = nullptr;

	m_region_growing.reset_growing(m_projected_cell_idx, m_growing_level);

	m_selection_points = m_region_growing.selection_points();
	m_selection_cell_idx = m_region_growing.selection_cell_idx();

	m_performed_closing = false;
}

void GrowingSelection::upscale_growing() {
	m_region_growing.upscale_selection(m_growing_level);

	m_selection_grid_bitfield = m_region_growing.selection_grid_bitfield();
	m_selection_points = m_region_growing.selection_points();
	m_selection_cell_idx = m_region_growing.selection_cell_idx();
	m_selection_labels = std::vector<uint8_t>(m_selection_points.size(), 0);
	m_growing_level = m_region_growing.growing_level();
}

//GUI Button "Grow region" and "Grow Far", first function
void GrowingSelection::grow_region(bool ed_flag, int growing_steps) {
	int cell_idx = m_projected_cell_idx.at(0);
	m_region_growing.reset_push_m_growing_queue(cell_idx);
	m_region_growing.reset_growing(m_projected_cell_idx, m_growing_level);
  
	m_region_growing.grow_region(ed_flag, m_density_threshold, m_region_growing_mode, m_growing_level, growing_steps);
	m_selection_grid_bitfield = m_region_growing.selection_grid_bitfield();
	m_selection_points = m_region_growing.selection_points();
	m_selection_cell_idx = m_region_growing.selection_cell_idx();
	m_selection_labels = std::vector<uint8_t>(m_selection_points.size(), 0);
	m_growing_level = m_region_growing.growing_level();
	//Modifiche aggiunte per avere gli stessi punti_post-scribbling e punti_post-growing
	/*if (ed_flag == true) {
		m_projected_pixels = m_region_growing.selection_points();
		m_projected_cell_idx = m_region_growing.selection_cell_idx();
		m_projected_labels = std::vector<uint8_t>(m_selection_points.size(), 0);
	}*/

	m_performed_closing = false;
}

void GrowingSelection::dilate() {
	//Aggiungere parametri alla funzione dilate per poi passarli al dilate interno
	m_selection_grid_bitfield = m_MM_operations->dilate(m_selection_grid_bitfield, m_growing_level, m_selection_points, m_selection_cell_idx);

	m_selection_labels = std::vector<uint8_t>(m_selection_points.size(), 0);
}


void GrowingSelection::erode() {
	//Aggiungere parametri alla funzione erode per poi passarli al erode interno

	m_selection_grid_bitfield = m_MM_operations->erode(m_selection_grid_bitfield, m_growing_level, m_selection_points, m_selection_cell_idx);

	m_selection_labels = std::vector<uint8_t>(m_selection_points.size(), 0);
}

// TODO: work directly on the bitfield computed before instead of float (we have a 0/1 occupancy field)
void GrowingSelection::extract_fine_mesh() {
	std::cout << "GrowingSelection::extract_fine_mesh()" << std::endl;

	// Make sure dilation erosion are performed before extracting the mesh 
	if (m_use_morphological && !m_performed_closing) {
		m_performed_closing = true;
		dilate();
		erode();
	}

	// Will be related to max level!
	// First, compute the object bounding_box and its corresponding voxel resolution
	BoundingBox object_aabb;
	Eigen::Vector3i res3d = Eigen::Vector3i::Constant(NERF_GRIDSIZE());
	const float scale = scalbnf(1.0f, m_growing_level);

	object_aabb.enlarge(scale * (Vector3f{0.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f));
	object_aabb.enlarge(scale * (Vector3f{1.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f));
	const uint32_t n_elements = (res3d.x()*res3d.y()*res3d.z());
	tcnn::GPUMemory<float> density(n_elements);
	std::vector<float> density_host(n_elements, 0.f);
	// Then, set the density accordingly with the following ordering:
	// x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
	for (const auto& current_cell : m_selection_cell_idx) {
		const uint32_t level = current_cell / (NERF_GRIDVOLUME());
		const uint32_t pos_idx = current_cell % (NERF_GRIDVOLUME());

		// Make sure we don´t have points on the boundary of the bbox
		// This is a hack because the MC algorithm fails to produce manifold meshes
		// in this situation...
		// TODO: replace the default MC algorithm with a more robust one
		if (is_boundary(pos_idx)) {
			continue;
		}

		if (level == m_growing_level) {
			// uint32_t x = tcnn::morton3D_invert(pos_idx>>0) - voxel_aabb.min.x();
			// uint32_t y = tcnn::morton3D_invert(pos_idx>>1) - voxel_aabb.min.y();
			// uint32_t z = tcnn::morton3D_invert(pos_idx>>2) - voxel_aabb.min.z();
			uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
			uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
			uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
			uint32_t density_idx = x+ y*res3d.x() + z*res3d.x()*res3d.y();
			density_host[density_idx] = 1.f;
		}
	}

	// Copy to GPU
	density.copy_from_host(density_host, n_elements);

    // Create temp GPU mesh
    MeshState selection_mesh_gpu;

	// Then, perform marching cubes
	marching_cubes_gpu(m_stream, object_aabb, res3d, 0.5f, density, selection_mesh_gpu.verts, selection_mesh_gpu.indices);

	uint32_t n_verts = (uint32_t)selection_mesh_gpu.verts.size();
	
	compute_mesh_1ring(selection_mesh_gpu.verts, selection_mesh_gpu.indices, selection_mesh_gpu.verts_smoothed, selection_mesh_gpu.vert_normals);

    // Copy to host
    selection_mesh.vertices.resize(selection_mesh_gpu.verts.size());
    selection_mesh.indices.resize(selection_mesh_gpu.indices.size());
    selection_mesh.normals.resize(selection_mesh_gpu.vert_normals.size());
    selection_mesh_gpu.verts.copy_to_host(selection_mesh.vertices);
    selection_mesh_gpu.indices.copy_to_host(selection_mesh.indices);
    selection_mesh_gpu.vert_normals.copy_to_host(selection_mesh.normals);
}

static uint32_t cubemap_map[6] {
	1, 3, 4, 2, 0, 5
};

// See: https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing
void convert_cube_uv_to_xyz(int index, float u, float v, float *x, float *y, float *z)
{
  switch (index)
  {
    case 0: *x =  1.0f; *y =    v; *z =   -u; break;	// POSITIVE X
    case 1: *x = -1.0f; *y =    v; *z =    u; break;	// NEGATIVE X
    case 2: *x =    u; *y =  1.0f; *z =   -v; break;	// POSITIVE Y
    case 3: *x =    u; *y = -1.0f; *z =    v; break;	// NEGATIVE Y
    case 4: *x =    u; *y =    v; *z =  1.0f; break;	// POSITIVE Z
    case 5: *x =   -u; *y =    v; *z = -1.0f; break;	// NEGATIVE Z
  }
}

__global__ void activate_network_output(
	const uint32_t n_elements,
	int padded_output_width,
	const tcnn::network_precision_t* network_output,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	Array3f* color,
	float* density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) { return; }

	network_output += i * padded_output_width;
	tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
	color[i] = network_to_rgb(local_network_output, rgb_activation);
	density[i] = network_to_density(float(local_network_output[3]), density_activation);
}

__global__ void filter_empty(
	const uint32_t n_elements,
	const BoundingBox aabb,
	const uint8_t* occupancy_grid,
	const PitchedPtr<NerfCoordinate> coords,
	float* density)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) { return; }

	auto coord = coords(i);
	auto pos = unwarp_position(coord->pos.p, aabb);
	int mip = mip_from_pos(pos);

	if (!density_grid_occupied_at(pos, occupancy_grid, mip))
	{
		density[i] = 0.0f;
	}
}

//is_inside indicates whether the computation should be done inside or outside the proxy cage  
void GrowingSelection::compute_poisson_boundary(const bool is_inside) {
	//Check if the proxy cage has any vertices. If not, print a message and return.
	if (proxy_cage.vertices.size() == 0) {
		std::cout << "Computing boundary values requires a proxy cage..." << std::endl;
		return;
	}
	//Set the vertices variable to either the original vertices or the proxy cage vertices.
	const std::vector<point_t>& vertices = is_inside ? proxy_cage.original_vertices : proxy_cage.vertices;
	const uint32_t n_verts = vertices.size();
	const uint32_t n_sh_samples = m_poisson_editing.sh_sampling_width *  m_poisson_editing.sh_sampling_width;

	const uint32_t padded_output_width = m_nerf_network->padded_output_width();
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct
	const uint32_t n_samples = n_verts * n_sh_samples;
	const uint32_t n_elements = next_multiple(n_samples, tcnn::batch_size_granularity); // ensure batch sizes have the right granularity

	// Create a vector called coords_host to store the sampling coordinates. 
	// Initialize sampling coords first
	std::vector<float> coords_host(n_samples * floats_per_coord);
	// Initialize a pitched pointer to this vector.
	PitchedPtr<NerfCoordinate> coords_host_ptr = PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_host.data(), 1, 0, extra_stride);
	// Loop through all the vertices and SH samples.
	for (uint32_t k = 0; k < n_verts; k++) {
		for (uint32_t i = 0; i < m_poisson_editing.sh_sampling_width; i++) {
			for (uint32_t j = 0; j < m_poisson_editing.sh_sampling_width; j++) {
				// Compute the Cartesian components for each point, and store them in the coords_host vector.
				/* We now find the cartesian components for the point (i,j) */
				float u,v,theta,phi,x,y,z;
				
				// First compute discretized 2d inputs in [0, 1]
				u = (i + (float)std::rand() / RAND_MAX) / (m_hemisphere_width);	
				v = (j + (float)std::rand() / RAND_MAX) / (m_hemisphere_width);
				
				theta = 2.f * M_PI * v;
				phi = acos(2.f*u-1.f);

				x = std::cos(theta)*std::sin(phi);
				y = std::sin(theta)*std::sin(phi);
				z = std::cos(phi);

				Eigen::Vector3f dir(x, y, z);
				coords_host_ptr(n_sh_samples * k + i * m_poisson_editing.sh_sampling_width + j)->pos.p = warp_position(vertices[k], m_aabb);
				coords_host_ptr(n_sh_samples * k + i * m_poisson_editing.sh_sampling_width + j)->dir.d = warp_direction(dir);
			}
		}
	}

	//std::cout << "Sampled coordinates..." << std::endl;

	// Allocate GPU memory for the coordinates, MLP output, RGB output, and density output. 
	tcnn::GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		float, // coords
		network_precision_t, // mlp_out
		Array3f, // rgb_out
		float // density_out
	>(
		m_stream, &alloc,
		n_elements * floats_per_coord,
		n_elements * padded_output_width,
		n_elements,
		n_elements
	);

	float* coords = std::get<0>(scratch);
	network_precision_t* mlp_out = std::get<1>(scratch);
	Array3f* rgb_out = std::get<2>(scratch);
	float* density_out = std::get<3>(scratch);

	// Copy coords to GPU and prepare containers
	CUDA_CHECK_THROW(cudaMemcpyAsync(coords, coords_host.data(), n_samples*floats_per_coord*sizeof(float), cudaMemcpyHostToDevice, m_stream));
	GPUMatrix<float> coord_matrix((float*)coords, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
	GPUMatrix<network_precision_t> rgbsigma_matrix((network_precision_t*)mlp_out, padded_output_width, n_elements);
	
	// Perform inference on the neural network with the input coordinates and store the output in the rgbsigma_matrix.
	m_nerf_network->inference_mixed_precision(m_stream, coord_matrix, rgbsigma_matrix);

	//std::cout << "Performed inference..." << std::endl;

	// Activate the network output using a linear kernel function and store the results in the RGB and density outputs. 
	linear_kernel(activate_network_output, 0, m_stream,
		n_samples,
		padded_output_width,
		mlp_out,
		m_rgb_activation,
		m_density_activation,
		rgb_out,
		density_out);

	// Filter out empty samples using another linear kernel function.
	if (is_inside)
	{
		linear_kernel(filter_empty, 0, m_stream,
			n_samples,
			m_aabb,
			m_density_grid_bitfield.data(),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			density_out);
	}

	// std::cout << "Activated output..." << std::endl;

	std::vector<Array3f> rgb_host(n_samples);
	std::vector<float> density_host(n_samples);

	// Copy back RGB and density outputs to host
	CUDA_CHECK_THROW(cudaMemcpyAsync(rgb_host.data(), rgb_out, n_samples*sizeof(Eigen::Array3f), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK_THROW(cudaMemcpyAsync(density_host.data(), density_out, n_samples*sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));

	// Store the density values in the appropriate target density vector (inside or outside the proxy cage).
	std::vector<float>& target_density = is_inside ? proxy_cage.inside_density : proxy_cage.outside_density;
	target_density.resize(n_verts);
	for (int k = 0; k < n_verts; k++) {
		target_density[k] = density_host[k*n_sh_samples];
		// proxy_cage.colors[k] = (1 - std::exp(-proxy_cage.boundary_density[k] * MIN_CONE_STEPSIZE())) * Eigen::Vector3f(1.0f, 0.f, 0.f);
		// std::cout << "Alpha: " << 1 - std::exp(-target_density[k] * MIN_CONE_STEPSIZE()) << std::endl;
		// std::cout << "Density: " << target_density[k] << std::endl;
	}

	// Resize the target spherical harmonics (SHs) vector 
	// and loop through all the vertices to compute the averaged SHs for each vertex.
	// Fit SHs with samples
	std::vector<SH9RGB>& target_shs = is_inside ? proxy_cage.inside_shs : proxy_cage.outside_shs;
	target_shs.resize(n_verts);
	for (int k = 0; k < n_verts; k++) {
		SH9RGB averaged_sh = SH9RGB::Zero();
		for (int i = 0; i < n_sh_samples; i++) {
			averaged_sh += project_sh9(unwarp_direction(coords_host_ptr(k*n_sh_samples+i)->dir.d), rgb_host[k*n_sh_samples+i]);
		}
		averaged_sh *= 4*M_PI / (n_sh_samples);
		// Store the computed SHs in the target SHs vector.
		target_shs[k] = averaged_sh;
		// std::cout << "Evaluated color: " << evaluate_sh9(averaged_sh, Eigen::Vector3f(0.f, 0.f, 1.f)) << std::endl;
		// proxy_cage.colors[k] = evaluate_sh9(averaged_sh, Eigen::Vector3f(0.f, 0.f, 1.f));
	}

	// std::cout << "Stored density and SHs..." << std::endl;
}

void GrowingSelection::interpolate_poisson_boundary() {
	if (!tet_interpolation_mesh || tet_interpolation_mesh->vertices.size() == 0) {
		std::cout << "Updating poisson MVC-interpolated tet values requires a valid tet mesh" << std::endl;
		return;
	}

	if (proxy_cage.inside_shs.size() != proxy_cage.vertices.size()) {
		compute_poisson_boundary(true);
		// std::cout << "Updating inside shs" << std::endl;
	}

	// In any case, re-compute the outside values
	compute_poisson_boundary(false);

	uint32_t n_tet_vertices = tet_interpolation_mesh->vertices.size();
	uint32_t n_cage_vertices = proxy_cage.vertices.size();
	tet_interpolation_mesh->boundary_shs_gpu.resize(n_tet_vertices);
	tet_interpolation_mesh->boundary_residual_density_gpu.resize(n_tet_vertices);
	tet_interpolation_mesh->boundary_outside_density_gpu.resize(n_tet_vertices);
	std::vector<SH9RGB> boundary_shs_host(n_tet_vertices, SH9RGB::Zero());
	std::vector<float> boundary_residual_density_host(n_tet_vertices, 0.f);
	std::vector<float> boundary_outside_density_host(n_tet_vertices, 0.f);
	std::vector<float> boundary_inside_density_host(n_tet_vertices, 0.f);
	for (int i = 0; i < n_tet_vertices; i++) {
		float sh_weights_sum = 0.f;
		float sh_weights_in_sum = 0.f;
		for (int j = 0; j < n_cage_vertices; j++) {
			float alpha_out = 1 - std::exp(- proxy_cage.outside_density[j] * MIN_CONE_STEPSIZE());
			float alpha_in =  1 - std::exp(- proxy_cage.inside_density[j] * MIN_CONE_STEPSIZE());

			// Outside always take the lead
			float w_outside = 1.f;
			float w_inside = std::min(alpha_in / alpha_out, 1.f);
			// Weight the sh-diff by each density and renormalize
			SH9RGB sh_diff = w_outside*proxy_cage.outside_shs[j] - w_inside*proxy_cage.inside_shs[j];
			sh_weights_sum += tet_interpolation_mesh->gamma_coordinates[i][j] * alpha_out;
			sh_weights_in_sum += tet_interpolation_mesh->gamma_coordinates[i][j] * alpha_in;
			boundary_shs_host[i] += tet_interpolation_mesh->gamma_coordinates[i][j] * alpha_out * sh_diff;
			boundary_outside_density_host[i] += tet_interpolation_mesh->gamma_coordinates[i][j] * proxy_cage.outside_density[j];
			boundary_residual_density_host[i] += tet_interpolation_mesh->gamma_coordinates[i][j] * (proxy_cage.outside_density[j] - proxy_cage.inside_density[j]);
		}

		boundary_shs_host[i] /= sh_weights_sum + 1e-6;
		boundary_residual_density_host[i] = std::max(boundary_residual_density_host[i], 0.f);

	}
	tet_interpolation_mesh->boundary_shs_gpu.copy_from_host(boundary_shs_host);
	tet_interpolation_mesh->boundary_residual_density_gpu.copy_from_host(boundary_residual_density_host);
	tet_interpolation_mesh->boundary_outside_density_gpu.copy_from_host(boundary_outside_density_host);

	// m_debug_points.clear();
	// m_debug_colors.clear();
	// for (int i = 0; i < n_tet_vertices; i++) {
	// 	m_debug_points.push_back(tet_interpolation_mesh->vertices[i]);
	// 	m_debug_colors.push_back(evaluate_sh9(boundary_shs_host[i], Eigen::Vector3f(0.f, 0.f, 1.f)));
	// }


	// std::cout << "Updated poisson MVC-interpolated tet values residuals..." << std::endl;
}

void GrowingSelection::generate_poisson_cube_map() {
	if (proxy_cage.inside_shs.size() == 0) {
		compute_poisson_boundary(true);
		// std::cout << "Computing SHs to generate a cube map..." << std::endl;
	}

	if (proxy_cage.inside_shs.size() == 0) {
		std::cout << "Boundary SHs are required to display associated cube maps..." << std::endl;
		return;	
	}

	uint32_t debug_vertex_idx = cage_edition.selected_vertices.size() == 0 ? m_debug_ray_idx : cage_edition.selected_vertices[0];

	SH9RGB chosen_sh = proxy_cage.inside_shs[debug_vertex_idx];

	// For each face:
	for (int f = 0; f < 6; f++) {
		// Create a OpenGL texture identifier
		glBindTexture(GL_TEXTURE_2D, m_poisson_editing.sh_cubemap_textures[f]);

		// Setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

		// Generate the corresponding image
		uint8_t rgb_image[DEBUG_CUBEMAP_WIDTH][DEBUG_CUBEMAP_WIDTH][3];
		for (int i = 0; i < DEBUG_CUBEMAP_WIDTH; i++) {
			for (int j = 0; j < DEBUG_CUBEMAP_WIDTH; j++) {
				float v = (j - DEBUG_CUBEMAP_WIDTH/2.0)/(DEBUG_CUBEMAP_WIDTH/2.0);  
				float u = (i - DEBUG_CUBEMAP_WIDTH/2.0)/(DEBUG_CUBEMAP_WIDTH/2.0);
				Eigen::Vector3f dir;
				convert_cube_uv_to_xyz(cubemap_map[f], u, v, &(dir.x()), &(dir.y()), &(dir.z()));
				dir.normalize();

				Eigen::Vector3f rgb = evaluate_sh9(chosen_sh, dir);
				rgb_image[i][j][0] = (uint8_t)tcnn::clamp(rgb.x() * 255, 0.f, 255.f);
				rgb_image[i][j][1] = (uint8_t)tcnn::clamp(rgb.y() * 255, 0.f, 255.f);
				rgb_image[i][j][2] = (uint8_t)tcnn::clamp(rgb.z() * 255, 0.f, 255.f);
				
			}
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, DEBUG_CUBEMAP_WIDTH, DEBUG_CUBEMAP_WIDTH, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_image);
	
	}	
}
//Utilizzato dallo Split Button
void GrowingSelection::set_render_mode_to_PROJ() {
	render_mode = ESelectionRenderMode::Projection;
}

std::vector<Eigen::Vector2i> GrowingSelection::mega_scribble(std::vector<Eigen::Vector2i> arg_m_selected_pixels, Eigen::Vector2i resolution, Vector2f focal_length, Vector2f screen_center, Eigen::Matrix<float, 3, 4> camera_matrix) {
	helper_selected_pixels helperObj;

	if (arg_m_selected_pixels.size() > 0) {
			m_selected_pixels = arg_m_selected_pixels;
		}
		project_selection_pixels(m_selected_pixels, resolution, focal_length, camera_matrix, screen_center, m_stream);
		//std::cout << "project_selection_pixels DONE!" << std::endl;
		
		if (m_projected_cell_idx.size() > 0) render_mode = ESelectionRenderMode::Projection;
		return helperObj.get_selected_pixels();
}

//Restituisce l'indice di una cella superficiale casuale in m_selected_pixels
//TODO: fare in modo che restituisca l'indice di una cella distante dall'ultima cella 
int GrowingSelection::random_index_in_selected_pixels(){
	std::random_device rd;
	std::mt19937 gen(rd());
	int randomIndex;

	//Estrae un numero casuale compreso tra 0 e m_projected_pixels.size()											
	std::uniform_int_distribution<int> distribution(0, m_projected_pixels.size() - 1);
	randomIndex = distribution(gen);
	//std::cout << "randomIndex: " << randomIndex << std::endl;
	return randomIndex;
}

void GrowingSelection::remove_but_one(int randomIndex) {
	Eigen::Vector3f one_projection_point = m_projected_pixels[randomIndex];
	uint32_t one_projection_idx= m_projected_cell_idx[randomIndex];

	//Rimuove tutti gli altri punti per precauzione
	m_projected_pixels.clear();
	m_projected_labels.clear();
	m_projected_cell_idx.clear();
	//Aggiunge solo il punto desiderato
	m_projected_pixels.push_back(one_projection_point);
	m_projected_labels.push_back(0);          
	m_projected_cell_idx.push_back(one_projection_idx);
		
	//Lasciare solo 1 punto tra quelli che vengono utilizzati per la costruzione della cage
	m_selection_points.clear();
	m_selection_labels.clear();
	m_selection_cell_idx.clear();
	m_selection_points.push_back(one_projection_point);
	m_selection_labels.push_back(0);
	m_selection_cell_idx.push_back(one_projection_idx);
	only_point = one_projection_point;

	//Stampa di debug, da rimuovere
	std::cout << "one_selection_point: " << one_projection_idx << std::endl;
	std::cout << "one_selection_point: " << one_projection_point << std::endl;
}

void GrowingSelection::grow_and_cage() {
	grow_region(false, get_m_growing_steps());
	render_mode = ESelectionRenderMode::RegionGrowing;
	
	if (m_selection_grid_bitfield.size() > 0){
			fix_proxy_mesh();
			update_tet_mesh();
			interpolate_poisson_boundary();
	}
}

//Modifica il flag che permette le modifiche automaticamente quando è true
void GrowingSelection::set_apply_all_edits_flag(bool value){
	std::cout << "Current Operator apply_all_edits_flag from : " << (apply_all_edits_flag ? "True" : "False") << " to ";

	if (value == true) {            //AGGIUNGERE CHECK CHE CONTROLLA ESISTENZA DELLA CAGE, (evitabile date le circostanze)
		apply_all_edits_flag = true;
		std::cout << (apply_all_edits_flag ? "True" : "False") << std::endl;
	}else {
		apply_all_edits_flag = false;
		std::cout << "Disabled  " << std::endl;
	}
}

void GrowingSelection::to_json(nlohmann::json& j) {

	j["projected_pixels"] = m_projected_pixels;
	j["projected_labels"] = m_projected_labels;
	j["projected_cell_idx"] = m_projected_cell_idx;
	// j["projected_features"] = m_projected_features;

	j["selection_points"] = m_selection_points;
	j["selection_labels"] = m_selection_labels;
	j["selection_cell_idx"] = m_selection_cell_idx;
	j["m_selection_grid_bitfield"] = m_selection_grid_bitfield;

	j["growing_level"] = m_growing_level;
	j["region_growing"] = m_region_growing.to_json();

	j["selection_mesh"] = selection_mesh;
	j["proxy_cage"] = proxy_cage;

	if (tet_interpolation_mesh) {
		j["interpolation_mesh"] = *tet_interpolation_mesh;
	}
}

NGP_NAMESPACE_END

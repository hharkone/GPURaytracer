#include <glm/gtc/type_ptr.hpp>
#include <deque>
#include <string>
#include <iostream>
#include <filesystem>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "imoguizmo.hpp"
#include "Renderer.h"
#include "Camera.h"
#include "cuda_runtime.h"
/*
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image.h"
#include "stb_image_write.h"
#include "zlib.h"
#include "tinyexr.h"
*/
using namespace Walnut;

class ExampleLayer : public Walnut::Layer
{
public:

	ExampleLayer() : m_camera(50.0f, 0.1f, 1000000.0f)
	{
		m_rendetTimeVec.resize(20);
		cudaGetDeviceProperties(&prop, 0);
	}

	virtual void OnAttach() override
	{
		std::string path = "Images";
		for (const auto& entry : std::filesystem::directory_iterator(path))
		{
			if (entry.is_regular_file() && entry.path().extension() == ".exr")
			{
				m_exrFilePaths.push_back(entry);
			}
		}

		if (m_exrFilePaths.at(0).exists())
		{
			m_scene.envImgPath = m_exrFilePaths.at(0).path().string();
			m_scene.envImgPathChanged = true;
			m_sceneChanged = true;
		}
	}

	virtual void OnUpdate(float ts) override
	{
		if (m_camera.OnUpdate(ts))
		{
			m_renderer.ResetFrameIndex();
		}
	}

	virtual void OnUIRender() override
	{
		ImGui::ShowDemoWindow();
		ImGuiIO& io = ImGui::GetIO();
		io.FontGlobalScale = 0.8f;
		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowPadding = { 5.0f, 5.0f };
		style.FramePadding = { 4.0f, 2.0f };
		style.GrabMinSize = 5.0f;
		style.FrameRounding = 2.0f;
		style.GrabRounding = 4.0f;

		ImGuiColorEditFlags misc_flags = ImGuiColorEditFlags_HDR;
		static ImGuiSliderFlags flags = ImGuiSliderFlags_None;
		static ImGuiSliderFlags flagLog = ImGuiSliderFlags_Logarithmic;
		ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_DisplayHSV);

		ImGui::Begin("Settings");
		ImGui::Text("CUDA Device: %s", prop.name);
		ImGui::Text("Last render time: %.3f ms   FPS: %i", m_renderTimeMs, (1000u / std::max((uint32_t)m_renderTimeMs, 1u)));
		ImGui::Text("%i Million primary rays per second", m_raysPerSec);
		ImGui::Text("Sample Index: %i", m_renderer.GetFrameIndex());

		const char* resolutionFactors[] = { "1:1", "1:2", "1:3", "1:4" };
		static int resolutionFactorIndex = 1;
		ImGui::Combo("Resolution", &resolutionFactorIndex, resolutionFactors, IM_ARRAYSIZE(resolutionFactors));
		ImGui::Text("(%i x %i)", m_viewportWidth, m_viewportHeight);

		ImGui::Checkbox("Accumulate", &m_renderer.GetSettings().accumulate);
		ImGui::Checkbox("Use OPTIX Denoiser", &m_renderer.GetSettings().denoise);
		if (ImGui::SliderInt("Max Bounces", &m_renderer.GetSettings().bounces, 1, 30)) { m_sceneChanged = true; }
		if (ImGui::SliderInt("BVH Debug", &m_renderer.GetSettings().debug, 0, 64)) { m_sceneChanged = true; }
		if (ImGui::Checkbox("BVH Debug Enable", &m_renderer.GetSettings().bvhDebug)) { m_sceneChanged = true; }

		if (ImGui::Button("Reset") || m_sceneChanged)
		{
			m_renderer.ResetFrameIndex();
			m_scene.UploadMaterials();
			m_sceneChanged = false;
		}

		if (ImGui::Button("Save Render"))
		{
			m_renderer.SaveRenderToDisk("Renderoutput/render0.exr");
		}

		ImGui::End();
		
		ImGui::Begin("Camera");
		ImGui::Text("Camera POS: %.2f, %.2f, %.2f", m_camera.GetPosition().x, m_camera.GetPosition().y, m_camera.GetPosition().z);
		ImGui::SliderFloat("Camera Speed", &m_camera.GetSpeed(), 0.01f, 100.0f, "%.3f", flagLog);
		if (ImGui::DragFloat("Camera FOV", &m_camera.GetFOV(), 0.1f, 0.01f, 179.0f)) { m_sceneChanged = true; m_camera.RecalculateProjection(); }
		if (ImGui::DragFloat("Aperture", &m_camera.m_aperture, 0.001f, 0.0f, 1.0f)) { m_sceneChanged = true; }
		if (ImGui::DragFloat("Focus Distance", &m_camera.m_focusDistance, 0.01f, 0.01f, 1000.0f)) { m_sceneChanged = true; }

		ImGui::End();

		ImGui::Begin("Scene");
		ImGui::Text("Models:");
		ImGui::AlignTextToFramePadding();
		if (ImGui::DragFloat4("Matrix[0]", &m_scene.sceneMesh.transformMatrix[0][0], 0.01f)) { m_sceneChanged = true; m_scene.sceneMesh.ApplyTransform(); }
		if (ImGui::DragFloat4("Matrix[1]", &m_scene.sceneMesh.transformMatrix[1][0], 0.01f)) { m_sceneChanged = true; m_scene.sceneMesh.ApplyTransform(); }
		if (ImGui::DragFloat4("Matrix[2]", &m_scene.sceneMesh.transformMatrix[2][0], 0.01f)) { m_sceneChanged = true; m_scene.sceneMesh.ApplyTransform(); }
		if (ImGui::DragFloat4("Matrix[3]", &m_scene.sceneMesh.transformMatrix[3][0], 0.01f)) { m_sceneChanged = true; m_scene.sceneMesh.ApplyTransform(); }

		ImGui::Text("Spheres: ", m_scene.sphereCount);
		for (size_t i = 0u; i < m_scene.sphereCount; i++)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Sphere& sphere = m_scene.spheresSimple[i];
			int dragInt = (int)sphere.materialIndex;

			ImGui::Text("Sphere: %i", i);
			if (ImGui::DragFloat("Radius", &sphere.rad, 0.01f)) { m_sceneChanged = true; }
			if (ImGui::DragFloat3("Position", &sphere.pos.x, 0.01f)) { m_sceneChanged = true; }
			if (ImGui::SliderInt("Material ID", &dragInt, 0, (int)m_scene.materialCount-1)) { m_sceneChanged = true; }
			sphere.materialIndex = (uint16_t)dragInt;

			ImGui::Separator();
			ImGui::PopID();
		}

		ImGui::Text("Boxes: ", m_scene.boxCount);
		for (size_t i = 0u; i < m_scene.boxCount; i++)
		{
			ImGui::PushID((int)i + (int)m_scene.sphereCount);
			ImGui::AlignTextToFramePadding();

			Box& box = m_scene.boxSimple[i];
			int dragInt = (int)box.materialIndex;

			ImGui::Text("Box: %i", i);
			if (ImGui::DragFloat3("Size", &box.size.x, 0.01f)) { m_sceneChanged = true; }
			if (ImGui::DragFloat3("Position", &box.pos.x, 0.01f)) { m_sceneChanged = true; }
			if (ImGui::SliderInt("Material ID", &dragInt, 0, (int)m_scene.materialCount - 1)) { m_sceneChanged = true; }
			box.materialIndex = (uint16_t)dragInt;

			ImGui::Separator();
			ImGui::PopID();
		}

		ImGui::End();

		ImGui::Begin("Scene Settings");
		static int item_current = 2;
		if (ImGui::Combo("Environment Type", &item_current, "Solid\0Procedural Sky\0HDRI\0\0"))
		{
			m_scene.envType = EnvironmentType(item_current);
			m_sceneChanged = true;
		}
		ImGui::Text("");
		ImGui::Separator();

		if (item_current == static_cast<int>(EnvironmentType::EnvType_ProceduralSky))
		{
			if (ImGui::SliderFloat("Background Brightness", &m_scene.backgroundBrightness, 0.0f, 1.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color", &(m_scene.skyColor.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sky Brightness", &m_scene.skyBrightness, 0.0f, 10.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sun Focus", &m_scene.sunFocus, 1.0f, 100000.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sun Intensity", &m_scene.sunIntensity, 0.0f, 100.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat3("Sun Direction", &(m_scene.sunDirection.x), -1.0f, 1.0f)) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color Horizon", &(m_scene.skyColorHorizon.x))) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color Zenith", &(m_scene.skyColorZenith.x))) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Ground Color", &(m_scene.groundColor.x))) { m_sceneChanged = true; }
		}
		else if (item_current == static_cast<int>(EnvironmentType::EnvType_Solid))
		{
			if (ImGui::SliderFloat("Background Brightness", &m_scene.backgroundBrightness, 0.0f, 1.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color", &(m_scene.skyColor.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sky Brightness", &m_scene.skyBrightness, 0.0f, 10.0f, "%.3f", flags)) { m_sceneChanged = true; }
		}
		else
		{
			static int exr_item_index = 0;
			const char* combo_exr_name = _strdup(m_exrFilePaths.at(exr_item_index).path().string().c_str());

			if (ImGui::BeginCombo("EXR Image", combo_exr_name, ImGuiComboFlags_PopupAlignLeft))
			{
				for (int n = 0; n < m_exrFilePaths.size(); n++)
				{
					const bool is_selected = (exr_item_index == n);
					if (ImGui::Selectable(m_exrFilePaths.at(n).path().string().c_str(), is_selected))
					{
						exr_item_index = n;
						m_scene.envImgPath = m_exrFilePaths.at(n).path().string();
						m_scene.envImgPathChanged = true;
						m_sceneChanged = true;
					}

					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::Separator();

			if (ImGui::SliderFloat("Background Brightness", &m_scene.backgroundBrightness, 0.0f, 1.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color", &(m_scene.skyColor.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sky Brightness", &m_scene.skyBrightness, 0.0f, 10.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sky Rotation", &m_scene.skyRotation, 0.0f, 360.0f, "%.3f", flags)) { m_sceneChanged = true; }
		}

		ImGui::Text("");
		ImGui::Separator();

		ImGui::Text("Tonemapper");
		ImGui::DragFloat("A", &m_scene.tonemap.A, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("B", &m_scene.tonemap.B, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("C", &m_scene.tonemap.C, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("D", &m_scene.tonemap.D, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("E", &m_scene.tonemap.E, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("F", &m_scene.tonemap.F, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("W", &m_scene.tonemap.W, 0.01f, 0.0f, 10.0f);
		ImGui::DragFloat("Exposure", &m_scene.tonemap.Exposure, 0.01f, 0.0f, 20.0f);

		ImGui::End();

		ImGui::Begin("Materials");
		for (size_t i = 1u; i < m_scene.materialCount; i++) //Skip first material index (Air)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Material& mat = m_scene.materials[i];

			if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_None))
			{
				ImGui::Text("Material: %i", i);
				ImGui::Text("Surface");
				if (ImGui::ColorEdit3("Albedo", &(mat.albedo.x))) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Vertex Color Influence", &(mat.vcolor), 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Metalness", &(mat.metalness), 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Roughness", &(mat.roughness), 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("IOR", &mat.ior, 1.0f, 32.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
				ImGui::Text("Emission");
				if (ImGui::ColorEdit3("Emission", &(mat.emission.x))) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Emission Intensity", &(mat.emissionIntensity), 0.0f, 1000.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
				ImGui::Text("Transmission");
				if (ImGui::ColorEdit3("Transmission Color", &(mat.transmissionColor.x))) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Transmission", &mat.transmission, 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Transmission Roughness", &mat.transmissionRoughness, 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Transmission Aberration", &mat.transmissionAberration, 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::DragFloat("Transmission Density", &mat.transmissionDensity, 0.01f, 0.0f, 100.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Inscatter", &mat.transmissionInscatter, 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				if (ImGui::SliderFloat("Inscatter Anisotropy", &mat.transmissionInscatterAnisotropy, -1.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
				ImGui::Text("");
				ImGui::Separator();
				ImGui::Separator();
				
			}
			else
			{
				ImGui::SameLine();
				ImGui::Text("Material: %i", i);
				ImGui::SameLine();
				if (ImGui::ColorEdit3("", &(mat.albedo.x), ImGuiColorEditFlags_NoInputs)) { m_sceneChanged = true; }
			}

			ImGui::PopID();
		}
		ImGui::End();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Viewport");

		m_viewportWidth = (uint32_t)ImGui::GetContentRegionAvail().x / (resolutionFactorIndex + 1);
		m_viewportHeight = (uint32_t)ImGui::GetContentRegionAvail().y / (resolutionFactorIndex + 1);
		uint32_t actualHeight = (uint32_t)ImGui::GetContentRegionAvail().y;
		// (resolutionFactorIndex + 1)

		auto image = m_renderer.GetFinalImage();

		if (image)
		{
			ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth() * (float)(resolutionFactorIndex + 1), (float)image->GetHeight() * (float)(resolutionFactorIndex + 1) });
			//inverted
			//ImGui::Image(image->GetDescriptorSet(),
			//	{ (float)image->GetWidth(), (float)image->GetHeight() },
			//	ImVec2(0, 1), ImVec2(1, 0));
		}

		bool cameraControls = (/*ImGui::IsWindowHovered() &&*/ io.MouseDown[1]);
		m_camera.SetIsContextFocused(cameraControls);

		// it is recommended to use a separate projection matrix since the values that work best
		// can be very different from what works well with normal renderings
		// e.g., with glm -> glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
		glm::mat4 gizmoProjection = glm::perspective(glm::radians(60.0f), 1.0f, 1.0f, 1000.0f);

		// optional: configure color, axis length and more
		ImOGuizmo::config.axisLengthScale = 1.5f;
		ImOGuizmo::config.lineThicknessScale = 0.12f;
		ImOGuizmo::config.hoverCircleRadiusScale = 1.0f;
		ImOGuizmo::config.negativeRadiusScale = 0.5f;
		ImOGuizmo::config.positiveRadiusScale = 0.5f;

		// specify position and size of gizmo (and its window when using ImOGuizmo::BeginFrame())
		const float widgetSize = 16.0f;
		const float offset = widgetSize * 2.0f + 32.0f;

		ImOGuizmo::SetRect(ImGui::GetWindowPos().x + offset, ImGui::GetWindowPos().y - offset + actualHeight, widgetSize);
		//ImOGuizmo::BeginFrame(); // to use you own window remove this call 
		// and wrap everything in between ImGui::Begin() and ImGui::End() instead

		// optional: set distance to pivot (-> activates interaction)
		glm::mat4 viewMat = m_camera.GetView();
		if(ImOGuizmo::DrawGizmo(&(viewMat[0][0]), &gizmoProjection[0][0], 3.0f /* optional: default = 0.0f */))
		{
			// in case of user interaction viewMatrix gets updated
			m_camera.SetView(viewMat);
			m_sceneChanged = true;
		}

		ImGui::End();
		ImGui::PopStyleVar();

		Render();
	}

	void Render()
	{
		Timer timer;

		m_renderer.OnResize(m_scene, m_viewportWidth, m_viewportHeight);
		m_camera.OnResize(m_viewportWidth, m_viewportHeight);

		if (m_scene.envImgPathChanged)
		{
			m_scene.envImgPathChanged = false;
			m_renderer.LoadHDRI(m_scene);
		}
		
		m_renderer.Render(m_scene, m_camera);

		m_rendetTimeVec.push_front(timer.ElapsedMillis());
		m_rendetTimeVec.pop_back();

		float sum = 0.0f;
		for(size_t i = 0; i < m_rendetTimeVec.size(); i++)
		{
			sum += m_rendetTimeVec.at(i);
		}
		m_renderTimeMs = sum / m_rendetTimeVec.size();

		if (m_renderTimeMs > 0.0f)
		{
			m_raysPerSec = uint32_t((m_viewportWidth * m_viewportHeight) * (1000.0 / m_renderTimeMs) / 1000000);
			if (!m_renderer.GetSettings().accumulate)
			{
				m_raysPerSec = 0;
			}
		}
	}

private:
	Renderer m_renderer;
	Camera m_camera;
	float m_renderTimeMs = 0.0f;
	std::deque<float> m_rendetTimeVec;
	uint32_t m_viewportWidth = 0, m_viewportHeight = 0;
	uint32_t m_raysPerSec = 0;
	Scene m_scene;
	cudaDeviceProp prop;
	bool m_sceneChanged = false;
	std::vector<std::filesystem::directory_entry> m_exrFilePaths;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "CUDA Raytracer";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}
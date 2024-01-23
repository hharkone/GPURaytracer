#include <glm/gtc/type_ptr.hpp>
#include <deque>


#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "Renderer.h"
#include "Camera.h"
#include "cuda_runtime.h"


using namespace Walnut;

class ExampleLayer : public Walnut::Layer
{
public:

	ExampleLayer() : m_camera(50.0f, 0.1f, 1000000.0f)
	{
		m_rendetTimeVec.resize(20);
		cudaGetDeviceProperties(&prop, 0);
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
		//ImGui::ShowDemoWindow();
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

		ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_PickerHueWheel);

		ImGui::Begin("Settings");
		ImGui::Text("CUDA Device: %s", prop.name);
		ImGui::Text("Last render time: %.3f ms", m_renderTimeMs);
		ImGui::Text("%i Million primary rays per second", m_raysPerSec);
		ImGui::Text("Sample Index: %i", m_renderer.GetFrameIndex());

		ImGui::Checkbox("Accumulate", &m_renderer.GetSettings().accumulate);
		ImGui::Checkbox("Use OPTIX Denoiser", &m_renderer.GetSettings().denoise);
		if (ImGui::SliderInt("Max Bounces", &m_renderer.GetSettings().bounces, 0, 30)) { m_sceneChanged = true; }
		if (ImGui::SliderInt("BVH Debug", &m_renderer.GetSettings().samples, 0, 1000)) { m_sceneChanged = true; }

		if (ImGui::Button("Reset") || m_sceneChanged)
		{
			m_renderer.ResetFrameIndex();
			m_sceneChanged = false;
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
		ImGui::End();

		ImGui::Begin("Scene Settings");
		static int item_current = 1;
		if (ImGui::Combo("Environment Type", &item_current, "Solid\0Procedural Sky\0\0"))
		{
			m_scene.envType = EnvironmentType(item_current);
			m_sceneChanged = true;
		}
		ImGui::Text("");
		if (item_current == 0)
		{
			if (ImGui::ColorEdit3("Sky Color", &(m_scene.skyColor.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sky Brightness", &m_scene.skyBrightness, 0.0f, 10.0f, "%.3f", flags)) { m_sceneChanged = true; }
		}
		else
		{
			if (ImGui::ColorEdit3("Sky Color", &(m_scene.skyColor.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sky Brightness", &m_scene.skyBrightness, 0.0f, 10.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sun Focus", &m_scene.sunFocus, 1.0f, 100000.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Sun Intensity", &m_scene.sunIntensity, 0.0f, 100.0f, "%.3f", flags)) { m_sceneChanged = true; }
			if (ImGui::SliderFloat3("Sun Direction", &(m_scene.sunDirection.x), -1.0f, 1.0f)) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color Horizon", &(m_scene.skyColorHorizon.x))) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Sky Color Zenith", &(m_scene.skyColorZenith.x))) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Ground Color", &(m_scene.groundColor.x))) { m_sceneChanged = true; }
		}

		ImGui::Text("");
		ImGui::Separator();
		ImGui::Separator();

		ImGui::Text("Tonemapper");
		ImGui::DragFloat("A", &m_scene.A, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("B", &m_scene.B, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("C", &m_scene.C, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("D", &m_scene.D, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("E", &m_scene.E, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("F", &m_scene.F, 0.001f, 0.0f, 10.0f);
		ImGui::DragFloat("W", &m_scene.W, 0.01f, 0.0f, 10.0f);
		ImGui::DragFloat("Exposure", &m_scene.Exposure, 0.01f, 0.0f, 20.0f);

		ImGui::End();

		ImGui::Begin("Materials");
		for (size_t i = 0u; i < m_scene.materialCount; i++)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Material& mat = m_scene.materials[i];

			ImGui::Text("Material: %i", i);
			ImGui::Text("Surface");
			if (ImGui::ColorEdit3("Albedo", &(mat.albedo.x))) { m_sceneChanged = true; }
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
			if (ImGui::SliderFloat("Transmission Density", &mat.transmissionDensity, 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
			ImGui::Text("");
			ImGui::Separator();
			ImGui::Separator();
			ImGui::PopID();
		}
		ImGui::End();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Viewport");

		m_viewportWidth = (uint32_t)ImGui::GetContentRegionAvail().x;
		m_viewportHeight = (uint32_t)ImGui::GetContentRegionAvail().y;

		auto image = m_renderer.GetFinalImage();

		if (image)
		{
			ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() } );
			//inverted
			//ImGui::Image(image->GetDescriptorSet(),
			//	{ (float)image->GetWidth(), (float)image->GetHeight() },
			//	ImVec2(0, 1), ImVec2(1, 0));
		}

		bool cameraControls = (/*ImGui::IsWindowHovered() &&*/ io.MouseDown[1]);
		if (cameraControls)
		{
			m_camera.SetIsContextFocused(true);
		}
		else
		{
			m_camera.SetIsContextFocused(false);
		}

		ImGui::End();
		ImGui::PopStyleVar();

		Render();
	}

	void Render()
	{
		Timer timer;

		m_renderer.OnResize(m_viewportWidth, m_viewportHeight);
		m_camera.OnResize(m_viewportWidth, m_viewportHeight);
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
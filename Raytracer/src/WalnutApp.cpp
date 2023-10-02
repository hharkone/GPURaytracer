#include <glm/gtc/type_ptr.hpp>
#include <deque>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "Renderer.h"
#include "Camera.h"

using namespace Walnut;

class ExampleLayer : public Walnut::Layer
{
public:

	ExampleLayer()
		: m_camera(50.0f, 0.1f, 1000.0f)
	{
		m_rendetTimeVec.resize(20);
		/*
		{
			Sphere sphere;
			sphere.position = { -9.0f, 1.0f, 0.0f };
			sphere.radius = 1.0f;
			sphere.materialIndex = 4;

			m_scene.spheres.push_back(sphere);
		}
		{
			Sphere sphere;
			sphere.position = { -6.0f, 1.0f, 0.0f };
			sphere.radius = 1.0f;
			sphere.materialIndex = 5;

			m_scene.spheres.push_back(sphere);
		}
		{
			Sphere sphere;
			sphere.position = { -3.0f, 1.0f, 0.0f };
			sphere.radius = 1.0f;
			sphere.materialIndex = 6;

			m_scene.spheres.push_back(sphere);
		}
		{
			Sphere sphere;
			sphere.position = { -0.0f, 1.0f, 0.0f };
			sphere.radius = 1.0f;
			sphere.materialIndex = 7;

			m_scene.spheres.push_back(sphere);
		}
		{
			Sphere sphere;
			sphere.position = { 3.0f, 1.0f, 0.0f };
			sphere.radius = 1.0f;
			sphere.materialIndex = 8;

			m_scene.spheres.push_back(sphere);
		}
		{
			Material mat;
			mat.albedo = { 0.7f, 0.7f, 0.7f };
			mat.roughness = 0.2f;
			mat.name = "White";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.8f, 0.3f, 0.3f };
			mat.roughness = 0.2f;
			mat.name = "Red";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.4f, 0.8f, 0.4f };
			mat.roughness = 0.2f;
			mat.name = "Green";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.0f, 0.0f, 0.0f };
			mat.roughness = 0.2f;
			mat.emissionColor = { 1.0f, 1.0f, 1.0f };
			mat.emissionPower = 10.0f;
			mat.name = "Emissive";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.1f, 0.1f, 0.1f };
			mat.roughness = 0.9f;
			mat.name = "Rough1";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.1f, 0.1f, 0.1f };
			mat.roughness = 0.7f;
			mat.name = "Rough2";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.1f, 0.1f, 0.1f };
			mat.roughness = 0.5f;
			mat.name = "Rough3";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.1f, 0.1f, 0.1f };
			mat.roughness = 0.3f;
			mat.name = "Rough4";

			m_scene.materials.push_back(mat);
		}
		{
			Material mat;
			mat.albedo = { 0.1f, 0.1f, 0.1f };
			mat.roughness = 0.0f;
			mat.name = "Rough5";

			m_scene.materials.push_back(mat);
		}
		{
			Mesh meshclass = Mesh();
			Mesh mesh = meshclass.LoadOBJFile("T:\\GIT\\GPURaytracer\\Raytracer\\cube.obj");
			mesh.materialIndex = 1;
			mesh.Transform = glm::vec3(0.0f, 2.0f, 0.0f);
			//m_scene.meshes.push_back(mesh);
		}
		{
			Mesh meshclass = Mesh();
			Mesh mesh = meshclass.LoadOBJFile("T:\\GIT\\GPURaytracer\\Raytracer\\plane.obj");
			mesh.materialIndex = 2;
			m_scene.meshes.push_back(mesh);
		}
		{
			Mesh meshclass = Mesh();
			Mesh mesh = meshclass.LoadOBJFile("T:\\GIT\\GPURaytracer\\Raytracer\\light.obj");
			mesh.materialIndex = 3;
			m_scene.meshes.push_back(mesh);
		}

		{
			Mesh meshclass = Mesh();
			Mesh mesh = meshclass.LoadOBJFile("T:\\GIT\\GPURaytracer\\Raytracer\\suzanne.obj");
			mesh.materialIndex = 1;
			m_scene.meshes.push_back(mesh);
		}
*/
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

		ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_PickerHueWheel);

		ImGui::Begin("Settings");
		ImGui::Text("Last render time: %.3f ms", m_renderTimeMs);
		ImGui::Text("Sample Index: %i", m_renderer.GetFrameIndex());
		/*
		//ImGui::Text("Camera DirX: %.2f, %.2f, %.2f, %.2f", m_camera.GetProjection()[0].x, m_camera.GetProjection()[0].y, m_camera.GetProjection()[0].z, m_camera.GetProjection()[0].w);
		//ImGui::Text("Camera DirY: %.2f, %.2f, %.2f, %.2f", m_camera.GetProjection()[1].x, m_camera.GetProjection()[1].y, m_camera.GetProjection()[1].z, m_camera.GetProjection()[1].w);
		//ImGui::Text("Camera DirZ: %.2f, %.2f, %.2f, %.2f", m_camera.GetProjection()[2].x, m_camera.GetProjection()[2].y, m_camera.GetProjection()[2].z, m_camera.GetProjection()[2].w);
		//ImGui::Text("Camera DirW: %.2f, %.2f, %.2f, %.2f", m_camera.GetProjection()[3].x, m_camera.GetProjection()[3].y, m_camera.GetProjection()[3].z, m_camera.GetProjection()[3].w);
		//ImGui::Text("Camera ViewX: %.2f, %.2f, %.2f, %.2f", m_camera.GetView()[0].x, m_camera.GetView()[0].y, m_camera.GetView()[0].z, m_camera.GetView()[0].w);
		//ImGui::Text("Camera ViewY: %.2f, %.2f, %.2f, %.2f", m_camera.GetView()[1].x, m_camera.GetView()[1].y, m_camera.GetView()[1].z, m_camera.GetView()[1].w);
		//ImGui::Text("Camera ViewZ: %.2f, %.2f, %.2f, %.2f", m_camera.GetView()[2].x, m_camera.GetView()[2].y, m_camera.GetView()[2].z, m_camera.GetView()[2].w);
		//ImGui::Text("Camera ViewW: %.2f, %.2f, %.2f, %.2f", m_camera.GetView()[3].x, m_camera.GetView()[3].y, m_camera.GetView()[3].z, m_camera.GetView()[3].w);
		*/
		ImGui::Text("Camera POS: %.2f, %.2f, %.2f", m_camera.GetPosition().x, m_camera.GetPosition().y, m_camera.GetPosition().z);
		ImGui::Checkbox("Accumulate", &m_renderer.GetSettings().accumulate);
		ImGui::DragFloat("Camera Speed", &m_camera.GetSpeed(), 0.1f);
		if (ImGui::DragFloat("Camera FOV", &m_camera.GetFOV(), 0.1f, 0.01f, 179.0f)) { m_sceneChanged = true; m_camera.RecalculateProjection(); }
		if (ImGui::DragFloat("Aperture", &m_camera.m_aperture, 0.1f, 1.0f, 22.0f)) { m_sceneChanged = true; }
		if (ImGui::DragFloat("Focus Distance", &m_camera.m_focusDistance, 0.1f, 0.01f, 1000.0f)) { m_sceneChanged = true; }
		if (ImGui::SliderInt("Max Bounces", &m_renderer.GetSettings().bounces, 0, 30)) { m_sceneChanged = true; }

		if (ImGui::Button("Reset") || m_sceneChanged)
		{
			m_renderer.ResetFrameIndex();
			m_sceneChanged = false;
		}

		ImGui::End();

		ImGui::Begin("Scene");
		ImGuiColorEditFlags misc_flags = ImGuiColorEditFlags_HDR;
		static ImGuiSliderFlags flags = ImGuiSliderFlags_None;
		static ImGuiSliderFlags flagLog = ImGuiSliderFlags_Logarithmic;

		if (ImGui::ColorEdit3("Sky Color", &(m_scene.skyColor.x))) { m_sceneChanged = true; }
		if (ImGui::SliderFloat("Sky Brightness", &m_scene.skyBrightness, 0.0f, 10.0f, "%.3f", flags))	{ m_sceneChanged = true; }
		if (ImGui::SliderFloat("Sun Focus", &m_scene.sunFocus, 1.0f, 100000.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
		if (ImGui::SliderFloat("Sun Intensity", &m_scene.sunIntensity, 0.0f, 100.0f, "%.3f", flags)) { m_sceneChanged = true; }
		if (ImGui::ColorEdit3("Sky Color Horizon", &(m_scene.skyColorHorizon.x))) { m_sceneChanged = true; }
		if (ImGui::ColorEdit3("Sky Color Zenith", &(m_scene.skyColorZenith.x))) { m_sceneChanged = true; }
		if (ImGui::ColorEdit3("Ground Color", &(m_scene.groundColor.x))) { m_sceneChanged = true; }

		ImGui::End();

		ImGui::Begin("Materials");
		for (size_t i = 0u; i < m_scene.materialCount; i++)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Material& mat = m_scene.materials[i];

			ImGui::Text("Material: %i", i);
			if (ImGui::ColorEdit3("Albedo", &(mat.albedo.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Metalness", &(mat.metalness), 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Roughness", &(mat.roughness), 0.0f, 1.0f, "%.3f")) { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Emission", &(mat.emission.x))) { m_sceneChanged = true; }
			if (ImGui::SliderFloat("Emission Intensity", &(mat.emissionIntensity), 0.0f, 1000.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
			ImGui::Separator();
			ImGui::PopID();
		}
		ImGui::End();
		/*
		for (size_t i = 0u; i < m_scene.meshes.size(); i++)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Mesh& mesh = m_scene.meshes[i];
			ImGui::Separator();
			ImGui::Text("Mesh: %s", mesh.name);
			if (ImGui::DragFloat3("Position", glm::value_ptr(mesh.Transform), 0.1f)) { m_sceneChanged = true; }
			ImGui::Separator();
			ImGui::PopID();
		}
		*/
		

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
	}

private:
	Renderer m_renderer;
	Camera m_camera;
	float m_renderTimeMs = 0.0f;
	std::deque<float> m_rendetTimeVec;
	uint32_t m_viewportWidth = 0, m_viewportHeight = 0;
	Scene m_scene;
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
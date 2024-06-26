#include <glm/gtc/type_ptr.hpp>

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
			//m_scene.meshes.push_back(mesh);
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
		//ImGui::ShowDemoWindow();

		ImGui::Begin("Settings");
		ImGui::Text("Last render time: %.3f ms", m_renderTimeMs);
		ImGui::Checkbox("Accumulate", &m_renderer.GetSettings().accumulate);
		ImGui::DragFloat("Camera Speed", &m_camera.GetSpeed(), 0.1f);
		if (ImGui::DragFloat("Camera FOV", &m_camera.GetFOV(), 0.1f)) { m_sceneChanged = true; }
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

		if (ImGui::ColorEdit3("Sky Color", glm::value_ptr(m_scene.m_skyColor)))				{ m_sceneChanged = true; }
		if (ImGui::SliderFloat("Sky Brightness", &m_scene.m_skyBrightness, 0.0f, 10.0f, "%.3f", flags))	{ m_sceneChanged = true; }
		if (ImGui::SliderFloat("Sun Focus", &m_scene.m_sunFocus, 50.0f, 100000.0f, "%.3f", flagLog)) { m_sceneChanged = true; }
		if (ImGui::SliderFloat("Sun Intensity", &m_scene.m_sunIntensity, 0.0f, 100.0f, "%.3f", flags)) { m_sceneChanged = true; }
		if (ImGui::ColorEdit3("Sky Color Horizon", glm::value_ptr(m_scene.m_skyColorHorizon))) { m_sceneChanged = true; }
		if (ImGui::ColorEdit3("Sky Color Zenith", glm::value_ptr(m_scene.m_skyColorZenith))) { m_sceneChanged = true; }
		if (ImGui::ColorEdit3("Ground Color", glm::value_ptr(m_scene.m_groundColor))) { m_sceneChanged = true; }

		for (size_t i = 0u; i < m_scene.meshes.size(); i++)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Mesh& mesh = m_scene.meshes[i];
			Material& mat = m_scene.materials[mesh.materialIndex];
			ImGui::Separator();
			ImGui::Text("Mesh: %s", mesh.name);
			ImGui::Text("Material: %s", mat.name);
			if (ImGui::DragInt("Material", &mesh.materialIndex, 0.1f, 0, (int)m_scene.materials.size() - 1)) { m_sceneChanged = true; }
			if (ImGui::DragFloat3("Position", glm::value_ptr(mesh.Transform), 0.1f)) { m_sceneChanged = true; }
			ImGui::Separator();
			ImGui::PopID();
		}

		for (size_t i = 0u; i < m_scene.spheres.size(); i++)
		{
			ImGui::PushID((int)i+m_scene.meshes.size());
			ImGui::AlignTextToFramePadding();

			Sphere& sphere = m_scene.spheres[i];
			Material& mat = m_scene.materials[sphere.materialIndex];
			ImGui::Separator();
			ImGui::Text("Sphere: %i", i);
			ImGui::Text("Material: %s", mat.name);
			if (ImGui::DragInt("Material", &sphere.materialIndex, 0.1f, 0, (int)m_scene.materials.size() - 1))   { m_sceneChanged = true; }
			if (ImGui::DragFloat3("Position", glm::value_ptr(sphere.position), 0.1f))						     { m_sceneChanged = true; }
			if (ImGui::DragFloat("Radius", &sphere.radius, 0.1f))											     { m_sceneChanged = true; }
			ImGui::Separator();
			ImGui::PopID();
		}
		ImGui::End();

		ImGui::Begin("Materials");
		for (size_t i = 0u; i < m_scene.materials.size(); i++)
		{
			ImGui::PushID((int)i);
			ImGui::AlignTextToFramePadding();

			Material& mat = m_scene.materials[i];
			ImGui::Text("%s", mat.name);
			if (ImGui::ColorEdit3("Albedo", glm::value_ptr(mat.albedo)))				      { m_sceneChanged = true; }
			if (ImGui::DragFloat("Roughness", &mat.roughness, 0.01f, 0.0f, 1.0f))			  { m_sceneChanged = true; }
			//if (ImGui::DragFloat("Specularity", &mat.specularProbability, 0.01f, 0.0f, 1.0f)) { m_sceneChanged = true; }
			if (ImGui::DragFloat("Metalness", &mat.metalness, 0.01f, 0.0f, 1.0f))			  { m_sceneChanged = true; }
			if (ImGui::ColorEdit3("Emission Color", glm::value_ptr(mat.emissionColor)))		  { m_sceneChanged = true; }
			if (ImGui::DragFloat("Emission Power", &mat.emissionPower, 0.1f, 0.0f, 100.0f))   { m_sceneChanged = true; }
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
			ImGui::Image(image->GetDescriptorSet(),
				{ (float)image->GetWidth(), (float)image->GetHeight() },
				ImVec2(0,1), ImVec2(1,0));
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

		m_renderTimeMs = timer.ElapsedMillis();
	}

private:
	Renderer m_renderer;
	Camera m_camera;
	float m_renderTimeMs = 0.0f;
	uint32_t m_viewportWidth = 0, m_viewportHeight = 0;
	Scene m_scene;
	bool m_sceneChanged = false;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "Walnut Example";

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
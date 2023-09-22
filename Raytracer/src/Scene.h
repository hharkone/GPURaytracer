#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Mesh.h"

struct Material
{
    glm::vec3 albedo{1.0f};
    float roughness = 1.0f;
    float metalness = 0.0f;
    glm::vec3 emissionColor { 0.0f };
    float emissionPower = 0.0f;
    std::string name = "Material";

    glm::vec3 GetEmission() const { return emissionColor * emissionPower; }
};

struct Sphere
{
    glm::vec3 position{0.0f};
    float radius = 0.5f;
    int materialIndex = 0;
};

struct Scene
{
    std::vector<Sphere> spheres;
    std::vector<Mesh::MeshData> meshes;
    std::vector<Material> materials;
    glm::vec3 m_skyColor = glm::vec4(0.6f, 0.7f, 0.9f, 1.0f);
    float m_skyBrightness = 1.0f;
};
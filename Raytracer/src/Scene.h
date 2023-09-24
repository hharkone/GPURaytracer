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
    float specularProbability = 0.0f;
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
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    glm::vec3 m_skyColor = glm::vec3(1.0f);
    float m_skyBrightness = 1.8f;
    glm::vec3 m_skyColorHorizon = glm::vec3(0.55f, 0.66f, 0.9f);
    glm::vec3 m_skyColorZenith  = glm::vec3(0.28f, 0.28f, 0.75f);
    glm::vec3 m_groundColor     = glm::vec3(0.08f, 0.08f, 0.08f);
    float m_sunFocus = 200.0f;
    float m_sunIntensity = 100.0f;
};
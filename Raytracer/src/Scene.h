#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Mesh.h"

struct Scene
{
    std::vector<Mesh> meshes;
    glm::vec3 m_skyColor = glm::vec3(1.0f);
    float m_skyBrightness = 1.8f;
    glm::vec3 m_skyColorHorizon = glm::vec3(0.55f, 0.66f, 0.9f);
    glm::vec3 m_skyColorZenith  = glm::vec3(0.28f, 0.28f, 0.75f);
    glm::vec3 m_groundColor     = glm::vec3(0.08f, 0.08f, 0.08f);
    float m_sunFocus = 200.0f;
    float m_sunIntensity = 100.0f;
};
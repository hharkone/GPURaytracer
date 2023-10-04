#pragma once

#include <memory>
#include <glm/glm.hpp>

#include "Walnut/Image.h"
#include "Camera.h"
#include "RayCPU.h"
#include "Scene.h"
#include "Mesh.h"
#include "CudaMain.cuh"

class CudaBuffer;

class Renderer
{

public:

    struct Settings
    {
        bool accumulate = true;
        int bounces = 5;
        int samples = 0;
    };

public:

    Renderer() = default;

    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }

    void ResetFrameIndex();
    Settings& GetSettings() { return m_settings; }
    uint32_t GetFrameIndex() { return m_frameIndex; }

private:

    struct Hit
    {
        bool didHit = false;
        int debugHit = 0;
        float hitDistance = std::numeric_limits<float>::max();
        glm::vec3 worldPosition = glm::vec3(0.0);
        glm::vec3 worldNormal = glm::vec3(0.0);
        int objectIndex = -1;
        int primIndex = -1;
        int materialIndex;
    };
    /*
    glm::vec4 PerPixel(uint32_t x, uint32_t y);
    glm::vec3 TraceRay(RayCPU& ray, uint32_t& seed);
    Hit raySphere(const RayCPU& ray, const Sphere& sphere);
    Hit rayTriangleIntersect(const RayCPU& ray, const Mesh::Triangle& tri, const glm::vec3& origin);
    Hit CalculateRayCollision(const RayCPU& ray);
    glm::vec3 GetEnvironmentLight(RayCPU& ray);
    */
private:

    const Scene* m_activeScene = nullptr;
    const Camera* m_activeCamera = nullptr;
    std::shared_ptr<Walnut::Image> m_finalImage;

    uint32_t* m_imageData = nullptr;
    Settings m_settings;
    uint32_t m_frameIndex = 1;
    std::shared_ptr<CudaRenderer> m_cudaRenderer = nullptr;
};
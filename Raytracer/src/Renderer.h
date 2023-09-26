#pragma once

#include <memory>
#include <glm/glm.hpp>

#include "Walnut/Image.h"
#include "Camera.h"
#include "Ray.h"
#include "Scene.h"
#include "Mesh.h"

class CudaBuffer;

class Renderer
{

public:

    struct Settings
    {
        bool accumulate = true;
        int bounces = 5;
    };

public:

    Renderer() = default;

    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }

    void ResetFrameIndex() { m_frameIndex = 1; }
    Settings& GetSettings() { return m_settings; }
    float* GetCudaData() { return m_cudaData; }

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

    glm::vec4 PerPixel(uint32_t x, uint32_t y);
    glm::vec3 TraceRay(Ray& ray, uint32_t& seed);
    Hit raySphere(const Ray& ray, const Sphere& sphere);
    Hit rayTriangleIntersect(const Ray& ray, const Mesh::Triangle& tri, const glm::vec3& origin);
    Hit CalculateRayCollision(const Ray& ray);
    glm::vec3 Renderer::GetEnvironmentLight(Ray& ray);

private:

    const Scene* m_activeScene = nullptr;
    const Camera* m_activeCamera = nullptr;
    std::shared_ptr<Walnut::Image> m_finalImage;
    std::vector<uint32_t> m_imgHorizontalIterator, m_imgVerticalIterator;

    uint32_t* m_imageData = nullptr;
    glm::vec4* m_accumulationData = nullptr;
    Settings m_settings;
    uint32_t m_frameIndex = 1;
    float* m_cudaData = nullptr;
    //CudaBuffer* m_cudaBuffer;
    std::shared_ptr<CudaBuffer> m_cudaBuffer;
};
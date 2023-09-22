#pragma once
#include <memory>
#include <glm/glm.hpp>

#include "Walnut/Image.h"
#include "Camera.h"
#include "Ray.h"
#include "Scene.h"
#include "Mesh.h"

class Renderer
{
public:

    struct Settings
    {
        bool accumulate = true;
    };

public:

    Renderer() = default;

    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }

    void ResetFrameIndex() { m_frameIndex = 1; }
    Settings& GetSettings() { return m_settings; }

private:

    struct Hit
    {
        float hitDistance = -1;;
        glm::vec3 worldPosition;
        glm::vec3 worldNormal;
        int objectIndex = -1;
        int primIndex = -1;
    };

    glm::vec4 PerPixel(uint32_t x, uint32_t y);
    Hit TraceRay(const Ray& ray);
    Hit rayTriangleIntersect(const Ray& ray, const Mesh::Triangle& tri, const glm::vec3& origin);
    Hit ClosestHit(const Ray& ray, float hitDistance, int objectIndex);
    Hit ClosestHit(const Ray& ray, float hitDistance, int objectIndex, int triIndex);
    Hit Miss(const Ray& ray, const bool missDistMax);

private:

    const Scene* m_activeScene = nullptr;
    const Camera* m_activeCamera = nullptr;
    std::shared_ptr<Walnut::Image> m_finalImage;
    std::vector<uint32_t> m_imgHorizontalIterator, m_imgVerticalIterator;

    uint32_t* m_imageData = nullptr;
    glm::vec4* m_accumulationData = nullptr;
    Settings m_settings;
    uint32_t m_frameIndex = 1;
};
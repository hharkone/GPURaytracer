#pragma once
#include <memory>
#include <glm/glm.hpp>

#include "Walnut/Image.h"
#include "Camera.h"
#include "Ray.h"


class Renderer
{
public:
    Renderer() = default;

    void OnResize(uint32_t width, uint32_t height);
    void Render(const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }

private:
    glm::vec4 TraceRay(const Ray& ray);
private:
    std::shared_ptr<Walnut::Image> m_finalImage;
    uint32_t* m_imageData = nullptr;
};
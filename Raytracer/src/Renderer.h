#pragma once

#include <memory>

#include <glm/glm.hpp>

#include "Walnut/Image.h"
#include "Camera.h"
#include "RayCPU.h"
#include "Scene.h"
#include "Mesh.h"
#include "CudaMain.cuh"
#include "Denoiser.cuh"

class CudaBuffer;

class Renderer
{
public:

    struct Settings
    {
        bool accumulate = true;
        int bounces = 15;
        int samples = 0;
        bool denoise = true;
    };

    Renderer() = default;
    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }

    void ResetFrameIndex();
    Settings& GetSettings() { return m_settings; }
    uint32_t GetFrameIndex() { return m_frameIndex; }

private:

    const Scene* m_activeScene = nullptr;
    const Camera* m_activeCamera = nullptr;
    std::shared_ptr<Walnut::Image> m_finalImage;
    Settings m_settings;
    uint32_t m_frameIndex = 1;
    std::shared_ptr<CudaRenderer> m_cudaRenderer = nullptr;
    Denoiser m_denoiser;
};
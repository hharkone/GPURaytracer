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
#include "RendererSettings.h"

class CudaBuffer;

class Renderer
{
public:

    Renderer() = default;
    void SaveRenderToDisk(const std::string path);
    void LoadHDRI(const Scene& scene);
    void OnResize(const Scene& scene, uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }

    void ResetFrameIndex();
    RenderSettings& GetSettings() { return m_settings; }
    uint32_t GetFrameIndex() { return m_frameIndex; }

private:

    const Scene* m_activeScene = nullptr;
    const Camera* m_activeCamera = nullptr;
    std::shared_ptr<Walnut::Image> m_finalImage;
    RenderSettings m_settings;
    uint32_t m_frameIndex = 1;
    std::shared_ptr<CudaRenderer> m_cudaRenderer = nullptr;
    Denoiser m_denoiser;
};
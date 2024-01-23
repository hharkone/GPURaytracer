#pragma once

#include <memory>
#include <optix.h>

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
        int bounces = 15;
        int samples = 0;
        bool denoise = true;
    };

    Renderer() = default;
    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_finalImage; }
    //uint32_t* GetRawImagePrt() const { return m_imageData; }

    void ResetFrameIndex();
    Settings& GetSettings() { return m_settings; }
    uint32_t GetFrameIndex() { return m_frameIndex; }

private:

    bool Renderer::InitOptix(uint32_t width, uint32_t height);
    bool Denoise();
    const Scene* m_activeScene = nullptr;
    const Camera* m_activeCamera = nullptr;
    std::shared_ptr<Walnut::Image> m_finalImage;

    uint32_t* m_imageData = nullptr;
    float* m_imageDataFloatDenoised = nullptr;
    float* m_imageDataFloatDenoisedDevice = nullptr;
    Settings m_settings;
    uint32_t m_frameIndex = 1;
    std::shared_ptr<CudaRenderer> m_cudaRenderer = nullptr;
    OptixDenoiser m_optixDenoiser = nullptr;
    OptixDeviceContext m_optix_context = nullptr;
    cudaStream_t m_cudaStream = nullptr;
    OptixDenoiserLayer m_optixLayer = {};
    OptixDenoiserParams m_optixParams = {};
    OptixDenoiserOptions m_optixOptions = {};
    void* m_denoiser_state_buffer = nullptr;
    void* m_denoiser_scratch_buffer = nullptr;
    OptixDenoiserSizes m_denoiser_sizes = {};
};
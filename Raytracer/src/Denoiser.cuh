#pragma once
#include <memory>
#include <optix.h>

#include "CudaBuffer.h"
#include "Scene.h"

class Denoiser
{
public:
	Denoiser() {}
    ~Denoiser();

    void Denoise(const TonemapSettings* tonemapper, bool enabled);
    void InitOptix(void* inputBeautyBuffer, void* inputAlbedoBuffer, void* inputNormalBuffer, uint32_t width, uint32_t height);
    float* GetDenoisedBuffer() { return m_finalOutputBuffer; }
    float* GetLinearDenoisedBuffer() { return m_finalLinearOutputBuffer; }
    cudaStream_t* GetCudaStream() { return &m_cudaStream; }

private:
    OptixDenoiser m_optixDenoiser = nullptr;
    OptixDeviceContext m_optix_context = nullptr;
    cudaStream_t m_cudaStream = nullptr;
    OptixDenoiserLayer m_optixLayer = {};
    OptixDenoiserParams m_optixParams = {};
    OptixDenoiserOptions m_optixOptions = {};
    void* m_denoiser_state_buffer = nullptr;
    void* m_denoiser_scratch_buffer = nullptr;
    OptixDenoiserSizes m_denoiser_sizes = {};
    OptixDenoiserGuideLayer m_guide_layer = {};
    uint32_t m_width = 0u;
    uint32_t m_height = 0u;

    CUDABuffer m_floatDenoisedBuffer_GPU;     //Denoised linear float output on the device
    CUDABuffer m_floatTonemappedBuffer_GPU;   //Final tonemapped float output on the device

    float* m_finalOutputBuffer = nullptr;	//Final tonemapped output
    float* m_finalLinearOutputBuffer = nullptr;	//Final float output
};
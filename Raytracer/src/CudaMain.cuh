#pragma once
#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

class CudaRenderer
{
public:
	CudaRenderer(uint32_t width, uint32_t height, uint32_t* sampleIndex, size_t samples, int* bounces) : m_bufferSize(width * height * sizeof(float3)),
		m_sampleIndex(sampleIndex), m_samples(samples), m_bounces(bounces), m_width(width), m_height(height)
	{
		m_outputBuffer = new float[width * height * 3];
		m_imageData = new uint32_t[width * height];
		memset(m_imageData, 0, (size_t)width * (size_t)height * sizeof(uint32_t));

		cudaMalloc(&m_outputBuffer_GPU, m_bufferSize);
		cudaMalloc(&m_accumulationBuffer_GPU, m_bufferSize);
		cudaMalloc(&m_imageData_GPU, (size_t)width * (size_t)height * sizeof(uint32_t));
		cudaMemset(m_accumulationBuffer_GPU, 0, m_bufferSize);
		cudaMemset(m_imageData_GPU, 0, (size_t)width * (size_t)height * sizeof(uint32_t));

		m_cameraPos = { 0.0f, 0.0f, 0.0f };
		m_invViewMat = new float[16];
		m_invProjMat = new float[16];
		m_viewMat    = new float[16];
	}

	~CudaRenderer()
	{
		cudaDeviceSynchronize();

		cudaFree(m_accumulationBuffer_GPU);
		cudaFree(m_imageData_GPU);
		cudaFree(m_outputBuffer_GPU);
	}

	void SetCamera(float3 pos, float3 dir, float fov);
	void SetInvViewMat(float4 x, float4 y, float4 z, float4 w);
	void SetInvProjMat(float4 x, float4 y, float4 z, float4 w);
	void SetViewMat(float4 x, float4 y, float4 z, float4 w);
	void Compute(void);
	void Clear(void);
	void SetBounces(int bounces) { m_bounces = &bounces; }
	float* getFloatOutputData(void) { return m_outputBuffer; }
	uint32_t* getImageData(void) { return m_imageData; }

	uint32_t m_width;
	uint32_t m_height;

private:
	const size_t m_bufferSize;
	uint32_t* m_sampleIndex;
	size_t m_samples;
	int* m_bounces;
	float3 m_cameraPos = { 0.0f, 0.0f, 0.0f };
	float3 m_cameraDir = {0.0f, 0.0f, -1.0f};
	float* m_invViewMat = nullptr;
	float* m_invProjMat = nullptr;
	float* m_viewMat = nullptr;
	float m_fov = 50.0f;
	uint32_t* m_imageData = nullptr;
	float* m_outputBuffer = nullptr;
	float3* m_outputBuffer_GPU = nullptr;
	float3* m_accumulationBuffer_GPU = nullptr;
	uint32_t* m_imageData_GPU = nullptr;
};
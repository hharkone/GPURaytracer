#pragma once
#include <memory>
//#include <vector>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

class CudaRenderer
{
public:
	CudaRenderer(size_t width, size_t height, float fov, size_t sampleIndex, size_t samples, size_t bounces) : m_bufferSize(width * height * sizeof(float3)),
		m_fov(fov), m_sampleIndex(sampleIndex), m_samples(samples), m_bounces(bounces), m_width(width), m_height(height)
	{
		m_accumulationBuffer = new float[width * height * 3];
	}

	void Compute(void);
	float* getAccumulationData(void) { return m_accumulationBuffer; }

private:
	const size_t m_bufferSize;
	size_t m_sampleIndex;
	size_t m_samples;
	size_t m_bounces;
	size_t m_width;
	size_t m_height;
	float m_fov;
	float* m_accumulationBuffer = nullptr;
};
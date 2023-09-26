#pragma once
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class CudaBuffer
{
public:
	CudaBuffer(size_t bufferSize) : m_bufferSize(bufferSize)
	{
		vec = std::vector<float>(bufferSize);

		m_cudaBufferA = new float[bufferSize];
		m_cudaBufferB = new float[bufferSize];
		m_cudaBufferC = new float[bufferSize];

		for (int i = 0; i < bufferSize; i++)
		{
			vec.at(i) = 10.0f * i;
		}

		for (int i = 0; i < bufferSize; i++)
		{
			m_cudaBufferA[i] = float(i+10);
			m_cudaBufferB[i] = float(i);
			m_cudaBufferC[i] = float(i*2);
		}
	}

	void Compute(void);
	float* getData(void) { return m_cudaBufferC; }
	std::vector<float> vec;

private:
	const size_t m_bufferSize;
	float* m_cudaBufferA;
	float* m_cudaBufferB;
	float* m_cudaBufferC;
};
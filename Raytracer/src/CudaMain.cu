#include "CudaMain.cuh"

int checkCudaError(cudaError_t& error)
{
	if (error == cudaSuccess)
	{
		return 0;
	}

	return 1;
}

__global__ void addKernel(float* a, float* b, float* c)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float* a, float* b, float* c, unsigned int size)
{
	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (checkCudaError(cudaStatus)) { goto Error; }

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
	if (checkCudaError(cudaStatus)) { goto Error; }

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
	if (checkCudaError(cudaStatus)) { goto Error; }

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
	if (checkCudaError(cudaStatus)) { goto Error; }

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	if (checkCudaError(cudaStatus)) { goto Error; }

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
	if (checkCudaError(cudaStatus)) { goto Error; }

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size>>> (dev_a, dev_b, dev_c);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (checkCudaError(cudaStatus)) { goto Error; }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (checkCudaError(cudaStatus)) { goto Error; }

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (checkCudaError(cudaStatus)) { goto Error; }

Error:
	//fprintf(stderr, "Cuda kernel failed!");
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

void CudaBuffer::Compute(void)
{
	cudaError_t cudaStatus = addWithCuda(m_cudaBufferA, m_cudaBufferB, m_cudaBufferC, (unsigned int)m_bufferSize);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda compute failed!");
	}
}
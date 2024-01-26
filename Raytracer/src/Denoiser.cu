#include "Denoiser.cuh"
#include "cutil_math.cuh"
#include "device_launch_parameters.h"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

inline void cudaCheckReportError(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		fprintf(stderr, "\n");
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define CU_CHECK(val) cudaCheckReportError((val), #val, __FILE__, __LINE__)

inline void optixCheckReportError(OptixResult result, char const* const func, const char* const file, int const line)
{
	if (result != OPTIX_SUCCESS)
	{
		fprintf(stderr, "OptiX error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), optixGetErrorName(result), func);
		fprintf(stderr, "\n");
	}
}

// This will output the proper OptiX error strings in the event
// that a OptiX host call returns an error
#define OPTIX_CHECK(val) optixCheckReportError((val), #val, __FILE__, __LINE__)


__global__ void tonemapper_kernel2(float4* inputBuffer, float4* outputBuffer, uint32_t width, uint32_t height, const Scene* scene)
{
	uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;


	if ((x >= width) || (y >= height))
		return;

	// Index of current pixel (calculated using thread index)
	uint32_t i = (height - y - 1) * width + x;

	float A = scene->tonemap.A;
	float B = scene->tonemap.B;
	float C = scene->tonemap.C;
	float D = scene->tonemap.D;
	float E = scene->tonemap.E;
	float F = scene->tonemap.F;
	float W = scene->tonemap.W;
	float Exp = scene->tonemap.Exposure;

	float3 c = (make_float3(inputBuffer[i].x, inputBuffer[i].y, inputBuffer[i].z)) * Exp;

	float wScale = (((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F);

	float3 outColor = (((c * (A * c + C * B) + D * E) / (c * (A * c + B) + D * F)) - E / F) * (1.0f / wScale);

	outColor.x = clamp(outColor.x, 0.0f, 1.0f);
	outColor.y = clamp(outColor.y, 0.0f, 1.0f);
	outColor.z = clamp(outColor.z, 0.0f, 1.0f);
	float alpha = clamp(inputBuffer[i].w, 0.0f, 1.0f);

	outputBuffer[i] = make_float4(powf(outColor, 0.4646464), alpha);
}

void Denoiser::InitOptix(void* inputBeautyBuffer, void* inputAlbedoBuffer, void* inputNormalBuffer, uint32_t width, uint32_t height)
{
	m_width = width;
	m_height = height;

	CU_CHECK(cudaSetDevice(0u));
	CU_CHECK(cudaStreamCreate(&m_cudaStream));

	if (m_optixDenoiser != nullptr)
	{
		optixDenoiserDestroy(m_optixDenoiser);
	}

	if (m_optix_context != nullptr)
	{
		optixDeviceContextDestroy(m_optix_context);
	}

	OptixResult result = optixInit();
	if (result != OPTIX_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize OptiX library (%d)\n", result);
		return;
	}

	m_deviceScene.resize(sizeof(Scene));

	m_floatDenoisedBuffer_GPU.resize(width * height * sizeof(float4));
	m_floatTonemappedBuffer_GPU.resize(width * height * sizeof(float4));

	// Initialize our optix context
	CUcontext cuCtx = 0; // Zero means take the current context
	m_optix_context = nullptr;
	result = optixDeviceContextCreate(cuCtx, nullptr, &m_optix_context);
	if (result != OPTIX_SUCCESS)
	{
		fprintf(stderr, "Could not create OptiX context: (%d) %s\n", result, optixGetErrorName(result));
		return;
	}

	// Set the denoiser options
	m_optixOptions.guideAlbedo = true;
	m_optixOptions.guideNormal = true;
	m_optixOptions.denoiseAlpha = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY;

	// Iniitalize the OptiX denoiser
	OPTIX_CHECK(optixDenoiserCreate(m_optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &m_optixOptions, &m_optixDenoiser));


	memset(&m_denoiser_sizes, 0, sizeof(OptixDenoiserSizes));
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_optixDenoiser, width, height, &m_denoiser_sizes));

	// Allocate this space on the GPU
	CU_CHECK(cudaFree(m_denoiser_state_buffer));
	CU_CHECK(cudaFree(m_denoiser_scratch_buffer));
	CU_CHECK(cudaMalloc(&m_denoiser_state_buffer, m_denoiser_sizes.stateSizeInBytes));
	CU_CHECK(cudaMalloc(&m_denoiser_scratch_buffer, m_denoiser_sizes.withoutOverlapScratchSizeInBytes));

	// Setup the denoiser
	OPTIX_CHECK(optixDenoiserSetup(m_optixDenoiser, 0, width, height,
		(CUdeviceptr)m_denoiser_state_buffer, m_denoiser_sizes.stateSizeInBytes,
		(CUdeviceptr)m_denoiser_scratch_buffer, m_denoiser_sizes.withoutOverlapScratchSizeInBytes));

	// Set the denoiser parameters
	memset(&m_optixParams, 0, sizeof(OptixDenoiserParams));
	m_optixParams.blendFactor = 0.0f;

	//if (CU_CHECK(cudaMalloc((void**)&m_optixParams.hdrIntensity, sizeof(float)))) return false; //WTF!!?!?!

	// Create and set our OptiX layers
	memset(&m_optixLayer, 0, sizeof(OptixDenoiserLayer));

	//Input-Output data
	//m_optixLayer.input.data = (CUdeviceptr)(m_cudaRenderer->getFloatOutputDataDevice()->d_pointer());

	m_optixLayer.input.data = (CUdeviceptr)inputBeautyBuffer;
	m_optixLayer.input.width = width;
	m_optixLayer.input.height = height;
	m_optixLayer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	m_optixLayer.input.pixelStrideInBytes = sizeof(float4);
	m_optixLayer.input.rowStrideInBytes = width * sizeof(float4);
	m_optixLayer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

	m_optixLayer.output.data = m_floatDenoisedBuffer_GPU.d_pointer();
	m_optixLayer.output.width = width;
	m_optixLayer.output.height = height;
	m_optixLayer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	m_optixLayer.output.pixelStrideInBytes = sizeof(float4);
	m_optixLayer.output.rowStrideInBytes = width * sizeof(float4);

	//Guide layers
	m_guide_layer.albedo.data = (CUdeviceptr)inputAlbedoBuffer;
	m_guide_layer.albedo.width = width;
	m_guide_layer.albedo.height = height;
	m_guide_layer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	m_guide_layer.albedo.pixelStrideInBytes = sizeof(float3);
	m_guide_layer.albedo.rowStrideInBytes = width * sizeof(float3);

	m_guide_layer.normal.data = (CUdeviceptr)inputNormalBuffer;
	m_guide_layer.normal.width = width;
	m_guide_layer.normal.height = height;
	m_guide_layer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	m_guide_layer.normal.pixelStrideInBytes = sizeof(float3);
	m_guide_layer.normal.rowStrideInBytes = width * sizeof(float3);

	delete[] m_finalOutputBuffer;
	m_finalOutputBuffer = new float[width * height * 4];
	memset(m_finalOutputBuffer, 0, width * height * sizeof(float4));

	return;
}

void Denoiser::Denoise(const Scene* scene, bool enabled)
{
	int tx = 8;
	int ty = 8;

	m_deviceScene.upload(scene, 1u);

	// dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 blocks(m_width / tx + 1, m_height / ty + 1, 1);
	dim3 threads(tx, ty);

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	}

	if (enabled)
	{
		// Execute the denoiser
		OPTIX_CHECK(optixDenoiserInvoke(m_optixDenoiser, m_cudaStream, &m_optixParams,
			(CUdeviceptr)m_denoiser_state_buffer, m_denoiser_sizes.stateSizeInBytes,
			&m_guide_layer, &m_optixLayer, 1u, 0u, 0u,
			(CUdeviceptr)m_denoiser_scratch_buffer, m_denoiser_sizes.withoutOverlapScratchSizeInBytes));

		CU_CHECK(cudaStreamSynchronize(m_cudaStream));
		CU_CHECK(cudaDeviceSynchronize());

		tonemapper_kernel2 <<<blocks, threads >>> ((float4*)m_floatDenoisedBuffer_GPU.d_pointer(), (float4*)m_floatTonemappedBuffer_GPU.d_pointer(), m_width, m_height, (Scene*)m_deviceScene.d_pointer());
	}
	else
	{
		tonemapper_kernel2 <<<blocks, threads >>> ((float4*)m_optixLayer.input.data, (float4*)m_floatTonemappedBuffer_GPU.d_pointer(), m_width, m_height, (Scene*)m_deviceScene.d_pointer());
	}

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "tonemapper_kernel2 failed!\n");
	}

	cudaDeviceSynchronize();

	CU_CHECK(cudaMemcpy(m_finalOutputBuffer, (void*)m_floatTonemappedBuffer_GPU.d_pointer(), m_width * m_height * sizeof(float4), cudaMemcpyDeviceToHost));

	CU_CHECK(cudaStreamSynchronize(m_cudaStream));
}

Denoiser::~Denoiser()
{
	cudaDeviceSynchronize();

	optixDenoiserDestroy(m_optixDenoiser);
	optixDeviceContextDestroy(m_optix_context);
	cudaFree(m_denoiser_state_buffer);
	cudaFree(m_denoiser_scratch_buffer);
	m_floatDenoisedBuffer_GPU.free();
	m_floatTonemappedBuffer_GPU.free();
	m_deviceScene.free();
}
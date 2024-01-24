#include <execution>
#include <glm/gtc/type_ptr.hpp>

#include "Renderer.h"
#include "Walnut/Random.h"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

inline bool cudaCheckReportError(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		fprintf(stderr, "\n");
		return true;
	}
	return false;
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define CU_CHECK(val) cudaCheckReportError((val), #val, __FILE__, __LINE__)

inline bool optixCheckReportError(OptixResult result, char const* const func, const char* const file, int const line)
{
	if (result != OPTIX_SUCCESS)
	{
		fprintf(stderr, "OptiX error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), optixGetErrorName(result), func);
		fprintf(stderr, "\n");
		return true;
	}
	return false;
}

// This will output the proper OptiX error strings in the event
// that a OptiX host call returns an error
#define OPTIX_CHECK(val) optixCheckReportError((val), #val, __FILE__, __LINE__)

namespace Utils
{
	float4 vec4Tofloat4(glm::vec4& v)
	{
		return make_float4(v.x, v.y, v.z, v.w);
	}
}

bool Renderer::InitOptix(uint32_t width, uint32_t height)
{
	cudaSetDevice(0u);
	//cudaFree(0);
	cudaStreamCreate(&m_cudaStream);

	//if (m_cudaStream != nullptr)
	//{
		//cudaStreamDestroy(m_cudaStream);
		//cudaFree(&m_optixLayer.output.data);
	//}

	cudaError_t cudaStatus = cudaErrorStartupFailure;
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Renderer Optix stream destroy failed: %s\n", cudaGetErrorString(cudaStatus));
	}

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
		return false;
	}

	// Initialize our optix context
	CUcontext cuCtx = 0; // Zero means take the current context
	m_optix_context = nullptr;
	result = optixDeviceContextCreate(cuCtx, nullptr, &m_optix_context);
	if (result != OPTIX_SUCCESS)
	{
		fprintf(stderr, "Could not create OptiX context: (%d) %s\n", result, optixGetErrorName(result));
		return false;
	}

	// Set the denoiser options
	m_optixOptions.guideAlbedo = false;
	m_optixOptions.guideNormal = false;
	m_optixOptions.denoiseAlpha = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY;

	// Iniitalize the OptiX denoiser
	optixDenoiserCreate(m_optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &m_optixOptions, &m_optixDenoiser);

	
	memset(&m_denoiser_sizes, 0, sizeof(OptixDenoiserSizes));
	if(OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_optixDenoiser, width, height, &m_denoiser_sizes))) return false;

	// Allocate this space on the GPU
	m_denoiser_state_buffer = nullptr;
	m_denoiser_scratch_buffer = nullptr;
	if (CU_CHECK(cudaMalloc(&m_denoiser_state_buffer, m_denoiser_sizes.stateSizeInBytes))) return false;
	if (CU_CHECK(cudaMalloc(&m_denoiser_scratch_buffer, m_denoiser_sizes.withoutOverlapScratchSizeInBytes))) return false;

	// Setup the denoiser
	if(OPTIX_CHECK(optixDenoiserSetup(m_optixDenoiser, 0, width, height,
		(CUdeviceptr)m_denoiser_state_buffer, m_denoiser_sizes.stateSizeInBytes,
		(CUdeviceptr)m_denoiser_scratch_buffer, m_denoiser_sizes.withoutOverlapScratchSizeInBytes))) return false;

	// Set the denoiser parameters
	memset(&m_optixParams, 0, sizeof(OptixDenoiserParams));
	m_optixParams.blendFactor = 0.0f;

	//if (CU_CHECK(cudaMalloc((void**)&m_optixParams.hdrIntensity, sizeof(float)))) return false; //WTF!!?!?!

	// Create and set our OptiX layers
	memset(&m_optixLayer, 0, sizeof(OptixDenoiserLayer));

	//Input-Output data
	m_optixLayer.input.data = (CUdeviceptr)(m_cudaRenderer->getFloatOutputDataDevice()->d_pointer());
	m_optixLayer.input.width = width;
	m_optixLayer.input.height = height;
	m_optixLayer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	m_optixLayer.input.pixelStrideInBytes = sizeof(float4);
	m_optixLayer.input.rowStrideInBytes = width * sizeof(float4);
	m_optixLayer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

	m_optixLayer.output.data = (CUdeviceptr)m_imageDataFloatDenoisedDevice;
	m_optixLayer.output.width = width;
	m_optixLayer.output.height = height;
	m_optixLayer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	m_optixLayer.output.pixelStrideInBytes = sizeof(float4);
	m_optixLayer.output.rowStrideInBytes = width * sizeof(float4);

	//Guide layers
	m_guide_layer.albedo.data = (CUdeviceptr)(m_cudaRenderer->getFloatAlbedoOutputDataDevice()->d_pointer());
	m_guide_layer.albedo.width = width;
	m_guide_layer.albedo.height = height;
	m_guide_layer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	m_guide_layer.albedo.pixelStrideInBytes = sizeof(float3);
	m_guide_layer.albedo.rowStrideInBytes = width * sizeof(float3);

	m_guide_layer.normal.data = (CUdeviceptr)(m_cudaRenderer->getFloatNormalOutputDataDevice()->d_pointer());
	m_guide_layer.normal.width = width;
	m_guide_layer.normal.height = height;
	m_guide_layer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	m_guide_layer.normal.pixelStrideInBytes = sizeof(float3);
	m_guide_layer.normal.rowStrideInBytes = width * sizeof(float3);

	return true;
}

bool Renderer::Denoise()
{
	m_optixLayer.input.data = (CUdeviceptr)(m_cudaRenderer->getFloatOutputDataDevice()->d_pointer());
	//cudaMemcpy(&m_optixLayer.input.data, m_cudaRenderer->getFloatOutputDataDevice(), sizeof(CUdeviceptr), cudaMemcpyDeviceToDevice);

	// Execute the denoiser
	if (OPTIX_CHECK(optixDenoiserInvoke(m_optixDenoiser, m_cudaStream, &m_optixParams,
		(CUdeviceptr)m_denoiser_state_buffer, m_denoiser_sizes.stateSizeInBytes,
		&m_guide_layer, &m_optixLayer, 1u, 0u, 0u,
		(CUdeviceptr)m_denoiser_scratch_buffer, m_denoiser_sizes.withoutOverlapScratchSizeInBytes))) return false;

	cudaStreamSynchronize(m_cudaStream);

	cudaMemcpy(m_imageDataFloatDenoised, (void*)m_optixLayer.output.data, m_cudaRenderer->m_width * m_cudaRenderer->m_height * sizeof(float4), cudaMemcpyDeviceToHost);

	cudaStreamSynchronize(m_cudaStream);

	return true;
}

void Renderer::ResetFrameIndex()
{
	m_frameIndex = 1;

	if(m_cudaRenderer)
		m_cudaRenderer->Clear();
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_finalImage)
	{
		if (m_finalImage->GetWidth() == width && m_finalImage->GetHeight() == height)
			return;

		m_finalImage->Resize(width, height);
		ResetFrameIndex();
	}
	else
	{
		m_finalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA32F);
	}

	delete[] m_imageData;
	m_imageData = nullptr;

	//m_imageData = new uint32_t[width * height];
	//memset(m_imageData, 0, (size_t)width * (size_t)height * sizeof(uint32_t));

	delete[] m_imageDataFloatDenoised;
	m_imageDataFloatDenoised = nullptr;

	m_imageDataFloatDenoised = new float[(size_t)width * (size_t)height * sizeof(float4)];
	memset(m_imageDataFloatDenoised, 0, (size_t)width * (size_t)height * sizeof(float4));

	cudaMalloc(&m_imageDataFloatDenoisedDevice, (size_t)width * (size_t)height * sizeof(float4));
	cudaMemset(m_imageDataFloatDenoisedDevice, 0, (size_t)width * (size_t)height * sizeof(float4));

	if (!m_cudaRenderer)
	{
		m_cudaRenderer = std::shared_ptr<CudaRenderer>(new CudaRenderer(width, height, &m_activeScene, &m_frameIndex, &GetSettings().samples, &m_settings.bounces));
	}
	else
	{
		m_cudaRenderer->OnResize(width, height);
	}

	Renderer::InitOptix(width, height);
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_activeScene = &scene;
	m_activeCamera = &camera;

	if (m_cudaRenderer)
	{
		if (m_settings.accumulate)
		{
			glm::vec3 glmPos = m_activeCamera->GetPosition();
			float3 pos = make_float3(glmPos.x, glmPos.y, glmPos.z);
			glm::vec3 glmDir = m_activeCamera->GetDirection();
			float3 dir = make_float3(glmDir.x, glmDir.y, glmDir.z);

			glm::mat4x4 invView = m_activeCamera->GetInverseView();
			float4 x1 = Utils::vec4Tofloat4(invView[0]);
			float4 y1 = Utils::vec4Tofloat4(invView[1]);
			float4 z1 = Utils::vec4Tofloat4(invView[2]);
			float4 w1 = Utils::vec4Tofloat4(invView[3]);

			glm::mat4x4 invProj = m_activeCamera->GetInverseProjection();
			float4 x2 = Utils::vec4Tofloat4(invProj[0]);
			float4 y2 = Utils::vec4Tofloat4(invProj[1]);
			float4 z2 = Utils::vec4Tofloat4(invProj[2]);
			float4 w2 = Utils::vec4Tofloat4(invProj[3]);

			glm::mat4x4 viewMat = m_activeCamera->GetView();
			float4 x3 = Utils::vec4Tofloat4(viewMat[0]);
			float4 y3 = Utils::vec4Tofloat4(viewMat[1]);
			float4 z3 = Utils::vec4Tofloat4(viewMat[2]);
			float4 w3 = Utils::vec4Tofloat4(viewMat[3]);

			glm::mat4x4 localToWorldMat = m_activeCamera->GetLocalToWorld();
			float4 x4 = Utils::vec4Tofloat4(localToWorldMat[0]);
			float4 y4 = Utils::vec4Tofloat4(localToWorldMat[1]);
			float4 z4 = Utils::vec4Tofloat4(localToWorldMat[2]);
			float4 w4 = Utils::vec4Tofloat4(localToWorldMat[3]);

			m_cudaRenderer->SetCamera(pos, dir, m_activeCamera->m_aperture, m_activeCamera->m_focusDistance);
			m_cudaRenderer->SetInvViewMat(x1, y1, z1, w1);
			m_cudaRenderer->SetInvProjMat(x2, y2, z2, w2);
			m_cudaRenderer->SetViewMat(x3, y3, z3, w3);
			m_cudaRenderer->SetLocalToWorldMat(x4, y4, z4, w4);

			m_cudaRenderer->Compute();

			m_frameIndex++;
		}

		if (m_settings.denoise)
		{
			if (m_settings.accumulate)
			{
				Denoise();
			}

			if (m_imageDataFloatDenoised)
			{
				m_finalImage->SetData(m_imageDataFloatDenoised);
			}
		}
		else
		{
			if (m_cudaRenderer->getFloatOutputData())
			{
				m_finalImage->SetData(m_cudaRenderer->getFloatOutputData());
			}
		}
	}
}

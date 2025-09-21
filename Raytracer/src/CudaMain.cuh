#pragma once
#include <memory>

#include "scene.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "GPU_Mesh.h"
#include "ImageLoader.h"
#include "CudaBuffer.h"
#include "RendererSettings.h"


class CudaRenderer
{
public:
	CudaRenderer(uint32_t width, uint32_t height, const Scene* scene, uint32_t* sampleIndex, RenderSettings* rendererSettings)
		: m_scene(scene), m_sampleIndex(sampleIndex), m_rendererSettings(rendererSettings)
	{
		cudaError_t cudaStatus = cudaErrorStartupFailure;

		OnResize(width, height);

		m_cameraPos = { 0.0f, 0.0f, 0.0f };
		m_invViewMat	  = new float[16];
		m_invProjMat	  = new float[16];
		m_viewMat		  = new float[16];
		m_localToWorldMat = new float[16];

		m_deviceScene.alloc(sizeof(Scene));
		m_deviceSettings.alloc(sizeof(RenderSettings));

		//m_imgLoaderTestTexture0.LoadImage_PNG("Images/Lion/LionAlbedo.png");
		//m_imgLoaderTestTexture1.LoadImage_PNG("Images/Lion/LionMetalRough.png");
		//m_imgLoaderTestTexture2.LoadImage_PNG("Images/Lion/LionNormal.png");

		//m_imgLoaderTestTexture0.LoadImage_PNG("Images/ConcreteBrick/T_vk2vcdl_4K_B.png");
		//m_imgLoaderTestTexture1.LoadImage_PNG("Images/ConcreteBrick/T_vk2vcdl_4K_ORM.png");
		//m_imgLoaderTestTexture2.LoadImage_PNG("Images/ConcreteBrick/T_vk2vcdl_4K_N.png");

		//m_imgLoaderTestTexture0.LoadImage_PNG("Images/OffroadTire/Offroad_Tire_Clean_BaseColor.png");
		//m_imgLoaderTestTexture1.LoadImage_PNG("Images/OffroadTire/Offroad_Tire_Clean_ORM.png");
		//m_imgLoaderTestTexture2.LoadImage_PNG("Images/OffroadTire/Offroad_Tire_Clean_Normal.png");

		if (scene->envImgPathChanged)
		{
			m_imgLoaderEnv.LoadImage_EXR(scene->envImgPath);
		}

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "ImageLoader: Data buffer copy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		}
	}

	~CudaRenderer()
	{
		cudaDeviceSynchronize();

		//m_accumulationBuffer_GPU.free();
		m_floatOutputBuffer_GPU.free();
		m_floatAlbedoBuffer_GPU.free();
		m_floatNormalBuffer_GPU.free();

		m_deviceScene.free();
		m_deviceSettings.free();
	}

	void SetHDRI(std::string path);
	void SetScene(const Scene* scene);
	void SetCamera(float3 pos, float3 dir, float aperture, float focusDist);
	void SetInvViewMat(float4 x, float4 y, float4 z, float4 w);
	void SetInvProjMat(float4 x, float4 y, float4 z, float4 w);
	void SetViewMat(float4 x, float4 y, float4 z, float4 w);
	void SetLocalToWorldMat(float4 x, float4 y, float4 z, float4 w);
	void Compute(void);
	void Clear(void);
	void OnResize(uint32_t width, uint32_t height);
	void SetBounces(int bounces) { m_bounces = &bounces; }
	float* getFloatOutputData(void) { return m_finalOutputBuffer; }
	CUDABuffer* getFloatOutputDataDevice(void) { return &m_floatOutputBuffer_GPU; }
	CUDABuffer* getFloatAlbedoOutputDataDevice(void) { return &m_floatAlbedoBuffer_GPU; }
	CUDABuffer* getFloatNormalOutputDataDevice(void) { return &m_floatNormalBuffer_GPU; }

	uint32_t m_width;
	uint32_t m_height;

private:
	float m_aperture;
	float m_focusDist;
	const Scene* m_scene = nullptr;
	//GPU_Mesh* m_hostMesh;
	//GPU_Mesh* m_deviceMesh;
	size_t m_bufferSize;
	uint32_t* m_sampleIndex;
	int* m_samples;
	int* m_bounces;
	float3 m_cameraPos = { 0.0f, 0.0f, 0.0f };
	float3 m_cameraDir = {0.0f, 0.0f, -1.0f};
	float* m_invViewMat = nullptr;
	float* m_invProjMat = nullptr;
	float* m_viewMat = nullptr;
	float* m_localToWorldMat = nullptr;
	const RenderSettings* m_rendererSettings = nullptr;

	//Float image Buffers
	//CUDABuffer m_accumulationBuffer_GPU;    //Raw samples buffer
	CUDABuffer m_floatOutputBuffer_GPU;     //Final float output on the device
	CUDABuffer m_floatAlbedoBuffer_GPU;     //Final float albedo output on the device
	CUDABuffer m_floatNormalBuffer_GPU;     //Final float normal output on the device

	CUDABuffer m_deviceScene;
	CUDABuffer m_deviceSettings;

	ImageLoader m_imgLoaderEnv;
	ImageLoader m_imgLoaderTestTexture0;
	ImageLoader m_imgLoaderTestTexture1;
	ImageLoader m_imgLoaderTestTexture2;
	//float* m_skyTexture;

	float* m_finalOutputBuffer = nullptr;	//Final float output
};
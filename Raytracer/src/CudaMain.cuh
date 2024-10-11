#pragma once
#include <memory>

#include "scene.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "GPU_Mesh.h"
#include "ImageLoader.h"
#include "CudaBuffer.h"

class CudaRenderer
{
public:
	CudaRenderer(uint32_t width, uint32_t height, const Scene* scene, uint32_t* sampleIndex, int* samples, int* bounces)
		: m_sampleIndex(sampleIndex), m_samples(samples),
		  m_bounces(bounces), m_scene(scene)
	{
		cudaError_t cudaStatus = cudaErrorStartupFailure;

		OnResize(width, height);

		m_cameraPos = { 0.0f, 0.0f, 0.0f };
		m_invViewMat	  = new float[16];
		m_invProjMat	  = new float[16];
		m_viewMat		  = new float[16];
		m_localToWorldMat = new float[16];

		m_hostMesh = new GPU_Mesh();
		//m_hostMesh->LoadOBJFile("meshes/cube_quads.obj", 0u);
		m_hostMesh->LoadOBJFile("meshes/puffer.obj", 0u);
		//m_hostMesh->LoadOBJFile("meshes/angel.obj", 0u);
		//m_hostMesh->LoadOBJFile("meshes/buddha.obj", 0u);
		m_hostMesh->BuildBVH();

		m_deviceScene.alloc(sizeof(Scene));

		ImageLoader imgLoader;
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/river_rocks_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/paul_lobe_haus_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/clarens_midday_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/bridge1.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/xanderklinge_8k.raw", 8192, 4096);
		m_skyTexture = (float*)imgLoader.LoadImageFile("Images/studio_19.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/circus_arena_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/trekker_monument_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/tief_etz_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/industrial_workshop_foundry_8k.raw", 8192, 4096);
		//m_skyTexture = (float*)imgLoader.LoadImageFile("Images/Panels2k.raw", 8192, 4096);


		cudaMalloc(&m_deviceMesh, sizeof(GPU_Mesh));
		cudaMemcpy(m_deviceMesh, m_hostMesh, sizeof(GPU_Mesh), cudaMemcpyHostToDevice);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Mesh buffer copy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		GPU_Mesh::Triangle* dTris;
		cudaMalloc(&dTris, m_hostMesh->numTris * sizeof(GPU_Mesh::Triangle));
		cudaMemcpy(dTris, m_hostMesh->triangleBuffer, m_hostMesh->numTris * sizeof(GPU_Mesh::Triangle), cudaMemcpyHostToDevice);
		cudaMemcpy(&m_deviceMesh->triangleBuffer, &dTris, sizeof(GPU_Mesh::Triangle*), cudaMemcpyHostToDevice);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "GPU_Mesh::Triangle* copy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		GPU_Mesh::MeshInfo* dMeshInfo;
		cudaMalloc(&dMeshInfo, m_hostMesh->numMeshes * sizeof(GPU_Mesh::MeshInfo));
		cudaMemcpy(dMeshInfo, m_hostMesh->meshInfoBuffer, m_hostMesh->numMeshes * sizeof(GPU_Mesh::MeshInfo), cudaMemcpyHostToDevice);
		cudaMemcpy(&m_deviceMesh->meshInfoBuffer, &dMeshInfo, sizeof(GPU_Mesh::MeshInfo*), cudaMemcpyHostToDevice);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "GPU_Mesh::MeshInfo* copy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		GPU_Mesh::BVHNode* dBVHNodes;
		cudaMalloc(&dBVHNodes, m_hostMesh->nodesUsed * sizeof(GPU_Mesh::BVHNode));
		cudaMemcpy(dBVHNodes, m_hostMesh->bvhNode, m_hostMesh->nodesUsed * sizeof(GPU_Mesh::BVHNode), cudaMemcpyHostToDevice);
		cudaMemcpy(&m_deviceMesh->bvhNode, &dBVHNodes, sizeof(GPU_Mesh::BVHNode*), cudaMemcpyHostToDevice);

		uint32_t* dtriIdx;
		cudaMalloc(&dtriIdx, m_hostMesh->numTris * sizeof(uint32_t));
		cudaMemcpy(dtriIdx, m_hostMesh->triIdx, m_hostMesh->numTris * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(&m_deviceMesh->triIdx, &dtriIdx, sizeof(uint32_t*), cudaMemcpyHostToDevice);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "GPU_Mesh::BVHNode* copy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		}
	}

	~CudaRenderer()
	{
		cudaDeviceSynchronize();

		//m_accumulationBuffer_GPU.free();
		m_floatOutputBuffer_GPU.free();
		m_floatAlbedoBuffer_GPU.free();
		m_floatNormalBuffer_GPU.free();
		m_envTextureBuffer_GPU.free();

		m_deviceScene.free();

		cudaFree(m_deviceMesh);
	}

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
	const Scene* m_scene;
	GPU_Mesh* m_hostMesh;
	GPU_Mesh* m_deviceMesh;
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

	//Float image Buffers
	//CUDABuffer m_accumulationBuffer_GPU;    //Raw samples buffer
	CUDABuffer m_floatOutputBuffer_GPU;     //Final float output on the device
	CUDABuffer m_floatAlbedoBuffer_GPU;     //Final float albedo output on the device
	CUDABuffer m_floatNormalBuffer_GPU;     //Final float normal output on the device
	CUDABuffer m_envTextureBuffer_GPU;     //Final float normal output on the device

	CUDABuffer m_deviceScene;

	float* m_skyTexture;

	float* m_finalOutputBuffer = nullptr;	//Final float output
};
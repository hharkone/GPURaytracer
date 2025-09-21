#pragma once
#include <string>
#include <vector>
#include "cuda_runtime.h"
#include "cutil_math.cuh"
#include "CudaBuffer.h"

struct MeshBuffer
{
    uint32_t nodesUsed = 1u;
    uint32_t numTris   = 0u;

    CUdeviceptr bvhNode = 0u;
    CUdeviceptr triangleBuffer = 0u;
    CUdeviceptr meshInfoBuffer = 0u;
    CUdeviceptr indexBuffer = 0u;
};

class GPU_Mesh
{
public:

    void LoadOBJFile(const std::string& path);
    void LoadOBJFile(const std::string& path, int materialIndex);
    void BuildBVH();

    GPU_Mesh();
    ~GPU_Mesh();

    struct Triangle
    {
        float3 pos0, pos1, pos2;
        float3 n0, n1, n2;
        float3 c0, c1, c2;
        float2 uv0, uv1, uv2;
        uint16_t matID;

    private:
        //float2 padding0;
    };

    struct MeshInfo
    {
        uint32_t firstTriangleIndex = 0u;
        uint32_t triangleCount = 0u;
        float3 bboxMin = { 0.0f, 0.0f, 0.0f };
        float3 bboxMax = { 0.0f, 0.0f, 0.0f };
        //uint16_t materialIndex = 0u;
        uint16_t materialCount = 1u;
    };

    struct BVHNode
    {
        float3 aabbMin, aabbMax;
        uint leftFirst, triCount;
    };

    struct aabb
    {
        float3 bmin = make_float3(1e30f, 1e30f, 1e30f);
        float3 bmax = make_float3(-1e30f, -1e30f, -1e30f);

        void grow(float3 p) { bmin = cfminf(bmin, p), bmax = cfmaxf(bmax, p); }
        float area()
        {
            float3 e = bmax - bmin; // box extent
            return e.x * e.y + e.y * e.z + e.z * e.x;
        }
    };

    MeshBuffer deviceMesh;
    //uint32_t nodesUsed = 1u;
    //uint32_t numTris = 0u;
    BVHNode* bvhNode = nullptr;
    //std::vector<BVHNode> bvhNodeVector;
    Triangle* triangleBuffer = nullptr;
    MeshInfo* meshInfoBuffer = nullptr;
    float3* triangleCentroidScratchBuffer = nullptr;
    uint32_t* triIdx = nullptr;
    //uint32_t numMeshes = 0u;
    //uint32_t maxNodes;
    std::string filepath;

private:
    CUDABuffer CUDAbvhBuffer;
    CUDABuffer CUDAtriangleBuffer;
    CUDABuffer CUDAmeshInfoBuffer;
    CUDABuffer CUDAindexBuffer;

    uint32_t rootNodeIdx = 0;
    float EvaluateSAH(BVHNode& node, int axis, float pos);
    void UpdateNodeBounds(uint32_t nodeIdx);
    void Subdivide(uint32_t nodeIdx);
    void CalculateBbox(GPU_Mesh::MeshInfo& meshInfo);
    void Upload();
    bool TryLoadCache(const std::string& filename);
    bool TrySaveCache(const std::string& filename);
    bool loadedFromCache = false;
};
#pragma once
#include <string>
#include <vector>
#include "cuda_runtime.h"
#include "cutil_math.cuh"

class GPU_Mesh
{
public:

    void LoadOBJFile(const std::string& path, uint16_t materialIndex);
    void BuildBVH();

    struct Triangle
    {
        float3 pos0, pos1, pos2;
        float3 n0, n1, n2;
        float2 uv0, uv1, uv2;
        float3 centroid;

    private:
        float padding0 = 1337.0f;
    };

    struct MeshInfo
    {
        uint32_t firstTriangleIndex = 0u;
        uint32_t triangleCount = 0u;
        float3 bboxMin = { 0.0f, 0.0f, 0.0f };
        float3 bboxMax = { 0.0f, 0.0f, 0.0f };
        uint16_t materialIndex = 0u;
    };

    struct BVHNode
    {
        float3 aabbMin, aabbMax;
        uint leftFirst, triCount;
    };

    BVHNode* bvhNode = nullptr;
    std::vector<BVHNode> bvhNodeVector;
    Triangle* triangleBuffer = nullptr;
    MeshInfo* meshInfoBuffer = nullptr;
    uint32_t numMeshes = 0u;
    uint32_t numTris = 0u;
    uint32_t nodesUsed = 1u;
    uint32_t maxNodes;
    uint32_t* triIdx = nullptr;
    int buildStackPtr;

private:
    uint32_t rootNodeIdx = 0;
    void UpdateNodeBounds(uint32_t nodeIdx);
    void Subdivide(uint32_t nodeIdx);
    void CalculateBbox(GPU_Mesh::MeshInfo& meshInfo);
};
#pragma once
#include <string>
#include "cuda_runtime.h"

class GPU_Mesh
{
public:

    void LoadOBJFile(const std::string& path);

    struct Triangle
    {
        float3 pos0, pos1, pos2;
        float3 n0, n1, n2;
        float2 uv0, uv1, uv2;
    };

    struct MeshInfo
    {
        size_t firstTriangleIndex = 0u;
        size_t triangleCount = 0u;
        float3 bboxMin = { 0.0f, 0.0f, 0.0f };
        float3 bboxMax = { 0.0f, 0.0f, 0.0f };
        size_t materialIndex = 0u;
    };

    Triangle* triangleBuffer = nullptr;
    MeshInfo* meshInfoBuffer = nullptr;
    size_t numMeshes = 0u;
    size_t numTris = 0u;

private:

    void CalculateBbox(GPU_Mesh::MeshInfo& meshInfo);

};
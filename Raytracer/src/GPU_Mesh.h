#pragma once
#include <string>
#include "cuda_runtime.h"

//#include "device_launch_parameters.h"
//#include "device_atomic_functions.h"

class GPU_Mesh
{
public:

    struct GPU_MeshList
    {
        float* vertexBuffer = nullptr;
        size_t vertexStride = 8u;
        size_t meshCount = 0u;
        size_t* vertexCounts = nullptr;
        size_t* meshOffsets = nullptr;
        float3* bboxMins = nullptr;
        float3* bboxMaxs = nullptr;
        size_t* materialIndices = nullptr;
    };

    void CalculateBbox();
    void LoadOBJFile(const std::string& path);
    void AddMeshToMeshList(GPU_MeshList& mlist, GPU_Mesh& mesh);

    size_t vertexCount = 0u;
    float* vertexBuffer = nullptr;
    size_t materialIndex = 0u;
    float3 bboxMin = { 0.0f, 0.0f, 0.0f };
    float3 bboxMax = { 0.0f, 0.0f, 0.0f };
};
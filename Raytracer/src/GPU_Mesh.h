#pragma once
#include <string>
#include "cuda_runtime.h"

//#include "device_launch_parameters.h"
//#include "device_atomic_functions.h"

class GPU_Mesh
{
public:
    GPU_Mesh(float* vertexData, size_t vertexCount, size_t materialIndex, float3 bboxMin, float3 bboxMax)
        : vertexCount(vertexCount), materialIndex(materialIndex), bboxMin(bboxMin), bboxMax(bboxMax)
    {
        vertexBuffer = new float[vertexCount];
    }

    GPU_Mesh()
    {
    }

    void CalculateBbox();
    void LoadOBJFile(const std::string& path);

    size_t vertexCount = 0u;
    size_t vertexStride = 8u;
    float* vertexBuffer = nullptr;
    size_t materialIndex = 0u;
    float3 bboxMin = { 0.0f, 0.0f, 0.0f };
    float3 bboxMax = { 0.0f, 0.0f, 0.0f };
};

/*

                //Positions
                vertexBuffer[i]   = pos[f1 - 1].x;
                vertexBuffer[i+1] = pos[f1 - 1].y;
                vertexBuffer[i+2] = pos[f1 - 1].z;

                i += 3;

                vertexBuffer[i]   = pos[f4 - 1].x;
                vertexBuffer[i+1] = pos[f4 - 1].y;
                vertexBuffer[i+2] = pos[f4 - 1].z;

                i += 3;

                vertexBuffer[i]     = pos[f7 - 1].x;
                vertexBuffer[i + 1] = pos[f7 - 1].y;
                vertexBuffer[i + 2] = pos[f7 - 1].z;

                i += 3;

                //Normals
                vertexBuffer[i]     = pos[f3 - 1].x;
                vertexBuffer[i + 1] = pos[f3 - 1].y;
                vertexBuffer[i + 2] = pos[f3 - 1].z;

                i += 3;

                vertexBuffer[i]     = pos[f6 - 1].x;
                vertexBuffer[i + 1] = pos[f6 - 1].y;
                vertexBuffer[i + 2] = pos[f6 - 1].z;

                i += 3;

                vertexBuffer[i]     = pos[f9 - 1].x;
                vertexBuffer[i + 1] = pos[f9 - 1].y;
                vertexBuffer[i + 2] = pos[f9 - 1].z;

                //UVs
                vertexBuffer[i]     = pos[f2 - 1].x;
                vertexBuffer[i + 1] = pos[f2 - 1].y;

                i += 2;

                vertexBuffer[i]     = pos[f5 - 1].x;
                vertexBuffer[i + 1] = pos[f5 - 1].y;

                i += 2;

                vertexBuffer[i]     = pos[f8 - 1].x;
                vertexBuffer[i + 1] = pos[f8 - 1].y;

                i += 2;

*/
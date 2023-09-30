#include <iostream>
#include <fstream>
#include <vector>

#include "GPU_Mesh.h"

void GPU_Mesh::CalculateBbox()
{
    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = std::numeric_limits<float>::max();
    max_x = max_y = max_z = std::numeric_limits<float>::min();

    for (size_t i = 0u; i < vertexCount * 3u; i+=8u)
    {
        if (vertexBuffer[i + 0] < min_x) min_x = vertexBuffer[i + 0];
        if (vertexBuffer[i + 0] > max_x) max_x = vertexBuffer[i + 0];
        if (vertexBuffer[i + 1] < min_y) min_y = vertexBuffer[i + 1];
        if (vertexBuffer[i + 1] > max_y) max_y = vertexBuffer[i + 1];
        if (vertexBuffer[i + 2] < min_z) min_z = vertexBuffer[i + 2];
        if (vertexBuffer[i + 2] > max_z) max_z = vertexBuffer[i + 2];
    }

    bboxMin = make_float3(min_x, min_y, min_z);
    bboxMax = make_float3(max_x, max_y, max_z);
}

void GPU_Mesh::LoadOBJFile(const std::string& path)
{
    vertexCount = 0u;

    std::ifstream infile(path, std::ifstream::in);
    std::string line;

    float x, y, z;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9;
    std::string s;

    std::vector<float3> pos;
    std::vector<float3> normal;
    std::vector<float2> uv;
    std::vector<int> tris;

    while (std::getline(infile, line))
    {
        std::string test = line.substr(0, 2);
        if (test == "v ")
        {
            if (sscanf_s(line.c_str(), "v %f %f %f\n", &x, &y, &z) == 3)
                pos.push_back(make_float3(x, y, z));
        }
        else if (test == "vn")
        {
            if (sscanf_s(line.c_str(), "vn %f %f %f\n", &x, &y, &z) == 3)
                normal.push_back(make_float3(x, y, z));
        }
        else if (test == "vt")
        {
            if (sscanf_s(line.c_str(), "vt %f %f\n", &x, &y) == 2)
                uv.push_back(make_float2(x, y));
        }
        else if (test == "f ")
        {
            if (sscanf_s(line.c_str(), "f %i/%i/%i %i/%i/%i %i/%i/%i\n", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9) == 9)
            {
                tris.push_back(f1 - 1);
                tris.push_back(f3 - 1);
                tris.push_back(f2 - 1);

                tris.push_back(f4 - 1);
                tris.push_back(f6 - 1);
                tris.push_back(f5 - 1);

                tris.push_back(f7 - 1);
                tris.push_back(f9 - 1);
                tris.push_back(f8 - 1);

                vertexCount += 3u;
            }

        }

    }
    //Construct the vertexBuffer
    size_t bufferSize = vertexCount * (3 + 3 + 2) * sizeof(float);
    vertexBuffer = new float[bufferSize];
    memset(vertexBuffer, 0, bufferSize);

    for (size_t i = 0, j = 0; i < vertexCount * 3u; i+=3, j+=8)
    {
        //Vertex
        vertexBuffer[j]     = pos[tris[i]].x;
        vertexBuffer[j + 1] = pos[tris[i]].y;
        vertexBuffer[j + 2] = pos[tris[i]].z;

        vertexBuffer[j + 3] = normal[tris[i + 1]].x;
        vertexBuffer[j + 4] = normal[tris[i + 1]].y;
        vertexBuffer[j + 5] = normal[tris[i + 1]].z;

        vertexBuffer[j + 6] = uv[tris[i + 2]].x;
        vertexBuffer[j + 7] = uv[tris[i + 2]].y;
    }


    CalculateBbox();
}

void GPU_Mesh::AddMeshToMeshList(GPU_Mesh::GPU_MeshList& mlist, GPU_Mesh& mesh)
{

    size_t meshcount = mlist.meshCount;
    size_t meshOffset = 0u;

    for (size_t i = 0u; i < meshcount; i++)
    {
        meshOffset = mlist.vertexCounts[i] * mlist.vertexStride * sizeof(float);
    }

    size_t thisMeshOffset = mesh.vertexCount * mlist.vertexStride * sizeof(float);
    size_t* newOffsets = new size_t[meshcount + 1u];
    memcpy(newOffsets, mlist.meshOffsets, meshcount * sizeof(size_t));
    memcpy(&newOffsets[meshcount * sizeof(size_t)], &thisMeshOffset, sizeof(size_t));

    mlist.meshOffsets = newOffsets;

    float* newVbo = new float[meshOffset + mesh.vertexCount * mlist.vertexStride * sizeof(float)];
    memcpy(newVbo, mlist.vertexBuffer, meshOffset);
    memcpy(&newVbo[meshOffset], mesh.vertexBuffer, +mesh.vertexCount * mlist.vertexStride * sizeof(float));

    mlist.vertexBuffer = newVbo;

    size_t* newVertexCounts = new size_t[meshcount * sizeof(size_t) + sizeof(size_t)];
    memcpy(newVertexCounts, mlist.vertexCounts, meshcount * sizeof(size_t));
    memcpy(&newVertexCounts[meshcount * sizeof(size_t)], &mesh.vertexCount, sizeof(size_t));

    mlist.vertexCounts = newVertexCounts;

    float3* newBboxMins = new float3[meshcount * sizeof(float3) + sizeof(float3)];
    memcpy(newBboxMins, mlist.bboxMins, meshcount * sizeof(float3));
    memcpy(&newBboxMins[meshcount * sizeof(float3)], &mesh.bboxMin, sizeof(float3));

    float3* newBboxMaxs = new float3[meshcount * sizeof(float3) + sizeof(float3)];
    memcpy(newBboxMaxs, mlist.bboxMaxs, meshcount * sizeof(float3));
    memcpy(&newBboxMaxs[meshcount * sizeof(float3)], &mesh.bboxMax, sizeof(float3));

    mlist.bboxMins = newBboxMins;
    mlist.bboxMaxs = newBboxMaxs;
    mlist.meshCount++;
    mesh.hasChanged = true;
}
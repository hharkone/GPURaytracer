#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "GPU_Mesh.h"
#include "cutil_math.cuh"

void GPU_Mesh::CalculateBbox(GPU_Mesh::MeshInfo& meshInfo)
{
    float min = FLT_MIN;
    float max = FLT_MAX;
    
    float3 maxVec = make_float3(max, max, max);
    float3 minVec = make_float3(min, min, min);

    meshInfo.bboxMax = minVec;
    meshInfo.bboxMin = maxVec;

    for (size_t i = meshInfo.firstTriangleIndex; i < (meshInfo.firstTriangleIndex + meshInfo.triangleCount); i++)
    {
        meshInfo.bboxMin = cfminf(meshInfo.bboxMin, triangleBuffer[i].pos0);
        meshInfo.bboxMin = cfminf(meshInfo.bboxMin, triangleBuffer[i].pos1);
        meshInfo.bboxMin = cfminf(meshInfo.bboxMin, triangleBuffer[i].pos2);
        meshInfo.bboxMax = cfmaxf(meshInfo.bboxMax, triangleBuffer[i].pos0);
        meshInfo.bboxMax = cfmaxf(meshInfo.bboxMax, triangleBuffer[i].pos1);
        meshInfo.bboxMax = cfmaxf(meshInfo.bboxMax, triangleBuffer[i].pos2);
    }
}


void GPU_Mesh::LoadOBJFile(const std::string& path)
{
    size_t importTriangleCount = 0u;

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

                importTriangleCount += 1u;
                numTris++;
            }

        }

    }

    size_t meshTriCount = 0u;

    for (size_t i = 0u; i < numMeshes; i++)
    {
        meshTriCount += meshInfoBuffer[i].triangleCount;
    }

    //Allocate new triangle buffer that can encompass all previous triangles + new imported ones.
    Triangle* newTriBuf = new Triangle[meshTriCount + importTriangleCount];
    std::memset(newTriBuf, 0, (meshTriCount + importTriangleCount) * sizeof(Triangle));
    std::memcpy(newTriBuf, triangleBuffer, meshTriCount * sizeof(Triangle));

    for (size_t i = 0, j = 0; i < importTriangleCount; i++, j+=9)
    {
        Triangle newTri;

        newTri.pos0 =    pos[tris[j + 0u]];
        newTri.n0   = normal[tris[j + 1u]];
        newTri.uv0  =     uv[tris[j + 2u]];

        newTri.pos1 =    pos[tris[j + 3u]];
        newTri.n1   = normal[tris[j + 4u]];
        newTri.uv1  =     uv[tris[j + 5u]];

        newTri.pos2 =    pos[tris[j + 6u]];
        newTri.n2   = normal[tris[j + 7u]];
        newTri.uv2  =     uv[tris[j + 8u]];

        std::memcpy(&newTriBuf[meshTriCount + i], &newTri, sizeof(Triangle));
    }

    triangleBuffer = newTriBuf;

    //Allocate new meshInfo buffer that can encompass all previous meshInfos + the new imported one.

    MeshInfo newMeshInfo;
    newMeshInfo.firstTriangleIndex = meshTriCount;
    newMeshInfo.triangleCount = importTriangleCount;

    CalculateBbox(newMeshInfo);

    MeshInfo* newMeshInfoBuf = new MeshInfo[numMeshes + 1u];
    std::memcpy(newMeshInfoBuf, meshInfoBuffer, numMeshes * sizeof(MeshInfo));
    std::memcpy(&newMeshInfoBuf[numMeshes], &newMeshInfo, sizeof(MeshInfo));

    meshInfoBuffer = newMeshInfoBuf;

    numMeshes++;
}

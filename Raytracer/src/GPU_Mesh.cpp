#include <iostream>
#include <fstream>
#include <memory>

#include "GPU_Mesh.h"

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

void GPU_Mesh::LoadOBJFile(const std::string& path, uint16_t materialIndex)
{
    uint32_t importTriangleCount = 0u;

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
                uv.push_back(make_float2(materialIndex, materialIndex));
        }
        else if (test == "f ")
        {
            if (sscanf_s(line.c_str(), "f %i//%i %i//%i %i//%i\n", &f1, &f2, &f3, &f4, &f5, &f6) == 6 && uv.size() == 0)
            {
                tris.push_back(f1 - 1);
                tris.push_back(f2 - 1);

                tris.push_back(f3 - 1);
                tris.push_back(f4 - 1);

                tris.push_back(f5 - 1);
                tris.push_back(f6 - 1);

                importTriangleCount += 1u;
                numTris++;
            }
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

    if (importTriangleCount == 0u)
        return;

    uint32_t meshTriCount = 0u;

    for (uint32_t i = 0u; i < numMeshes; i++)
    {
        meshTriCount += meshInfoBuffer[i].triangleCount;
    }

    //Allocate new triangle buffer that can encompass all previous triangles + new imported ones.
    Triangle* newTriBuf = new Triangle[meshTriCount + importTriangleCount];
    std::memset(newTriBuf, 0, (meshTriCount + importTriangleCount) * sizeof(Triangle));
    std::memcpy(newTriBuf, triangleBuffer, meshTriCount * sizeof(Triangle));

    if (uv.size() == 0)
    {
        for (uint32_t i = 0, j = 0; i < importTriangleCount; i++, j += 6)
        {
            Triangle newTri;

            newTri.pos0 = pos[tris[j + 0u]];
            newTri.n0 = normal[tris[j + 1u]];
            newTri.uv0 = {0.0f, 0.0f};

            newTri.pos1 = pos[tris[j + 2u]];
            newTri.n1 = normal[tris[j + 3u]];
            newTri.uv1 = { 0.0f, 0.0f };

            newTri.pos2 = pos[tris[j + 4u]];
            newTri.n2 = normal[tris[j + 5u]];
            newTri.uv2 = { 0.0f, 0.0f };

            std::memcpy(&newTriBuf[meshTriCount + i], &newTri, sizeof(Triangle));
        }
    }
    else
    {
        for (uint32_t i = 0, j = 0; i < importTriangleCount; i++, j += 9)
        {
            Triangle newTri;

            newTri.pos0 = pos[tris[j + 0u]];
            newTri.n0 = normal[tris[j + 1u]];
            newTri.uv0 = uv[tris[j + 2u]];

            newTri.pos1 = pos[tris[j + 3u]];
            newTri.n1 = normal[tris[j + 4u]];
            newTri.uv1 = uv[tris[j + 5u]];

            newTri.pos2 = pos[tris[j + 6u]];
            newTri.n2 = normal[tris[j + 7u]];
            newTri.uv2 = uv[tris[j + 8u]];

            std::memcpy(&newTriBuf[meshTriCount + i], &newTri, sizeof(Triangle));
        }
    }

    triangleBuffer = newTriBuf;

    //Allocate new meshInfo buffer that can encompass all previous meshInfos + the new imported one.

    MeshInfo newMeshInfo;
    newMeshInfo.firstTriangleIndex = meshTriCount;
    newMeshInfo.triangleCount = importTriangleCount;
    newMeshInfo.materialIndex = materialIndex;

    CalculateBbox(newMeshInfo);

    MeshInfo* newMeshInfoBuf = new MeshInfo[numMeshes + 1u];
    std::memcpy(newMeshInfoBuf, meshInfoBuffer, numMeshes * sizeof(MeshInfo));
    std::memcpy(&newMeshInfoBuf[numMeshes], &newMeshInfo, sizeof(MeshInfo));

    meshInfoBuffer = newMeshInfoBuf;

    numMeshes++;
}

void GPU_Mesh::UpdateNodeBounds(uint32_t nodeIdx)
{
    BVHNode& node = bvhNode[nodeIdx];
    //BVHNode& node = bvhNodeVector.at(nodeIdx);
    node.aabbMin = make_float3(1e30f, 1e30f, 1e30f);
    node.aabbMax = make_float3(-1e30f, -1e30f, -1e30f);

    for (uint32_t first = node.leftFirst, i = 0; i < node.triCount; i++)
    {
        uint32_t leafTriIdx = triIdx[first + i];
        Triangle& leafTri = triangleBuffer[leafTriIdx];
        node.aabbMin = cfminf(node.aabbMin, leafTri.pos0),
        node.aabbMin = cfminf(node.aabbMin, leafTri.pos1),
        node.aabbMin = cfminf(node.aabbMin, leafTri.pos2),
        node.aabbMax = cfmaxf(node.aabbMax, leafTri.pos0),
        node.aabbMax = cfmaxf(node.aabbMax, leafTri.pos1),
        node.aabbMax = cfmaxf(node.aabbMax, leafTri.pos2);
    }
}

void GPU_Mesh::Subdivide(uint32_t nodeIdx)
{
    // terminate recursion
    BVHNode& node = bvhNode[nodeIdx];
    //BVHNode& node = bvhNodeVector.at(nodeIdx);

    if (node.triCount <= 2)
    {
        return;
    }
    if (nodesUsed >= 7)
    {
        int debug = 1;
    }

    // determine split axis and position
    float3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z >= (&extent.x)[axis]) axis = 2;
    float splitPos = (&node.aabbMin.x)[axis] + (&extent.x)[axis] * 0.5f;

    // in-place partition
    uint32_t i = node.leftFirst;
    uint32_t j = i + node.triCount - 1;
    while (i <= j)
    {
        if ((&triangleBuffer[triIdx[i]].centroid.x)[axis] < splitPos)
            i++;
        else
            std::swap(triIdx[i], triIdx[j--]);
    }

    // abort split if one of the sides is empty
    uint32_t leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) return;

    // create child nodes
    uint32_t leftChildIdx = nodesUsed++;
    uint32_t rightChildIdx = nodesUsed++;
    bvhNode[leftChildIdx].leftFirst = node.leftFirst;
    bvhNode[leftChildIdx].triCount = leftCount;
    bvhNode[rightChildIdx].leftFirst = i;
    bvhNode[rightChildIdx].triCount = node.triCount - leftCount;

    //bvhNodeVector.at(leftChildIdx).leftFirst = node.leftFirst;
    //bvhNodeVector.at(leftChildIdx).triCount = leftCount;
    //bvhNodeVector.at(rightChildIdx).leftFirst = i;
    //bvhNodeVector.at(rightChildIdx).triCount = node.triCount - leftCount;

    node.leftFirst = leftChildIdx;
    node.triCount = 0;

    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);
    // recurse
    Subdivide(leftChildIdx);
    Subdivide(rightChildIdx);
}

void GPU_Mesh::BuildBVH()
{
    if (numTris == 0)
    {
        return;
    }

    bvhNode = new BVHNode[numTris * 2 - 1];
    triIdx = new uint32_t[numTris];
    //bvhNodeVector.resize(numTris * 2 - 1);

    for (uint32_t i = 0; i < numTris; i++)
    {
        triangleBuffer[i].centroid = (triangleBuffer[i].pos0 + triangleBuffer[i].pos1 + triangleBuffer[i].pos2) * 0.3333f;
        triIdx[i] = i;
    }

    // assign all triangles to root node
    BVHNode& root = bvhNode[rootNodeIdx];
    //BVHNode& root = bvhNodeVector.at(rootNodeIdx);

    root.leftFirst = 0;
    root.triCount = numTris;

    UpdateNodeBounds(rootNodeIdx);
    // subdivide recursively
    Subdivide(rootNodeIdx);

    //Resize
    BVHNode* newArr = new BVHNode[nodesUsed];
    memcpy(newArr, bvhNode, nodesUsed * sizeof(BVHNode));
    delete[] bvhNode;
    bvhNode = newArr;
}
#include <iostream>
#include <fstream>
#include <memory>
#include <map>

#include "zlib.h"
#include "GPU_Mesh.h"

//#define USE_UNOPTIMIZED_BVH_SPLITTING

void GPU_Mesh::Upload()
{
    CUDAbvhBuffer.alloc_and_upload(bvhNode, deviceMesh.nodesUsed);
    deviceMesh.bvhNode = CUDAbvhBuffer.d_pointer();

    CUDAindexBuffer.alloc_and_upload(triIdx, deviceMesh.numTris);
    deviceMesh.indexBuffer = CUDAindexBuffer.d_pointer();

    CUDAtriangleBuffer.alloc_and_upload(triangleBuffer, deviceMesh.numTris);
    deviceMesh.triangleBuffer = CUDAtriangleBuffer.d_pointer();

    CUDAmeshInfoBuffer.alloc_and_upload(meshInfoBuffer, 1u);
    deviceMesh.meshInfoBuffer = CUDAmeshInfoBuffer.d_pointer();
}

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
    LoadOBJFile(path, -1);
}

void GPU_Mesh::LoadOBJFile(const std::string& path, int materialIndex)
{
    filepath = path;

    if (TryLoadCache(filepath))
    {
        loadedFromCache = true;
        Upload();
        return;
    }

    uint32_t importTriangleCount = 0u;

    std::ifstream infile(path, std::ifstream::in);

    if (infile.fail())
    {
        fprintf(stderr, "Mesh importing faled, invalid filepath.\n");
        return;
    }

    std::string line;
    std::string matString;

    float x, y, z, r, g, b;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
    uint16_t materialGroup = 0u;
    char s[256] = { 0 };

    std::vector<float3> pos;
    std::vector<float3> normal;
    std::vector<float3> color;
    std::vector<float2> uv;
    std::vector<int> tris;
    std::vector<uint16_t> matIndex;
    std::map<std::string, uint16_t> materialMap;

    while (std::getline(infile, line))
    {
        std::string test = line.substr(0, 2);
        if (test == "v ")
        {
            if (sscanf_s(line.c_str(), "v %f %f %f %f %f %f\n", &x, &y, &z, &r, &g, &b) == 6) //vertex position with color data
            {
                pos.push_back(make_float3(x, y, z));
                color.push_back(make_float3(r, g, b));
            }
            else if (sscanf_s(line.c_str(), "v %f %f %f\n", &x, &y, &z) == 3) //vertex position
            {
                pos.push_back(make_float3(x, y, z));
                color.push_back(make_float3(1.0f, 1.0f, 1.0f));
            }

        }
        else if (test == "vn")
        {
            if (sscanf_s(line.c_str(), "vn %f %f %f\n", &x, &y, &z) == 3) // vertex normal
                normal.push_back(make_float3(x, y, z));
        }
        else if (test == "vt")
        {
            if (sscanf_s(line.c_str(), "vt %f %f\n", &x, &y) == 2) // vertex UV
                uv.push_back(make_float2(x, y));
        }
        else if (test == "g ") // face material
        {
            if (sscanf_s(line.c_str(), "%*s %s\n", s, (unsigned)line.length()) == 1)
            {
                if (!materialMap[s]) // if this material name is not found, add it and incement unique material count.
                {
                    materialMap.insert_or_assign(s, materialGroup++);
                }
            }
        }
        else if (test == "f ")
        {
            //Quads, with UV
            if (sscanf_s(line.c_str(), "f %i/%i/%i %i/%i/%i %i/%i/%i %i/%i/%i\n", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12) == 12)
            {
                tris.push_back(abs(f1) - 1);
                tris.push_back(abs(f3) - 1);
                tris.push_back(abs(f2) - 1);

                tris.push_back(abs(f4) - 1);
                tris.push_back(abs(f6) - 1);
                tris.push_back(abs(f5) - 1);

                tris.push_back(abs(f7) - 1);
                tris.push_back(abs(f9) - 1);
                tris.push_back(abs(f8) - 1);

                tris.push_back(abs(f7) - 1);
                tris.push_back(abs(f9) - 1);
                tris.push_back(abs(f8) - 1);

                tris.push_back(abs(f10) - 1);
                tris.push_back(abs(f12) - 1);
                tris.push_back(abs(f11) - 1);

                tris.push_back(abs(f1) - 1);
                tris.push_back(abs(f3) - 1);
                tris.push_back(abs(f2) - 1);


                importTriangleCount += 2u;
                deviceMesh.numTris += 2u;
            }
            //Tris, with UV
            else if (sscanf_s(line.c_str(), "f %i/%i/%i %i/%i/%i %i/%i/%i\n", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9) == 9)
            {
                tris.push_back(abs(f1) - 1);
                tris.push_back(abs(f3) - 1);
                tris.push_back(abs(f2) - 1);

                tris.push_back(abs(f4) - 1);
                tris.push_back(abs(f6) - 1);
                tris.push_back(abs(f5) - 1);

                tris.push_back(abs(f7) - 1);
                tris.push_back(abs(f9) - 1);
                tris.push_back(abs(f8) - 1);

                importTriangleCount += 1u;
                deviceMesh.numTris++;
            }
            //Quads, no UV
            else if (sscanf_s(line.c_str(), "f %i//%i %i//%i %i//%i %i//%i\n", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8) == 8 && uv.size() == 0)
            {
                tris.push_back(abs(f1) - 1);
                tris.push_back(abs(f2) - 1);

                tris.push_back(abs(f3) - 1);
                tris.push_back(abs(f4) - 1);

                tris.push_back(abs(f5) - 1);
                tris.push_back(abs(f6) - 1);

                tris.push_back(abs(f5) - 1);
                tris.push_back(abs(f6) - 1);

                tris.push_back(abs(f7) - 1);
                tris.push_back(abs(f8) - 1);

                tris.push_back(abs(f1) - 1);
                tris.push_back(abs(f2) - 1);

                importTriangleCount += 2u;
                deviceMesh.numTris += 2u;
            }
            //Tris, no UV
            else if (sscanf_s(line.c_str(), "f %i//%i %i//%i %i//%i\n", &f1, &f2, &f3, &f4, &f5, &f6) == 6 && uv.size() == 0)
            {
                tris.push_back(abs(f1) - 1);
                tris.push_back(abs(f2) - 1);

                tris.push_back(abs(f3) - 1);
                tris.push_back(abs(f4) - 1);

                tris.push_back(abs(f5) - 1);
                tris.push_back(abs(f6) - 1);

                importTriangleCount += 1u;
                deviceMesh.numTris++;
            }
            if (!materialMap.empty())
            {
                matIndex.push_back(materialGroup);
            }
        }

    }

    if (importTriangleCount == 0u)
        return;

    uint32_t meshTriCount = 0u;

    //for (uint32_t i = 0u; i < deviceMesh.numMeshes; i++)
    //{
    //    meshTriCount += meshInfoBuffer[i].triangleCount;
    //}

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
            newTri.c0 = color[tris[j + 0u]];
            newTri.n0 = normal[tris[j + 1u]];
            newTri.uv0 = {0.0f, 0.0f};

            newTri.pos1 = pos[tris[j + 2u]];
            newTri.c1 = color[tris[j + 2u]];
            newTri.n1 = normal[tris[j + 3u]];
            newTri.uv1 = { 0.0f, 0.0f };

            newTri.pos2 = pos[tris[j + 4u]];
            newTri.c2 = color[tris[j + 4u]];
            newTri.n2 = normal[tris[j + 5u]];
            newTri.uv2 = { 0.0f, 0.0f };

            if (!(materialIndex == -1))
            {
                newTri.matID = (uint16_t)materialIndex;
            }
            else
            {
                if (!matIndex.empty())
                {
                    newTri.matID = matIndex[i] + 4u;
                }
                else
                {
                    newTri.matID = 4u;
                }
            }

            std::memcpy(&newTriBuf[meshTriCount + i], &newTri, sizeof(Triangle));
        }
    }
    else
    {
        for (uint32_t i = 0, j = 0; i < importTriangleCount; i++, j += 9)
        {
            Triangle newTri;

            newTri.pos0 = pos[tris[j + 0u]];
            newTri.c0 = color[tris[j + 0u]];
            newTri.n0 = normal[tris[j + 1u]];
            newTri.uv0 = uv[tris[j + 2u]];

            newTri.pos1 = pos[tris[j + 3u]];
            newTri.c1 = color[tris[j + 3u]];
            newTri.n1 = normal[tris[j + 4u]];
            newTri.uv1 = uv[tris[j + 5u]];

            newTri.pos2 = pos[tris[j + 6u]];
            newTri.c2 = color[tris[j + 6u]];
            newTri.n2 = normal[tris[j + 7u]];
            newTri.uv2 = uv[tris[j + 8u]];

            if (!(materialIndex == -1))
            {
                newTri.matID = (uint16_t)materialIndex;
            }
            else
            {
                if (!matIndex.empty())
                {
                    newTri.matID = matIndex[i] + 4u;
                }
                else
                {
                    newTri.matID = 4u;
                }
            }

            std::memcpy(&newTriBuf[meshTriCount + i], &newTri, sizeof(Triangle));
        }
    }

    triangleBuffer = newTriBuf;

    //Allocate new meshInfo buffer that can encompass all previous meshInfos + the new imported one.

    MeshInfo newMeshInfo;
    newMeshInfo.firstTriangleIndex = meshTriCount;
    newMeshInfo.triangleCount = importTriangleCount;
    //newMeshInfo.materialIndex = materialIndex;
    newMeshInfo.materialCount = (materialGroup == 0 ? 1u : materialGroup + 1u);

    CalculateBbox(newMeshInfo);

    meshInfoBuffer = &newMeshInfo;

    //MeshInfo* newMeshInfoBuf = new MeshInfo[deviceMesh.numMeshes + 1u];
    //std::memcpy(newMeshInfoBuf, meshInfoBuffer, deviceMesh.numMeshes * sizeof(MeshInfo));
    //std::memcpy(&newMeshInfoBuf[deviceMesh.numMeshes], &newMeshInfo, sizeof(MeshInfo));

    //meshInfoBuffer = newMeshInfoBuf;

    //deviceMesh.numMeshes++;

    BuildBVH();
    Upload();
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

float GPU_Mesh::EvaluateSAH(BVHNode& node, int axis, float pos)
{
    // determine triangle counts and bounds for this split candidate
    aabb leftBox, rightBox;
    int leftCount = 0, rightCount = 0;
    for (uint i = 0; i < node.triCount; i++)
    {
        Triangle& triangle = triangleBuffer[triIdx[node.leftFirst + i]];
        //if (triangle.centroid[axis] < pos)
        if ((&triangleCentroidScratchBuffer[triIdx[node.leftFirst + i]].x)[axis] < pos)
        {
            leftCount++;
            leftBox.grow(triangle.pos0);
            leftBox.grow(triangle.pos1);
            leftBox.grow(triangle.pos2);
        }
        else
        {
            rightCount++;
            rightBox.grow(triangle.pos0);
            rightBox.grow(triangle.pos1);
            rightBox.grow(triangle.pos2);
        }
    }
    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0 ? cost : 1e30f;
}

void GPU_Mesh::Subdivide(uint32_t nodeIdx)
{
    // terminate recursion
    BVHNode& node = bvhNode[nodeIdx];

    if (nodeIdx == 190u)
    {
        float debugx = 1.0f;
    }
    /*
    if (node.triCount >= 0u)
    {
        maxNodeTris = min(maxNodeTris, node.triCount);
    }
    */

    if (node.triCount <= 8)
    {
        return;
    }

#ifdef USE_UNOPTIMIZED_BVH_SPLITTING

    // determine split axis and position
    float3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z >= (&extent.x)[axis]) axis = 2;
    float splitPos = (&node.aabbMin.x)[axis] + (&extent.x)[axis] * 0.5f;

#else
    int bestAxis = -1;
    float bestPos = 0, bestCost = 1e30f;
    uint splitCount = 2u;
    for (int axis = 0; axis < 3; axis++) for (uint i = 1u; i < splitCount; i++)
    {
        float xx = (1.0f / float(splitCount + 1u)) * float(i);
        float3 candidatePos = lerp(node.aabbMin, node.aabbMax, xx);

        float cost = EvaluateSAH(node, axis, (&candidatePos.x)[axis]);
        if (cost < bestCost)
            bestPos = (&candidatePos.x)[axis], bestAxis = axis, bestCost = cost;
    }
    int axis = bestAxis;
    float splitPos = bestPos;

    float3 e = node.aabbMax - node.aabbMin; // extent of parent
    float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
    float parentCost = node.triCount * parentArea;

    if (bestCost >= parentCost) return;
#endif

    // in-place partition
    uint32_t i = node.leftFirst;
    uint32_t j = i + node.triCount - 1;
    while (i <= j)
    {
        if ((&triangleCentroidScratchBuffer[triIdx[i]].x)[axis] < splitPos)
            i++;
        else
            std::swap(triIdx[i], triIdx[j--]);
    }

    // abort split if one of the sides is empty
    uint32_t leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) return;

    // create child nodes
    uint32_t leftChildIdx = deviceMesh.nodesUsed++;
    uint32_t rightChildIdx = deviceMesh.nodesUsed++;
    bvhNode[leftChildIdx].leftFirst = node.leftFirst;
    bvhNode[leftChildIdx].triCount = leftCount;
    bvhNode[rightChildIdx].leftFirst = i;
    bvhNode[rightChildIdx].triCount = node.triCount - leftCount;

    node.leftFirst = leftChildIdx;
    node.triCount = 0;

    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);
    // recurse
    Subdivide(leftChildIdx);
    Subdivide(rightChildIdx);
}

std::string CacheFilePath(const std::string& filename)
{
    std::string BVHcacheFilename(filename);
    size_t lastindex = BVHcacheFilename.find_last_of(".");
    return BVHcacheFilename.substr(0, lastindex) += ".bvh";
}

bool GPU_Mesh::TryLoadCache(const std::string& filename)
{
    std::string cacheFile = CacheFilePath(filename);
    FILE* fp = fopen(cacheFile.c_str(), "rb");

    if (fp == NULL)
    {
        fprintf(stderr, "No BVH cache exists.\n");
        return false;
    }

    // BVH has been built already and stored in a file, read the file
    fprintf(stderr, "Cache exists, reading the pre-calculated BVH data...\n");

    uLongf compressedSize = 0u;
    Bytef* compressedBuffer = nullptr;
    uLongf uncompressedSize = 0u;
    Bytef* uncompressedBuffer = nullptr;
    size_t destOffset = 0u;
    int c;

    if (1 != fread(&uncompressedSize, sizeof(uLongf), 1, fp)) goto CACHE_FAIL;
    if(uncompressedSize <= 0u) goto CACHE_FAIL;
    uncompressedBuffer = (Bytef*)malloc(uncompressedSize);
    if (uncompressedBuffer == nullptr) goto CACHE_FAIL;

    if (1 != fread(&compressedSize, sizeof(uLongf), 1, fp)) goto CACHE_FAIL;
    if (compressedSize <= 0u) goto CACHE_FAIL;
    compressedBuffer = (Bytef*)malloc(compressedSize);
    if (compressedBuffer == nullptr) goto CACHE_FAIL;

    if (compressedSize != fread(compressedBuffer, sizeof(Bytef), compressedSize, fp)) goto CACHE_FAIL;

    fprintf(stderr, "Uncompressing BVH data.\n");
    c = uncompress(uncompressedBuffer, &uncompressedSize, compressedBuffer, compressedSize);

    if (c != Z_OK)
    {
        fprintf(stderr, "BVH uncompression failure.\n");
        return false;
    }

    memcpy(&deviceMesh.nodesUsed, uncompressedBuffer,              sizeof(uint32_t));                       destOffset += sizeof(uint32_t);
    memcpy(&deviceMesh.numTris,   (void*)(uncompressedBuffer + destOffset), sizeof(uint32_t));                       destOffset += sizeof(uint32_t);

    bvhNode = new BVHNode[deviceMesh.nodesUsed];
    triIdx = new uint32_t[deviceMesh.numTris];
    memcpy(bvhNode,               (void*)(uncompressedBuffer + destOffset), sizeof(BVHNode) * deviceMesh.nodesUsed); destOffset += sizeof(BVHNode) * deviceMesh.nodesUsed;
    memcpy(triIdx,                (void*)(uncompressedBuffer + destOffset), sizeof(uint32_t) * deviceMesh.numTris);  destOffset += sizeof(uint32_t) * deviceMesh.numTris;

    triangleBuffer = new Triangle[deviceMesh.numTris];
    meshInfoBuffer = new MeshInfo;
    memcpy(triangleBuffer,        (void*)(uncompressedBuffer + destOffset), sizeof(Triangle) * deviceMesh.numTris);  destOffset += sizeof(Triangle) * deviceMesh.numTris;
    memcpy(meshInfoBuffer,        (void*)(uncompressedBuffer + destOffset), sizeof(MeshInfo));                       destOffset += sizeof(MeshInfo);
    /*
    if (1 != fread(&deviceMesh.nodesUsed, sizeof(uint32_t), 1, fp)) goto CACHE_FAIL;
    if (1 != fread(&deviceMesh.numTris,   sizeof(uint32_t), 1, fp)) goto CACHE_FAIL;

    bvhNode = new BVHNode[deviceMesh.nodesUsed];
    triIdx = new uint32_t[deviceMesh.numTris];
    triangleBuffer = new Triangle[deviceMesh.numTris];
    meshInfoBuffer = new MeshInfo[1u];
    //triangleCentroidScratchBuffer = new float3[deviceMesh.numTris];

    if (deviceMesh.nodesUsed != fread(bvhNode, sizeof(BVHNode), deviceMesh.nodesUsed, fp)) goto CACHE_FAIL;
    if (deviceMesh.numTris != fread(triIdx, sizeof(uint32_t), deviceMesh.numTris, fp)) goto CACHE_FAIL;
    if (deviceMesh.numTris != fread(triangleBuffer, sizeof(Triangle), deviceMesh.numTris, fp)) goto CACHE_FAIL;
    if (1 != fread(meshInfoBuffer, sizeof(MeshInfo), 1, fp)) goto CACHE_FAIL;
    //if (deviceMesh.numTris != fread(triangleCentroidScratchBuffer, sizeof(float3), deviceMesh.numTris, fp)) goto CACHE_FAIL;

    */

    fclose(fp);
    fprintf(stderr, "BVH cache read.\n");

    return true;

CACHE_FAIL:
    fclose(fp);
    fprintf(stderr, "ERROR: BVH cache is invalid.\n");

    return false;
}

bool GPU_Mesh::TrySaveCache(const std::string& filename)
{
    const Bytef* uncompressedBuffer = nullptr;
    uLongf uncompressedSize = 0u;
    uncompressedSize += sizeof(uint32_t);
    uncompressedSize += sizeof(uint32_t);
    uncompressedSize += sizeof(BVHNode) * deviceMesh.nodesUsed;
    uncompressedSize += sizeof(uint32_t) * deviceMesh.numTris;
    uncompressedSize += sizeof(Triangle) * deviceMesh.numTris;
    uncompressedSize += sizeof(MeshInfo);

    uncompressedBuffer = (Bytef*)malloc(uncompressedSize);

    if (uncompressedBuffer == nullptr)
    {
        return false;
    }

    size_t destOffset = 0u;
    memcpy((void*)(uncompressedBuffer + destOffset), &deviceMesh.nodesUsed, sizeof(uint32_t));                       destOffset += sizeof(uint32_t);
    memcpy((void*)(uncompressedBuffer + destOffset), &deviceMesh.numTris,   sizeof(uint32_t));                       destOffset += sizeof(uint32_t);
    memcpy((void*)(uncompressedBuffer + destOffset), bvhNode,               sizeof(BVHNode) * deviceMesh.nodesUsed); destOffset += sizeof(BVHNode) * deviceMesh.nodesUsed;
    memcpy((void*)(uncompressedBuffer + destOffset), triIdx,                sizeof(uint32_t) * deviceMesh.numTris);  destOffset += sizeof(uint32_t) * deviceMesh.numTris;
    memcpy((void*)(uncompressedBuffer + destOffset), triangleBuffer,        sizeof(Triangle) * deviceMesh.numTris);  destOffset += sizeof(Triangle) * deviceMesh.numTris;
    memcpy((void*)(uncompressedBuffer + destOffset), meshInfoBuffer,        sizeof(MeshInfo));                       destOffset += sizeof(MeshInfo);

//if (1 != fwrite(&deviceMesh.nodesUsed, sizeof(uint32_t), 1, fp)) goto CACHE_FAIL;
//if (1 != fwrite(&deviceMesh.numTris,   sizeof(uint32_t), 1, fp)) goto CACHE_FAIL;
//if (deviceMesh.nodesUsed != fwrite(bvhNode, sizeof(BVHNode), deviceMesh.nodesUsed, fp)) goto CACHE_FAIL;
//if (deviceMesh.numTris != fwrite(triIdx, sizeof(uint32_t), deviceMesh.numTris, fp)) goto CACHE_FAIL;
//if (deviceMesh.numTris != fwrite(triangleBuffer, sizeof(Triangle), deviceMesh.numTris, fp)) goto CACHE_FAIL;
//if (1 != fwrite(meshInfoBuffer, sizeof(MeshInfo), 1, fp)) goto CACHE_FAIL;

    uLongf compressedSize = (uncompressedSize + uncompressedSize / 500) + 12u; //Allocate slightly larger destination buffer.
    Bytef* compressedBuffer = (Bytef*)malloc(compressedSize);

    fprintf(stderr, "Compressing BVH data.\n");
    int c = compress(compressedBuffer, &compressedSize, uncompressedBuffer, uncompressedSize);
    if (c != Z_OK)
    {
        fprintf(stderr, "BVH compression failure.\n");
        return false;
    }

    //int c = uncompress(dest, &destAllocSize, dataBuffer, allocSize);

    std::string cacheFile = CacheFilePath(filename);
    FILE* fp = fopen(cacheFile.c_str(), "wb");
    {
        // Now store the results, if possible...
        fprintf(stderr, "Writing BVH data...\n");

        if (fp == NULL) return false;

        if (1 != fwrite(&uncompressedSize, sizeof(uLongf), 1, fp)) goto CACHE_FAIL;
        if (1 != fwrite(&compressedSize, sizeof(uLongf), 1, fp)) goto CACHE_FAIL;
        if (compressedSize != fwrite(compressedBuffer, sizeof(Bytef), compressedSize, fp)) goto CACHE_FAIL;


        //if (1 != fwrite(&deviceMesh.nodesUsed, sizeof(uint32_t), 1, fp)) goto CACHE_FAIL;
        //if (1 != fwrite(&deviceMesh.numTris,   sizeof(uint32_t), 1, fp)) goto CACHE_FAIL;
        //
        //if (deviceMesh.nodesUsed != fwrite(bvhNode, sizeof(BVHNode), deviceMesh.nodesUsed, fp)) goto CACHE_FAIL;
        //if (deviceMesh.numTris != fwrite(triIdx, sizeof(uint32_t), deviceMesh.numTris, fp)) goto CACHE_FAIL;
        //if (deviceMesh.numTris != fwrite(triangleBuffer, sizeof(Triangle), deviceMesh.numTris, fp)) goto CACHE_FAIL;
        //if (1 != fwrite(meshInfoBuffer, sizeof(MeshInfo), 1, fp)) goto CACHE_FAIL;
        //if (deviceMesh.numTris != fwrite(triangleCentroidScratchBuffer, sizeof(float3), deviceMesh.numTris, fp)) goto CACHE_FAIL;

        fclose(fp);
        fprintf(stderr, "BVH cache written.\n");

        return true;
    }

CACHE_FAIL:
    fclose(fp);
    fprintf(stderr, "ERROR: BVH cache file writing failed.\n");

    return false;
}

void GPU_Mesh::BuildBVH()
{
    if (loadedFromCache)
    {
        return;
    }

    if (deviceMesh.numTris == 0)
    {
        return;
    }

    bvhNode = new BVHNode[deviceMesh.numTris * 2 - 1];
    triIdx = new uint32_t[deviceMesh.numTris];
    triangleCentroidScratchBuffer = new float3[deviceMesh.numTris];

    for (uint32_t i = 0; i < deviceMesh.numTris; i++)
    {
        triangleCentroidScratchBuffer[i] = (triangleBuffer[i].pos0 + triangleBuffer[i].pos1 + triangleBuffer[i].pos2) * 0.3333333f;
        triIdx[i] = i;
    }

    // assign all triangles to root node
    BVHNode& root = bvhNode[rootNodeIdx];

    root.leftFirst = 0;
    root.triCount = deviceMesh.numTris;

    UpdateNodeBounds(rootNodeIdx);
    // subdivide recursively
    Subdivide(rootNodeIdx);

    //Resize
    BVHNode* newArr = new BVHNode[deviceMesh.nodesUsed];
    memcpy(newArr, bvhNode, deviceMesh.nodesUsed * sizeof(BVHNode));
    delete[] bvhNode;
    bvhNode = newArr;

    fprintf(stderr, "BVH built using: %i nodes\n", deviceMesh.nodesUsed);

    if (TrySaveCache(filepath))
    {
        return;
    }
}

void GPU_Mesh::ApplyTransform()
{
    CUDATransformMatrix.upload(&transformMatrix, 1);
    transformMatrixInverse = glm::inverse(transformMatrix);
    CUDATransformMatrixInverse.upload(&transformMatrixInverse, 1);
    transformMatrixInverseTranspose = glm::transpose(transformMatrixInverse);
    CUDATransformMatrixInverseTranspose.upload(&transformMatrixInverseTranspose, 1);
}

GPU_Mesh::GPU_Mesh()
{
    //deviceMesh = new MeshBuffer();
    CUDATransformMatrix.alloc_and_upload(&transformMatrix, 1);
    CUDATransformMatrixInverse.alloc_and_upload(&transformMatrixInverse, 1);
    CUDATransformMatrixInverseTranspose.alloc_and_upload(&transformMatrixInverseTranspose, 1);
}

GPU_Mesh::~GPU_Mesh()
{
    //delete[] bvhNode;
    //delete[] triangleBuffer;
    //delete[] meshInfoBuffer;
    //if (triIdx != nullptr) { delete[] triIdx; }
    //if (triangleCentroidScratchBuffer != nullptr) { delete[] triangleCentroidScratchBuffer; }
}
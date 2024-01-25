#include "ImageLoader.h"
#include <fstream>
#include <stdio.h>

void* ImageLoader::LoadImageFile(const std::string path, uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;

    delete[] m_imageData;

    m_imageData = new float3[width * height];

    FILE* fptr;
    fopen_s(&fptr, path.c_str(), "r");

    if (fptr == nullptr)
    {
        return nullptr;
    }

    const std::size_t n = std::fread(m_imageData, sizeof(float), width * height * 3, fptr);

    cudaMalloc(&m_devPtr, width * height * sizeof(float3));
    cudaMemcpy(m_devPtr, m_imageData, width * height * sizeof(float3), cudaMemcpyHostToDevice);

    cudaError_t cudaStatus = cudaErrorStartupFailure;
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy ImageLoader failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    /*
    size_t d_pitchBytes;
    cudaMallocPitch((void**)&m_devPtr, &d_pitchBytes, width * sizeof(float), height);

    size_t h_pitchBytes = width * sizeof(float);
    cudaMemcpy2D(m_devPtr, d_pitchBytes, m_imageData, h_pitchBytes, width * sizeof(float), height, cudaMemcpyHostToDevice);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = m_devPtr;
    texRes.res.pitch2D.desc = channelDesc;
    texRes.res.pitch2D.width = width;
    texRes.res.pitch2D.height = height;
    texRes.res.pitch2D.pitchInBytes = h_pitchBytes;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.borderColor[0] = 0.0f;
    texDescr.borderColor[1] = 0.0f;
    texDescr.disableTrilinearOptimization = 1;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.maxAnisotropy = 1;
    texDescr.minMipmapLevelClamp = 1.0f;
    texDescr.maxMipmapLevelClamp = 1.0f;
    texDescr.mipmapFilterMode = cudaFilterModePoint;
    texDescr.mipmapLevelBias = 0.0f;
    texDescr.normalizedCoords = 0;
    texDescr.readMode = cudaReadModeElementType;
    texDescr.sRGB = 0;

    cudaCreateTextureObject(pTexObject, &texRes, &texDescr, NULL);
    
    return pTexObject;
    */

    return m_devPtr;
}

ImageLoader::~ImageLoader()
{
    delete[] m_imageData;
    cudaFree(pTexObject);
}
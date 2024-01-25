#pragma once

#include "cuda_runtime.h"
#include "cutil_math.cuh"
#include <string>
#include <cuda.h>

class ImageLoader
{

public:

    ~ImageLoader();

    void* LoadImageFile(const std::string path, uint32_t width, uint32_t height);

private:

    float3* m_imageData = nullptr;
    void* m_devPtr = nullptr;
    uint32_t m_width;
    uint32_t m_height;

    cudaTextureObject_t* pTexObject;
};
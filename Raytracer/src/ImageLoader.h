#pragma once

//#include "cuda_runtime.h"
//#include "cutil_math.cuh"

#include <string>
//#include <cuda.h>

#include "CudaBuffer.h"

struct GPUImage
{
    CUdeviceptr imageData_GPU;
    size_t width = 0u;
    size_t height = 0u;
};

class ImageLoader
{

public:

    ~ImageLoader();

    void LoadImageFile(const std::string path);
    bool SaveImageFile(const float* rgb, int width, int height, const char* outfilename);
    //void* LoadImageFile(const std::string path, uint32_t width, uint32_t height);

    GPUImage gpuImage;
    CUDABuffer gpuBuffer;
};
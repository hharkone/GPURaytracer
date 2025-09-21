#pragma once

#include <string>
#include "CudaBuffer.h"

struct GPUImage
{
    CUdeviceptr imageData_GPU = 0;
    size_t width = 0u;
    size_t height = 0u;
};

class ImageLoader
{

public:

    ~ImageLoader();

    void LoadImage_EXR(const std::string path);
    void LoadImage_PNG(const std::string path);
    bool SaveImage_EXR(const float* rgb, int width, int height, const char* outfilename);

    GPUImage gpuImage;
    CUDABuffer gpuBuffer;
};
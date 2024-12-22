#include "ImageLoader.h"
#include <fstream>
#include <stdio.h>

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_THREAD 1
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image.h"
#include "stb_image_write.h"
#include "zlib.h"
#include "tinyexr.h"

void ImageLoader::LoadImageFile(const std::string path)
{
    float* out; // width * height * RGBA
    int width;
    int height;
    const char* err = NULL; // or nullptr in C++11

    int ret = LoadEXR(&out, &width, &height, path.c_str(), &err);

    if (ret != TINYEXR_SUCCESS)
    {
        if (err)
        {
            fprintf(stderr, "ImageLoader: %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }

    }
    else
    {
        gpuImage.height = (size_t)height;
        gpuImage.width = (size_t)width;

        if (gpuBuffer.sizeInBytes != 0u)
        {
            gpuBuffer.free();
        }

        gpuBuffer.alloc_and_upload(out, gpuImage.width * gpuImage.height * 4u);
        gpuImage.imageData_GPU = gpuBuffer.d_pointer();

        free(out); // release memory of image data
    }
}
/*
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
        fprintf(stderr, "LoadImageFile: Invalid file path.");
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

    return m_devPtr;
}
*/

ImageLoader::~ImageLoader()
{
    //delete[] m_imageData;
    //cudaFree(pTexObject);
    gpuBuffer.free();
}
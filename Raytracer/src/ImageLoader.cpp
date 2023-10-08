#include "ImageLoader.h"
#include <fstream>
#include <stdio.h>
#include "cutil_math.cuh"


float* ImageLoader::LoadImageFile(const std::string path, uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;

    int swap = 1;
    m_imageData = new float[width * height * 3];

    FILE* fptr;
    fopen_s(&fptr, path.c_str(), "r");
    const std::size_t n = std::fread(m_imageData, sizeof(float), width * height * 3, fptr);

    return m_imageData;
}

#pragma once

#include <string>

class ImageLoader
{

public:
    float* LoadImageFile(const std::string path, uint32_t width, uint32_t height);

private:

    float* m_imageData = nullptr;
    uint32_t m_width;
    uint32_t m_height;

};
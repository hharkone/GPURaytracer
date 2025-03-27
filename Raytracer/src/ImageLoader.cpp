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
#include "lodepng.h"

void ImageLoader::LoadImage_PNG(const std::string path)
{
    std::vector<unsigned char> png;
    std::vector<unsigned char> image; //the raw pixels
    unsigned width, height;

    //load and decode
    unsigned error = lodepng::load_file(png, path);
    if (!error) error = lodepng::decode(image, width, height, png);

    //if there's an error, display it
    if (error)
    {
        fprintf(stderr, "ImageLoader: PNG decoder error: %i ERROR: %s\n", error, lodepng_error_text(error));
    }
    else
    {
        gpuImage.height = (size_t)height;
        gpuImage.width = (size_t)width;

        if (gpuBuffer.sizeInBytes != 0u)
        {
            gpuBuffer.free();
        }

        //gpuBuffer.alloc_and_upload(&image.at(0u), image.size());
        //gpuBuffer.alloc_and_upload(out, image.size());
        gpuBuffer.alloc_and_upload(&image.at(0u), image.size());
        gpuImage.imageData_GPU = gpuBuffer.d_pointer();

        //free(out);
        fprintf(stderr, "ImageLoader: Loaded PNG file: %s\n", path.c_str());
    }
}

void ImageLoader::LoadImage_EXR(const std::string path)
{
    float* out; // width * height * RGBA
    int width;
    int height;
    const char* err = nullptr;

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
        fprintf(stderr, "ImageLoader: Loaded EXR file: %s\n", path.c_str());
    }
}

bool ImageLoader::SaveImage_EXR(const float* rgb, int width, int height, const char* outfilename)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 4;

    std::vector<float> images[4];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);
    images[3].resize(width * height);

    // Split RGBRGBRGB... into R, G and B layer
    for (int i = 0; i < width * height; i++)
    {
        images[0][i] = rgb[4 * i + 0];
        images[1][i] = rgb[4 * i + 1];
        images[2][i] = rgb[4 * i + 2];
        images[3][i] = rgb[4 * i + 3];
    }

    float* image_ptr[4];
    image_ptr[0] = &(images[3].at(0)); // A
    image_ptr[1] = &(images[2].at(0)); // B
    image_ptr[2] = &(images[1].at(0)); // G
    image_ptr[3] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 4;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy_s(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
    strncpy_s(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
    strncpy_s(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
    strncpy_s(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

    header.compression_type = TINYEXR_COMPRESSIONTYPE_RLE;
    header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++)
    {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
    if (ret != TINYEXR_SUCCESS)
    {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return ret;
    }

    printf("Saved exr file. [ %s ] \n", outfilename);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    return true;
}

ImageLoader::~ImageLoader()
{
    gpuBuffer.free();
}
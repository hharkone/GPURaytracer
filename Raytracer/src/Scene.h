#pragma once

#include <string>
#include <cuda_runtime.h>
#include "CudaBuffer.h"
#include "GPU_Mesh.h"

struct Material
{
    float3 albedo{ 0.8f, 0.8f, 0.8f };
    float vcolor{ 0.0f };
    float  roughness{ 0.6f };
    float3 emission{ 0.0f, 0.0f, 0.0f };
    float emissionIntensity = 0.0f;
    float ior = 1.5f;
    float transmission = 0.0f;
    float transmissionInscatter = 0.0f;
    float transmissionInscatterAnisotropy = 0.0f;
    float transmissionRoughness = 0.0f;
    float transmissionAberration = 0.0f;
    float transmissionDensity = 0.0f;
    float3 transmissionColor{ 1.0f, 1.0f, 1.0f };
    float  metalness = 0.0f;
};

struct Sphere
{
    float rad;              // Radius
    float3 pos;             // Position
    uint16_t materialIndex; // Material Index
};

struct Box
{
    float3 size;            // Size
    float3 pos;             // Position
    uint16_t materialIndex; // Material Index
};

struct TonemapSettings
{
    //Tonemapper
    float A = 0.389f;
    float B = 0.24f;
    float C = 0.18f;
    float D = 0.137f;
    float E = 0.03f;
    float F = 0.18f;
    float W = 1.8f;
    float Exposure = 1.3f;
};

enum class EnvironmentType
{
    EnvType_Solid = 0,
    EnvType_ProceduralSky = 1,
    EnvType_HDRI = 2
};

struct Scene
{
    Scene();
    ~Scene();

    void UploadMaterials();
    void AddNewDefaultMaterial(size_t index);
    void ImportMesh(std::string path);
    void ImportMesh(std::string path, size_t overrideMaterial);

    TonemapSettings tonemap;
    EnvironmentType envType = EnvironmentType::EnvType_HDRI;
    bool envImgPathChanged = false;
    std::string envImgPath = "";

    //Procedural Sky
    float3 skyColor = make_float3(1.0f, 1.0f, 1.0f);
    float skyBrightness = 1.0f;
    float3 skyColorHorizon = make_float3(0.55f, 0.66f, 0.9f);
    float3 skyColorZenith  = make_float3(0.28f, 0.28f, 0.75f);
    float3 groundColor     = make_float3(0.47f, 0.519f, 0.682f);

    float3 sunDirection = make_float3(1.0f, 0.42f, 0.58f);
    float sunFocus = 34.0f;
    float sunIntensity = 22.0f;
    float skyRotation = 0.0f;
    float backgroundBrightness = 1.0f;

    //Material materialAir = Material{ { 1.0f, 1.0f,  1.0f  }, 0.0f, 0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f };

    Material* materials = new Material[4]
    {
        //         Albedo,              vcolor amount, roughness, emission,            emission intensity, ior, trans, inscatter, inscatter anisotropy, trans rough, trans aber, trans dens, trans col,              metal
        Material{ { 1.0f, 1.0f,  1.0f  }, 0.0f, 0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, { 0.0f, 0.0f, 0.0f }, 0.0f },                                                                    //Air
        Material{ { 0.8f, 0.8f,  0.8f  }, 0.0f,        0.21f,     { 0.0f, 0.0f, 0.0f }, 0.0f,              1.5f, 0.0f, 0.0f,      0.0f,                 0.0f,        1.0f,       1.0f,       { 0.95f, 0.75f, 0.4f }, 0.0f }, //Diffuse
        Material{ { 0.0f, 0.0f,  0.0f  }, 0.0f,        0.1f,      { 1.0f, 0.8f, 0.6f }, 7.0f,              1.5f, 0.0f, 0.0f,      0.0f,                 0.0f,        0.0f,       0.1f,       { 1.0f, 1.0f, 1.0f },   0.0f }, //Light1
        Material{ { 0.0f, 0.0f,  0.0f  }, 0.0f,        0.1f,      { 0.6f, 0.8f, 1.0f }, 4.5f,              1.5f, 0.0f, 0.0f,      0.0f,                 0.0f,        0.0f,       0.1f,       { 1.0f, 1.0f, 1.0f },   0.0f }  //Light2
    };

    Sphere spheresSimple[2] =
    {
        //{ float radius, { float3 position }, { Material }}
          Sphere{ 1.0f,  { 10.0f, 1.9f, -1.77f }, 3u},
          //Sphere{ 0.25f, {  0.0f,  0.0f,  0.0f  }, 7u},
          //Sphere{ 19.0f, {  0.0f, -19.0f, 0.0f  }, 1u},
          Sphere{ 1.3f,  { -10.0f, 2.8f,  1.56f }, 2u}
    };

    Box boxSimple[2] =
    {
          //Box{ { 2.0f,  2.0f,   2.0f },  {  0.0f,  1.0f,  0.0f  }, 2u},
          Box{ { 10.0f, 10.0f, 10.0f },  {  0.0f, -5.2f, 0.0f   }, 1u},
          Box{ { 1.0f,  1.0f,   1.0f },  { -3.9f,  1.8f, -0.56f }, 1u}
    };

    size_t materialCount = 4u;
    size_t sphereCount = 2u;
    size_t boxCount = 1u;

    CUDABuffer materialBuffer;
    CUdeviceptr materialBufferPtr;
    GPU_Mesh sceneMesh;

private:
    void ResizeAndAddMaterials(size_t size);
};
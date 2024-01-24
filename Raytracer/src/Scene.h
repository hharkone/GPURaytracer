#pragma once

#include "cuda_runtime.h"

struct Material
{
    float3 albedo{ 0.8f, 0.8f, 0.8f };
    float  roughness{ 0.6f };
    float3 emission{ 0.0f, 0.0f, 0.0f };
    float emissionIntensity = 0.0f;
    float ior = 1.5f;
    float transmission = 0.0f;
    float transmissionRoughness = 0.0f;
    float transmissionDensity = 0.0f;
    float3 transmissionColor{ 1.0f, 1.0f, 1.0f };
    float  metalness = 0.0f;
};

struct Sphere
{
    float rad;            // Radius
    float3 pos;           // Position
    uint16_t materialIndex; // Material Index
};

enum class EnvironmentType
{
    EnvType_Solid = 0,
    EnvType_ProceduralSky = 1
};

struct Scene
{
    EnvironmentType envType = EnvironmentType::EnvType_ProceduralSky;

    //Procedural Sky
    float3 skyColor = make_float3(1.0f, 1.0f, 1.0f);
    float skyBrightness = 0.0f;
    float3 skyColorHorizon = make_float3(0.55f, 0.66f, 0.9f);
    float3 skyColorZenith  = make_float3(0.28f, 0.28f, 0.75f);
    float3 groundColor     = make_float3(0.47f, 0.519f, 0.682f);

    float3 sunDirection = make_float3(1.0f, 0.42f, 0.58f);
    float sunFocus = 34.0f;
    float sunIntensity = 22.0f;

    //Tonemapper
    float A = 0.4f;
    float B = 0.24f;
    float C = 0.13f;
    float D = 0.1f;
    float E = 0.03f;
    float F = 0.30f;
    float W = 2.0f;
    float Exposure = 1.0f;

    Material materials[8] =
    {
        Material{ { 0.5f, 0.5f,  0.5f  }, 0.21f, { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.5f, 1.0f, 1.0f, { 0.95f, 0.75f, 0.4f }, 0.0f }, //Glass
        Material{ { 0.7f, 0.1f,  0.1f  }, 0.05f, { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 0.0f }, //Red	
        Material{ { 0.5f, 0.7f,  0.8f  }, 0.1f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 0.0f }, //Blue
        Material{ { 0.8f, 0.8f,  0.8f  }, 0.5f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 0.0f }, //White
        Material{ { 1.0f, 0.9f,  0.6f  }, 0.1f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 1.0f }, //Gold
        Material{ { 0.98f,0.815f,0.75f }, 0.1f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 1.0f }, //Copper
        Material{ { 0.0f, 0.0f,  0.0f  }, 0.1f,  { 1.0f, 0.8f, 0.6f }, 7.0f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 0.0f }, //Light1
        Material{ { 0.0f, 0.0f,  0.0f  }, 0.1f,  { 0.6f, 0.8f, 1.0f }, 4.5f, 1.5f, 0.0f, 0.0f, 0.1f, { 1.0f, 1.0f, 1.0f }, 0.0f }  //Light2
    };

    Sphere spheresSimple[3] =
    {
        //{ float radius, { float3 position }, { Material }}
          Sphere{ 1.0f,  { 1.58f, 1.9f, -1.77f }, 7u},
          Sphere{ 19.0f, {  0.0f, -19.0f, 0.0f }, 1u},
          Sphere{ 1.3f,  { -3.9f, 1.8f, -0.56f }, 6u}
    };

    size_t materialCount = 8u;
    size_t sphereCount = 3u;
};
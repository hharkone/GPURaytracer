#pragma once

#include "cuda_runtime.h"

struct Material
{
    float3 albedo{ 0.8f, 0.8f, 0.8f };
    float  roughness{ 0.6f };
    float3 emission{ 0.0f, 0.0f, 0.0f };
    float emissionIntensity = 0.0f;
    float  metalness = 0.0f;
};

struct Scene
{
    float3 skyColor = make_float3(1.0f, 1.0f, 1.0f);
    float skyBrightness = 1.2f;
    float3 skyColorHorizon = make_float3(0.55f, 0.66f, 0.9f);
    float3 skyColorZenith  = make_float3(0.28f, 0.28f, 0.75f);
    float3 groundColor     = make_float3(0.47f, 0.519, 0.682f);
    float sunFocus = 50.0f;
    float sunIntensity = 10.0f;

    Material materials[8] =
    {
        Material{ { 0.7f, 0.7f,  0.7f  }, 0.05f, { 0.0f, 0.0f, 0.0f }, 0.0f, 0.0f }, //White
        Material{ { 0.7f, 0.1f,  0.1f  }, 0.05f, { 0.0f, 0.0f, 0.0f }, 0.0f, 0.0f }, //Red	
        Material{ { 0.5f, 0.7f,  0.8f  }, 0.1f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 0.0f }, //Blue
        Material{ { 1.0f, 1.0f,  1.0f  }, 0.0f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.0f }, //Mirror
        Material{ { 1.0f, 0.9f,  0.6f  }, 0.1f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.0f }, //Gold
        Material{ { 0.98f,0.815f,0.75f }, 0.1f,  { 0.0f, 0.0f, 0.0f }, 0.0f, 1.0f }, //Copper
        Material{ { 0.0f, 0.0f,  0.0f  }, 0.1f,  { 8.0f, 6.0f, 5.0f }, 1.0f, 0.0f }, //Light1
        Material{ { 0.0f, 0.0f,  0.0f  }, 0.1f,  { 5.0f, 6.0f, 8.0f }, 1.0f, 0.0f }	 //Light2
    };

    size_t materialCount = 8u;

};
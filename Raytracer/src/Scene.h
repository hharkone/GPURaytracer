#pragma once

#include "cuda_runtime.h"

struct Scene
{
    float3 m_skyColor = make_float3(1.0f, 1.0f, 1.0f);
    float m_skyBrightness = 1.2f;
    float3 m_skyColorHorizon = make_float3(0.55f, 0.66f, 0.9f);
    float3 m_skyColorZenith  = make_float3(0.28f, 0.28f, 0.75f);
    float3 m_groundColor     = make_float3(0.08f, 0.08f, 0.08f);
    float m_sunFocus = 50.0f;
    float m_sunIntensity = 10.0f;
};
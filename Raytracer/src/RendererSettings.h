#pragma once

struct RenderSettings
{
    bool accumulate = true;
    int bounces = 15;
    int debug = 0;
    bool denoise = true;
    bool bvhDebug = false;
};
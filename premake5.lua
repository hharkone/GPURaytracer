-- premake5.lua
workspace "Raytracer"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "Raytracer"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "Raytracer"
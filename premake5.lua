-- premake5.lua
workspace "Raytracer"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "Raytracer"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"
include "RaytracerExternal.lua"
include "Raytracer"

project "Dependencies/Premake"
kind "Utility"

files
{
"**.lua"
}

buildmessage 'Re-generating Project Files!'
postbuildcommands
{
'%{prj.location}/vendor/premake/premake5 vs2020'
}
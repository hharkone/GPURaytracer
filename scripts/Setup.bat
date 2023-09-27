@echo off

pushd ..
Walnut\vendor\bin\premake5.exe vs2022
popd
echo %VULKAN_SDK%
echo %CUDA_PATH%
pause
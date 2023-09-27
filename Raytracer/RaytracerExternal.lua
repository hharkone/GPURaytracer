-- RaytracerExternal.lua

CUDA_SDK = os.getenv("CUDA_PATH")

IncludeDir = {}
IncludeDir["CUDA"] = "%{CUDA_SDK}/include"

LibraryDir = {}
LibraryDir["CUDA"] = "%{CUDA_SDK}/lib/x64"

Library = {}
Library["CudaLib"] = "cuda.lib"
Library["CudartStaticLib"] = "cudart_static.lib"

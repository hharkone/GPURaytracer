-- RaytracerExternal.lua

CUDA_SDK = os.getenv("CUDA_PATH")

RT_IncludeDir = {}
RT_IncludeDir["CUDA"] = "%{CUDA_SDK}/include"

RT_LibraryDir = {}
RT_LibraryDir["CUDA"] = "%{CUDA_SDK}/lib/x64"

RT_Library = {}
RT_Library["CudaLib"] = "cuda.lib"
RT_Library["CudartStaticLib"] = "cudart_static.lib"

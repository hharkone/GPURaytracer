#include <cmath>

#include "CudaMain.cuh"
#include "cutil_math.cuh"

#define M_PI 3.14159265359f  // pi

int checkCudaError(cudaError_t& error)
{
	if (error == cudaSuccess)
	{
		return 0;
	}

	return 1;
}

__global__ void addKernel(float* a, float* b, float* c)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ inline float3 srgbToLinear(float3 c)
{
	return powf(c, 2.2222f);
}

__device__ inline uint32_t ConvertToRGBA(const float3& color)
{
	float3 outColor;
	outColor.x = clamp(color.x, 0.0f, 1.0f);
	outColor.y = clamp(color.y, 0.0f, 1.0f);
	outColor.z = clamp(color.z, 0.0f, 1.0f);

	outColor = powf(outColor, 0.4646464);

	uint8_t r = (uint8_t)(outColor.x * 255.0f);
	uint8_t g = (uint8_t)(outColor.y * 255.0f);
	uint8_t b = (uint8_t)(outColor.z * 255.0f);

	uint32_t returnValue = (255 << 24) | (b << 16) | (g << 8) | r;

	return returnValue;
}

struct Ray
{
	float3 origin; // ray origin
	float3 direction;  // ray direction
	__device__ Ray(float3 o_, float3 d_) : origin(o_), direction(d_) {}
};

struct Material
{
	float3 albedo    { 0.8f, 0.8f, 0.8f };
	float  roughness { 0.6f };
	float3 emission  { 0.0f, 0.0f, 0.0f };
	float  metalness = 0.0f;
};

struct HitInfo
{
	bool didHit = false;
	float dst = FLT_MAX;
	float3 hitPoint {0.0f, 0.0f, 0.0f};
	float3 normal{ 0.0f, 0.0f, 0.0f };
	Material material;
};

struct Sphere
{
	float rad;            // Radius
	float3 pos;           // Position
	Material mat;         // Material

#if true
	__device__ float intersect_sphere(const Ray& r) const
	{
		// Ray/sphere intersection returns distance t to intersection point, 0 if no hit ray
		// equation: p(x,y,z) = ray.orig + t*ray.dir general sphere equation: x^2 + y^2 + z^2 = rad^2
		// classic quadratic equation of form ax^2 + bx + c = 0 solution x = (-b +- sqrt(b*b - 4ac))
		// / 2a solve t^2*ray.dir*ray.dir + 2*t*(orig-p)*ray.dir + (orig-p)*(orig-p) - rad*rad = 0
		// more details in "Realistic Ray Tracing" book by P. Shirley or Scratchapixel.com

		float3 op = pos - r.origin;                                                 // Distance from ray.orig to center sphere
		float t, epsilon = 0.0001f;                                                 // Epsilon required to prevent floating point precision artefacts
		float b = dot(op, r.direction);                                             // b in quadratic equation
		float disc = b * b - dot(op, op) + rad * rad;                               // discriminant quadratic equation
		if (disc < 0) return 0;                                                     // if disc < 0, no real solution (we're not interested in complex roots)
		else disc = sqrtf(disc);                                                    // if disc >= 0, check for solutions using negative and positive discriminant
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);   // pick closest point in front of ray origin
	}
#else

	__device__ inline HitInfo intersect_sphere(const Ray& r) const
	{
		HitInfo hit; 

		float3 offsetRayOrigin = r.origin - pos;
		// From the equation: sqrLength(rayOrigin + rayDir * dst) = radius^2
		// Solving for dst results in a quadratic equation with coefficients:
		float a = dot(r.direction, r.direction); // a = 1 (assuming unit vector)
		float b = 2.0f * dot(offsetRayOrigin, r.direction);
		float c = dot(offsetRayOrigin, offsetRayOrigin) - rad * rad;
		// Quadratic discriminant
		float discriminant = b * b - 4.0f * a * c;

		// No solution when d < 0 (ray misses sphere)
		if (discriminant >= 0.0f)
		{
			// Distance to nearest intersection point (from quadratic formula)
			float dst = (-b - sqrtf(discriminant)) / (2.0f * a);

			// Ignore intersections that occur behind the ray
			if (dst >= 0)
			{
				hit.didHit = true;
				hit.dst = dst;
				hit.hitPoint = r.origin + r.direction * dst;
				hit.normal = normalize(hit.hitPoint - pos);
				hit.material = mat;
			}
		}
	}
#endif
};

struct Camera_GPU
{
	float fov;
	float invViewMat[16];
	float invProjMat[16];
	float viewMat[16];
};

void CudaRenderer::Clear()
{
	cudaDeviceSynchronize();
	memset(m_outputBuffer, 0, m_bufferSize);
	memset(m_imageData, 0, m_width * m_height * sizeof(uint32_t));
	cudaMemset(m_accumulationBuffer_GPU, 0, m_bufferSize);
	cudaMemset(m_imageData_GPU, 0, m_width * m_height * sizeof(uint32_t));
}
/*
__constant__ static float jitterMatrix[10] =
{
   -0.25,  0.75,
	0.75,  0.33333,
   -0.75, -0.25,
	0.25, -0.75,
	0.0f, 0.0f
};
*/
// SCENE 9 spheres forming a Cornell box small enough to be in constant GPU memory 
__constant__ Sphere spheres[] =
{
	  //{ float radius, { float3 position },      { float3 emission }, { float3 colour },       refl_type }
	  { 1e5f,{ 1e5f + 1.0f, 40.8f, 81.6f },     Material{ { 0.5f, 0.7f,  0.8f  }, 0.1f, { 0.0f, 0.0f, 0.0f }, 0.0f } }, //Left
	  { 1e5f,{ -1e5f + 99.0f, 40.8f, 81.6f },   Material{ { 0.7f, 0.1f,  0.1f  }, 0.1f, { 0.0f, 0.0f, 0.0f }, 0.0f } }, //Right
	  { 1e5f,{ 50.0f, 40.8f, 1e5f },            Material{ { 1.0f, 1.0f,  1.0f  }, 0.0f, { 0.0f, 0.0f, 0.0f }, 1.0f } }, //Back
	  { 1e5f,{ 50.0f, 40.8f, -1e5f + 600.0f },  Material{} }, //Frnt     	   
	  { 1e5f,{ 50.0f, 1e5f, 81.6f },            Material{ { 0.7f, 0.7f,  0.7f  }, 0.05f,{ 0.0f, 0.0f, 0.0f }, 0.0f } }, //Botm
	  { 1e5f,{ 50.0f, -1e5f + 81.6f, 81.6f },   Material{} }, //Top			   
	  { 16.5f,{ 27.0f, 16.5f, 47.0f },          Material{ { 0.7f, 0.7f,  0.7f  }, 0.05f,{ 0.0f, 0.0f, 0.0f }, 0.0f } }, // small sphere 1
	  { 16.5f,{ 73.0f, 16.5f, 78.0f },          Material{ { 1.0f, 0.9f,  0.6f  }, 0.05f, { 0.0f, 0.0f, 0.0f }, 1.0f } },  // gold sphere 2
	  { 16.5f,{ 73.0f, 16.5f, 118.0f },         Material{ { 0.98f,0.815f,0.75f }, 0.05f, { 0.0f, 0.0f, 0.0f }, 1.0f } }, // copper sphere 2
	  { 100.0f,{ 30.0f, 181.6f - 1.9f, 80.0f }, Material{ { 0.0f, 0.0f,  0.0f  }, 0.1f, { 8.0f, 6.0f, 5.0f }, 0.0f } },  // Light
	  { 100.0f,{ 70.0f, 181.6f - 1.9f, 80.0f }, Material{ { 0.0f, 0.0f,  0.0f  }, 0.1f, { 5.0f, 6.0f, 8.0f }, 0.0f } }   // Light
	  //{ 2.1f,{ 40.0f, 40.5f, 47.0f }, Material{ { 0.8f, 0.8f, 0.8f }, 0.1f, { 150.0f, 160.0f, 180.0f }, 0.0f} }      // Light
};

__constant__ Sphere spheresSimple[] =
{
	//{ float radius, { float3 position }, { Material }}
	  { 0.5f, { 0.0f, 0.0f, 0.0f },  Material{} },
	  { 0.5f, { 0.0f, -1.0f, 0.0f }, Material{ {0.0f, 0.0f, 0.0f}, 0.0f, {1.0f, 1.0f, 0.0f}, 0.0f }}
};

__device__ static float fresnel_schlick_ratio(float cos_theta_incident, float power)
{
	float p = 1.0f - cos_theta_incident;
	return pow(p, power);
}
__constant__ static float jitterMatrix[10] =
{
   -0.25,  0.75,
	0.75,  0.33333,
   -0.75, -0.25,
	0.25, -0.75,
	0.0f, 0.0f
};

// Random number generator from https://github.com/gz/rust-raytracer
__device__ static float getrandom(unsigned int* seed0, unsigned int* seed1)
{
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	// Convert to float
	union
	{
		float f;
		unsigned int ui;
	} res;

	// Bitwise AND, bitwise OR
	res.ui = (ires & 0x007fffff) | 0x40000000;

	return (res.f - 2.f) / 2.f;
}

// PCG (permuted congruential generator). Thanks to:
// www.pcg-random.org and www.shadertoy.com/view/XlGcRh
__device__ uint32_t nextRandom(uint32_t& state)
{
	state = state * 747796405 + 2891336453;
	uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
	result = (result >> 22) ^ result;
	return result;
}

__device__ float randomValue(uint32_t& state)
{
	return nextRandom(state) / 4294967295.0; // 2^32 - 1
}

__device__ float3 inUnitSphere(uint32_t& state)
{
	return normalize(make_float3(randomValue(state) * 2.0f - 1.0f, randomValue(state) * 2.0f - 1.0f, randomValue(state) * 2.0f - 1.0f));
}

__device__ float randomValueNormalDistribution(uint32_t& state)
{
	// Thanks to https://stackoverflow.com/a/6178290
	float theta = 2 * 3.1415926 * randomValue(state);
	float rho = fsqrtf(-2 * log(randomValue(state)));

	return rho * cos(theta);
}

__device__ float3 randomDirection(uint32_t& state)
{
	// Thanks to https://math.stackexchange.com/a/1585996
	float x = randomValueNormalDistribution(state);
	float y = randomValueNormalDistribution(state);
	float z = randomValueNormalDistribution(state);

	return normalize(make_float3(x, y, z));
}

__device__ void vector4_matrix4_mult(float* vec, float* mat, float* out)
{
	for (int i = 0; i < 4; i++)
	{
		out[i] = 0.0f;
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			out[i] += (mat[i + 4 * j] * vec[j]);
		}
	}
}

__device__ inline bool intersect_scene(const Ray& r, float& t, int& id)
{
	float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;  // t is distance to closest intersection, initialise t to a huge number outside scene
	for (int i = int(n); i--;)                                      // Test all scene objects for intersection
	{
		if ((d = spheres[i].intersect_sphere(r)) && d < t) // If newly computed intersection distance d is smaller than current closest intersection distance
		{
			t = d;  // Keep track of distance along ray to closest intersection point
			id = i; // and closest intersected object
		}
	}
	// Returns true if an intersection with the scene occurred, false when no hit
	return t < inf;
}

__device__ float3 radiance(Ray& r, uint32_t& s1, size_t bounces) // Returns ray color
{
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	for (size_t b = 0; b < bounces; b++)
	{
		float t;           // Distance to closest intersection
		int id = 0;        // Index of closest intersected sphere

		// Test ray for intersection with scene
		if (!intersect_scene(r, t, id))
		{
			accucolor += mask * make_float3(0.1f, 0.12f, 0.2f); // If miss, return sky
			break;
		}
		const Sphere& obj = spheres[id];
		float3 x = r.origin + r.direction * t;                   // hitpoint
		float3 n = normalize(x - obj.pos);             // normal
		float3 nl = dot(n, r.direction) < 0.0f ? n : n * -1.0f;    // front facing normal

		accucolor += mask * obj.mat.emission;

		// Create 2 random numbers
		float r1 = 2 * M_PI * randomValue(s1); // Pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
		float r2 = randomValue(s1);            // Pick random number for elevation
		float r2s = sqrtf(r2);

		float ndotl = fmaxf(dot(-r.direction, nl), 0.0f);
		float f = fresnel_schlick_ratio(ndotl, 8.0f);

		bool isSpecularBounce = max(obj.mat.metalness, max(f, 0.02f)) >= randomValue(s1);

		float3 diffuseDir = normalize(nl + randomDirection(s1));
		float3 specularDir = reflect(r.direction, normalize(nl + randomDirection(s1) * obj.mat.roughness));

		float3 linearSurfColor = srgbToLinear(obj.mat.albedo);

		r.direction = normalize(lerp(diffuseDir, specularDir, isSpecularBounce));

		// New ray origin is intersection point of previous ray with scene
		r.origin = x + nl * 0.1f; // offset ray origin slightly to prevent self intersection

		mask = mask * lerp(linearSurfColor, lerp(make_float3(1.0f), linearSurfColor, obj.mat.metalness), isSpecularBounce);

		float p = max(mask.x, max(mask.y, mask.z));
		if (randomValue(s1) >= p)
		{
			break;
		}
		mask *= 1.0f / p;

		//accucolor = { f, f, f };
	}


	return accucolor;
}

__global__ void render_kernel(float3* buf, uint32_t width, uint32_t height, Camera_GPU camera, size_t samples, size_t bounces, uint32_t sampleIndex)
{
	// Assign a CUDA thread to every pixel (x,y) blockIdx, blockDim and threadIdx are CUDA specific
	// Keywords replaces nested outer loops in CPU code looping over image rows and image columns
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height)) return;

	// Index of current pixel (calculated using thread index)
	uint32_t i = (height - y - 1) * width + x;
	
	// Seeds for random number generator
	uint32_t s1 = x * y * sampleIndex + i;
	//uint32_t s2 = y * sampleIndex + i;
	//const float fov = camera.fov * M_PI / 180.0f;
	//const float tf = std::tan(fov * 0.5f);

	float2 coord = { (float)x / (float)width, (float)y / (float)height };
	coord = coord * 2.0f - make_float2(1.0f, 1.0f); // -1 -> 1
	float viewCoord[4] = { coord.x, coord.y, -1.0f, 1.0f };
	float target[4];
	float target2[4];



	//glm::vec4 target = m_InverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
	//glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
	//m_RayDirections[x + y * m_ViewportWidth] = rayDirection;

	vector4_matrix4_mult(&viewCoord[0], &camera.invProjMat[0], &target[0]);

	//float4 target = m_InverseProjection * ;
	float4 projDir4 = make_float4(normalize(make_float3(target[0], target[1], target[2]) / target[3]), 0.0f);

	float projDir[4] = { projDir4.x, projDir4.y, projDir4.z, projDir4.w };

	vector4_matrix4_mult(&projDir[0], &camera.invViewMat[0], target2);

	float3 worldDir = normalize(make_float3(target2[0], target2[1], target2[2]));

	float3 cx = make_float3(camera.invViewMat[0], camera.invViewMat[1], camera.invViewMat[2]);
	float3 cy = make_float3(camera.invViewMat[4], camera.invViewMat[5], camera.invViewMat[6]);
	float3 cz = make_float3(camera.invViewMat[8], camera.invViewMat[9], camera.invViewMat[10]);

	float3 lightContribution;

	// Reset r to zero for every pixel
	lightContribution = make_float3(0.0f);

	float3 cameraPos = make_float3(camera.invViewMat[12], camera.invViewMat[13], camera.invViewMat[14]);

	// Samples per pixel
	for (size_t s = 0; s < samples; s++)
	{
		size_t jitterIndex = (s + sampleIndex) % 5u;
		float jitterX = (jitterMatrix[2u * jitterIndex]);
		float jitterY = (jitterMatrix[2u * jitterIndex + 1u]);

		// Compute primary ray direction
		float3 d = (cx * (jitterX / width) + cy * (jitterY / height));

		// Create primary ray, add incoming radiance to pixelcolor
		Ray ray = Ray(cameraPos, normalize(worldDir + d*0.5f));
		lightContribution += radiance(ray, s1, bounces) * (1.0 / samples);
	}

	// Write rgb value of pixel to image buffer on the GPU
	buf[i] += lightContribution;
}

__global__ void render_kernelDebug(float3* buf, uint32_t width, uint32_t height, Camera_GPU camera)
{
	// Assign a CUDA thread to every pixel (x,y) blockIdx, blockDim and threadIdx are CUDA specific
	// Keywords replaces nested outer loops in CPU code looping over image rows and image columns
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height))
		return;

	// Index of current pixel (calculated using thread index)
	uint32_t i = (height - y - 1) * width + x;


	float2 coord = { (float)x / (float)width, (float)y / (float)height };
	coord = coord * 2.0f - make_float2(1.0f, 1.0f); // -1 -> 1
	float viewCoord[4] = { coord.x, coord.y, 1.0f, 1.0f };
	float target[4];

	vector4_matrix4_mult(&viewCoord[0], &camera.invViewMat[0], &target[0]);

	float3 r;

	// Reset r to zero for every pixel
	r = make_float3(target[0], target[1], target[2]);

	// Write rgb value of pixel to image buffer on the GPU
	buf[i] = r;
}

__global__ void scale_accumulation_kernel(float3* m_outputBuffer, float3* m_accumulationBuffer, uint32_t width, uint32_t height, uint32_t sampleIndex)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height))
		return;

	// Index of current pixel (calculated using thread index)
	uint32_t i = (height - y - 1) * width + x;

	m_outputBuffer[i] = m_accumulationBuffer[i] / (float)sampleIndex;
}

__global__ void floatToImageData_kernel(uint32_t* outputBuffer, float3* inputBuffer, uint32_t width, uint32_t height, uint32_t sampleIndex)
{
	uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
				 

	if ((x >= width) || (y >= height))
		return;

	// Index of current pixel (calculated using thread index)
	uint32_t i = (height - y - 1) * width + x;

	outputBuffer[i] = ConvertToRGBA(inputBuffer[i] / (float)sampleIndex);
}

// Initialize and run the kernel
void CudaRenderer::Compute(void)
{
	//float3* output_buffer_gpu;                // pointer to memory for image on the device (GPU VRAM)

	// Allocate memory on the CUDA device (GPU VRAM)
	//cudaMalloc(&output_buffer_gpu, m_width * m_height * sizeof(float3));

	int tx = 8;
	int ty = 8;

	// dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 blocks(m_width / tx + 1, m_height / ty + 1, 1);
	dim3 threads(tx, ty);

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	//Camera_GPU camera_gpu
	//{
	//	m_fov,
	//	m_cameraPos,
	//	m_cameraDir,
	//	m_invViewMat,
	//	m_invProjMat
	//};

	Camera_GPU camera_buffer_obj;
	camera_buffer_obj.fov = 50.0f;
	memcpy(&camera_buffer_obj.invProjMat[0], m_invProjMat, sizeof(float) * 16);
	memcpy(&camera_buffer_obj.invViewMat[0], m_invViewMat, sizeof(float) * 16);
	memcpy(&camera_buffer_obj.viewMat[0],    m_viewMat,    sizeof(float) * 16);

	//cudaMalloc(&camera_buffer_obj, sizeof(Camera_GPU));

	// Schedule threads on device and launch CUDA kernel from host
	//render_kernelDebug <<<blocks, threads >>> (m_accumulationBuffer_GPU, m_width, m_height, camera_buffer_obj);
	render_kernel <<<blocks, threads>>> (m_accumulationBuffer_GPU, m_width, m_height, camera_buffer_obj, m_samples, *m_bounces, *m_sampleIndex);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "render_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching render_kernel!\n", cudaStatus);
		goto Error;
	}

	floatToImageData_kernel <<<blocks, threads >>> (m_imageData_GPU, m_accumulationBuffer_GPU, m_width, m_height, *m_sampleIndex);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "floatToImageData_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching floatToImageData_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(m_imageData, m_imageData_GPU, m_width * m_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//cudaFree(output_buffer_gpu);

	Error:
}

void CudaRenderer::SetCamera(float3 pos, float3 dir, float fov)
{
	m_cameraPos = pos;
	m_cameraDir = dir;
	m_fov = fov;
}

void CudaRenderer::SetInvViewMat(float4 x, float4 y, float4 z, float4 w)
{
	m_invViewMat[0]  = x.x;
	m_invViewMat[1]  = x.y;
	m_invViewMat[2]  = x.z;
	m_invViewMat[3]  = x.w;
				    
	m_invViewMat[4]  = y.x;
	m_invViewMat[5]  = y.y;
	m_invViewMat[6]  = y.z;
	m_invViewMat[7]  = y.w;

	m_invViewMat[8]  = z.x;
	m_invViewMat[9]  = z.y;
	m_invViewMat[10] = z.z;
	m_invViewMat[11] = z.w;

	m_invViewMat[12] = w.x;
	m_invViewMat[13] = w.y;
	m_invViewMat[14] = w.z;
	m_invViewMat[15] = w.w;
}

void CudaRenderer::SetInvProjMat(float4 x, float4 y, float4 z, float4 w)
{
	m_invProjMat[0] = x.x;
	m_invProjMat[1] = x.y;
	m_invProjMat[2] = x.z;
	m_invProjMat[3] = x.w;

	m_invProjMat[4] = y.x;
	m_invProjMat[5] = y.y;
	m_invProjMat[6] = y.z;
	m_invProjMat[7] = y.w;

	m_invProjMat[8] = z.x;
	m_invProjMat[9] = z.y;
	m_invProjMat[10] = z.z;
	m_invProjMat[11] = z.w;

	m_invProjMat[12] = w.x;
	m_invProjMat[13] = w.y;
	m_invProjMat[14] = w.z;
	m_invProjMat[15] = w.w;
}

void CudaRenderer::SetViewMat(float4 x, float4 y, float4 z, float4 w)
{
	m_viewMat[0] = x.x;
	m_viewMat[1] = x.y;
	m_viewMat[2] = x.z;
	m_viewMat[3] = x.w;

	m_viewMat[4] = y.x;
	m_viewMat[5] = y.y;
	m_viewMat[6] = y.z;
	m_viewMat[7] = y.w;

	m_viewMat[8] = z.x;
	m_viewMat[9] = z.y;
	m_viewMat[10] = z.z;
	m_viewMat[11] = z.w;

	m_viewMat[12] = w.x;
	m_viewMat[13] = w.y;
	m_viewMat[14] = w.z;
	m_viewMat[15] = w.w;
}
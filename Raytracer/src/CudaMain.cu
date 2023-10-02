#include <cmath>

#include "CudaMain.cuh"
#include "cutil_math.cuh"

#define M_PI 3.14159265359f  // pi
#define M_DEG2RAD 0.01745329252

int checkCudaError(cudaError_t& error)
{
	if (error == cudaSuccess)
	{
		return 0;
	}

	return 1;
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

struct HitInfo
{
	bool didHit = false;
	float dst = FLT_MAX;
	float3 hitPoint {0.0f, 0.0f, 0.0f};
	float3 normal{ 0.0f, 0.0f, 0.0f };
	size_t materialIndex;
};

struct Sphere
{
	float rad;            // Radius
	float3 pos;           // Position
	size_t materialIndex; // Material Index
};

struct Camera_GPU
{
	float localToWorldMatrix[16];
	float invViewMat[16];
	float invProjMat[16];
	float viewMat[16];
	float aperture;
	float focusDist;
	float3 pos;
};

void CudaRenderer::Clear()
{
	cudaDeviceSynchronize();
	memset(m_outputBuffer, 0, m_bufferSize);
	memset(m_imageData, 0, m_width * m_height * sizeof(uint32_t));
	cudaMemset(m_accumulationBuffer_GPU, 0, m_bufferSize);
	cudaMemset(m_imageData_GPU, 0, m_width * m_height * sizeof(uint32_t));
}

// SCENE 9 spheres forming a Cornell box small enough to be in constant GPU memory 
__constant__ Sphere spheres[] =
{
	  { 1e5f,{ 1e5f + 1.0f, 40.8f, 81.6f },     0u }, //Left
	  { 1e5f,{ -1e5f + 99.0f, 40.8f, 81.6f },   1u }, //Right
	  { 1e5f,{ 50.0f, 40.8f, 1e5f },            3u }, //Back
	  { 1e5f,{ 50.0f, 40.8f, -1e5f + 600.0f },  2u }, //Frnt     	   
	  { 1e5f,{ 50.0f, 1e5f, 81.6f },            2u }, //Botm
	  { 1e5f,{ 50.0f, -1e5f + 81.6f, 81.6f },   2u }, //Top			   
	  { 16.5f,{ 27.0f, 16.5f, 47.0f },          2u }, // small sphere 1
	  { 16.5f,{ 73.0f, 16.5f, 78.0f },          4u }, // gold sphere 2
	  { 16.5f,{ 73.0f, 16.5f, 118.0f },         5u }, // copper sphere 2
	  { 100.0f,{ 30.0f, 181.6f - 1.9f, 80.0f }, 6u }, // Light
	  { 100.0f,{ 70.0f, 181.6f - 1.9f, 80.0f }, 7u }  // Light
	  //{ 2.1f,{ 40.0f, 40.5f, 47.0f }, Material{ { 0.8f, 0.8f, 0.8f }, 0.1f, { 150.0f, 160.0f, 180.0f }, 0.0f} }      // Light
};

__constant__  Sphere spheresSimple[] =
{
	//{ float radius, { float3 position }, { Material }}
	  { 1.0f, { -4.0f, 1.0f, 0.0f }, 4u},
	  { 19.0f, { 0.0f, -19.0f, 0.0f }, 2u},
	  { 1.0f, { -6.5f, 1.0f, 0.0f }, 2u}
};

__constant__ static float jitterMatrix[10] =
{
   -0.25,  0.75,
	0.75,  0.33333,
   -0.75, -0.25,
	0.25, -0.75,
	0.0f, 0.0f
};

__device__ static float fresnel_schlick_ratio(float cos_theta_incident, float power)
{
	float p = 1.0f - cos_theta_incident;
	return pow(p, power);
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

__device__ float2 randomPointInCircle(uint32_t& state)
{
	float angle = randomValue(state) * 2 * M_PI;
	float2 pointOnCircle = make_float2(cos(angle), sin(angle));
	return pointOnCircle * fsqrtf(randomValue(state));
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

__device__ float3 getEnvironmentLight(const Ray& ray, const Scene* scene)
{
	float3 sunDir = make_float3(1.0f, 1.0f, 1.0f);
	sunDir = normalize(sunDir);

	float skyGradientT = powf(fmaxf(ray.direction.y, 0.0f), 0.35f);
	float groundToSkyT = powf(fmaxf(ray.direction.y, 0.0f), 0.1f);

	float3 skyGradient = lerp(srgbToLinear(scene->skyColorHorizon), srgbToLinear(scene->skyColorZenith), skyGradientT);
	float sun = powf(fmaxf(0.0f, dot(ray.direction, sunDir)), scene->sunFocus) * scene->sunIntensity;

	// Combine ground, sky, and sun
	float3 composite = lerp(srgbToLinear(scene->groundColor), skyGradient, groundToSkyT) * scene->skyBrightness + sun;

	return composite * scene->skyColor;
}

__device__ HitInfo intersect_sphere(const Ray& r, const Sphere& s)
{
	HitInfo hit;

	float3 offsetRayOrigin = r.origin - s.pos;
	float a = dot(r.direction, r.direction);
	float b = 2.0f * dot(offsetRayOrigin, r.direction);
	float c = dot(offsetRayOrigin, offsetRayOrigin) - s.rad * s.rad;
	// Quadratic discriminant
	float discriminant = b * b - 4.0f * a * c;

	// No solution when d < 0 (ray misses sphere)
	if (discriminant >= 0.0f)
	{
		// Distance to nearest intersection point (from quadratic formula)
		float dst = (-b - fsqrtf(discriminant)) / (2.0f * a);

		// Ignore intersections that occur behind the ray
		if (dst >= 0.0f)
		{
			hit.didHit = true;
			hit.dst = dst;
			hit.hitPoint = r.origin + r.direction * dst;
			hit.normal = normalize(hit.hitPoint - s.pos);
			hit.materialIndex = s.materialIndex;
		}
	}

	return hit;
}

__device__ HitInfo rayTriangleIntersect(const Ray& ray, const GPU_Mesh::Triangle& tri)
{
	float3 edgeAB = tri.pos1 - tri.pos0;
	float3 edgeAC = tri.pos2 - tri.pos0;
	float3 normalVector = cross(edgeAB, edgeAC);
	float3 ao = (ray.origin) - tri.pos0;
	float3 dao = cross(ao, ray.direction);

	float determinant = -dot(ray.direction, normalVector);
	float invDet = 1.0f / determinant;

	// Calculate dst to triangle & barycentric coordinates of intersection point
	float dst = dot(ao, normalVector) * invDet;
	float u = dot(edgeAC, dao) * invDet;
	float v = -dot(edgeAB, dao) * invDet;
	float w = 1.0f - u - v;

	// Initialize hit info
	HitInfo hit;
	hit.didHit = determinant >= 1E-6 && dst >= 0.0f && u >= 0.0f && v >= 0.0f && w >= 0.0f;
	hit.hitPoint = (ray.origin) + ray.direction * dst;
	hit.normal = normalize(tri.n0 * w + tri.n1 * u + tri.n2 * v);
	//hit.normal = normalVector;
	hit.dst = dst;

	return hit;
}

__device__ bool rayBoxIntersection(const Ray& r, const float3& min, const float3& max)
{
	float t[9];
	t[1] = (min.x - r.origin.x) / r.direction.x;
	t[2] = (max.x - r.origin.x) / r.direction.x;
	t[3] = (min.y - r.origin.y) / r.direction.y;
	t[4] = (max.y - r.origin.y) / r.direction.y;
	t[5] = (min.z - r.origin.z) / r.direction.z;
	t[6] = (max.z - r.origin.z) / r.direction.z;
	t[7] = fmaxf(fmax(fminf(t[1], t[2]), fminf(t[3], t[4])), fminf(t[5], t[6]));
	t[8] = fminf(fmin(fmaxf(t[1], t[2]), fmaxf(t[3], t[4])), fmaxf(t[5], t[6]));
	//t[9] = (t[8] < 0 || t[7] > t[8]) ? FLT_MAX : t[7];

	return (t[8] < 0 || t[7] > t[8]);
}

__device__ HitInfo intersect_triangles(const Ray& r, const GPU_Mesh* vbo)
{
	HitInfo hit;
	HitInfo closestHit;

	for (size_t mID = 0; mID < vbo->numMeshes; mID++)
	{
		if (rayBoxIntersection(r, vbo->meshInfoBuffer[mID].bboxMin, vbo->meshInfoBuffer[mID].bboxMax))
		{
			continue;
		}
		for (size_t tID = 0; tID < vbo->meshInfoBuffer[mID].triangleCount; tID++) // Test all scene objects for intersection
		{
			hit = rayTriangleIntersect(r, vbo->triangleBuffer[vbo->meshInfoBuffer[mID].firstTriangleIndex + tID]);

			if (hit.didHit && hit.dst < closestHit.dst) // If newly computed intersection distance d is smaller than current closest intersection distance
			{
				closestHit = hit;
				closestHit.materialIndex = vbo->meshInfoBuffer[mID].materialIndex;
			}
		}
}

	// Returns true if an intersection with the scene occurred, false when no hit
	//closestHit.materialIndex = bboxHitCount;
	
	return closestHit;
}

__device__ HitInfo intersect_scene(const Ray& r, const GPU_Mesh* vbo)
{
	HitInfo hit;
	HitInfo closestHit;

	float n = sizeof(spheresSimple) / sizeof(Sphere);

	for (size_t i = 0u; i < size_t(n); i++)
	{
		Sphere s = spheresSimple[i];
		hit = intersect_sphere(r, s);

		if (hit.didHit && hit.dst < closestHit.dst) // If newly computed intersection distance d is smaller than current closest intersection distance
		{
			closestHit = hit;
		}
	}

	hit = intersect_triangles(r, vbo);

	if (hit.didHit && hit.dst < closestHit.dst)
	{
		closestHit = hit;
	}

	// Returns true if an intersection with the scene occurred, false when no hit
	return closestHit;
}

__device__ float3 radiance(Ray& r, uint32_t& s1, const Scene* scene, size_t bounces, const GPU_Mesh* vbo) // Returns ray color
{
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	for (size_t b = 0; b < bounces; b++)
	{
		// Test ray for intersection with scene
		HitInfo hit = intersect_scene(r, vbo);

		if (!hit.didHit)
		{
			//accucolor += mask * make_float3(0.0494, 0.091, 0.164f); // If miss, return sky
			accucolor += mask * getEnvironmentLight(r, scene);
			break;
		}

		Material hitMat = scene->materials[hit.materialIndex];

		accucolor += mask * srgbToLinear(hitMat.emission) * hitMat.emissionIntensity;

		// Create 2 random numbers
		float r1 = 2 * M_PI * randomValue(s1); // Pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
		float r2 = randomValue(s1);            // Pick random number for elevation
		float r2s = sqrtf(r2);

		float ndotl = fmaxf(dot(-r.direction, hit.normal), 0.0f);
		float f = fresnel_schlick_ratio(ndotl, 8.0f);

		bool isSpecularBounce = max(hitMat.metalness, max(f, 0.02f)) >= randomValue(s1);

		float3 diffuseDir = normalize(hit.normal + randomDirection(s1));
		float3 specularDir = reflect(r.direction, normalize(hit.normal + randomDirection(s1) * hitMat.roughness));

		float3 linearSurfColor = powf(hitMat.albedo, 2.2f);

		r.direction = normalize(lerp(diffuseDir, specularDir, isSpecularBounce));
		r.origin = hit.hitPoint + hit.normal * 0.001f; // offset ray origin slightly to prevent self intersection

		mask = mask * lerp(linearSurfColor, lerp(make_float3(1.0f), linearSurfColor, hitMat.metalness), isSpecularBounce);

		//float p = max(mask.x, max(mask.y, mask.z));
		//if (randomValue(s1) >= p)
		{
		//	break;
		}
		//mask *= 1.0f / p;

		//Debug output
		//accucolor = { hit.dst*0.01f };
	}


	return accucolor;
}

__global__ void render_kernel(float3* buf, uint32_t width, uint32_t height, Camera_GPU camera, const Scene* scene, size_t samples,
							  size_t bounces, uint32_t sampleIndex, const GPU_Mesh* vbo)
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


	float2 coord = { (float)x / (float)width, (float)y / (float)height };
	coord = (coord * 2.0f) - make_float2(1.0f, 1.0f); // -1 -> 1

	float3 lightContribution;

	// Reset r to zero for every pixel
	lightContribution = make_float3(0.0f);

	// Calculate focus point
	float viewPointLocal[4] = { coord.x, coord.y, 1.0f, 1.0f };
	float target[4];
	vector4_matrix4_mult(&viewPointLocal[0], &camera.localToWorldMatrix[0], target);

	float3 viewPoint = make_float3(target[0], target[1], target[2]) * camera.focusDist + camera.pos;
	float3 camRight = make_float3(camera.localToWorldMatrix[0], camera.localToWorldMatrix[1], camera.localToWorldMatrix[2]);
	float3 camUp = make_float3(camera.localToWorldMatrix[4], camera.localToWorldMatrix[5], camera.localToWorldMatrix[6]);

	// Samples per pixel
	for (size_t s = 0; s < samples; s++)
	{
		// Create primary ray, add incoming radiance to pixelcolor
		Ray ray = Ray(camera.pos, {0.0f, 0.0f, 0.0f});

		float2 defocusJitter = randomPointInCircle(s1) * (1.0f / camera.aperture);
		ray.origin = camera.pos + camRight * defocusJitter.x + camUp * defocusJitter.y;

		float2 jitter = randomPointInCircle(s1) * 0.01f;
		float3 jitteredViewPoint = viewPoint + camRight * jitter.x + camUp * jitter.y;

		//ray.direction = normalize(jitteredViewPoint) * focusDistance;
		ray.direction = normalize(jitteredViewPoint - ray.origin);

		lightContribution += radiance(ray, s1, scene, bounces, vbo) * (1.0 / samples);
	}

	// Write rgb value of pixel to image buffer on the GPU
	buf[i] += lightContribution;
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

	cudaMalloc(&m_deviceScene, sizeof(Scene));
	cudaMemcpy(m_deviceScene, *m_scene, sizeof(Scene), cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	Camera_GPU camera_buffer_obj;
	memcpy(&camera_buffer_obj.invProjMat[0],		 m_invProjMat,      sizeof(float) * 16);
	memcpy(&camera_buffer_obj.invViewMat[0],		 m_invViewMat,      sizeof(float) * 16);
	memcpy(&camera_buffer_obj.viewMat[0],			 m_viewMat,         sizeof(float) * 16);
	memcpy(&camera_buffer_obj.localToWorldMatrix[0], m_localToWorldMat, sizeof(float) * 16);
	camera_buffer_obj.pos = m_cameraPos;
	camera_buffer_obj.aperture = m_aperture;
	camera_buffer_obj.focusDist = m_focusDist;

	render_kernel <<<blocks, threads>>> (m_accumulationBuffer_GPU, m_width, m_height, camera_buffer_obj, m_deviceScene, m_samples, *m_bounces, *m_sampleIndex, m_deviceMesh);

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

void CudaRenderer::SetCamera(float3 pos, float3 dir, float aperture, float focusDist)
{
	m_cameraPos = pos;
	m_cameraDir = dir;
	m_aperture = aperture;
	m_focusDist = focusDist;
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

void CudaRenderer::SetLocalToWorldMat(float4 x, float4 y, float4 z, float4 w)
{
	m_localToWorldMat[0] = x.x;
	m_localToWorldMat[1] = x.y;
	m_localToWorldMat[2] = x.z;
	m_localToWorldMat[3] = x.w;

	m_localToWorldMat[4] = y.x;
	m_localToWorldMat[5] = y.y;
	m_localToWorldMat[6] = y.z;
	m_localToWorldMat[7] = y.w;

	m_localToWorldMat[8] = z.x;
	m_localToWorldMat[9] = z.y;
	m_localToWorldMat[10] = z.z;
	m_localToWorldMat[11] = z.w;

	m_localToWorldMat[12] = w.x;
	m_localToWorldMat[13] = w.y;
	m_localToWorldMat[14] = w.z;
	m_localToWorldMat[15] = w.w;
}

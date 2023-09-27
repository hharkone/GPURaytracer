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

struct Ray
{
	float3 origin; // ray origin
	float3 direction;  // ray direction
	__device__ Ray(float3 o_, float3 d_) : origin(o_), direction(d_) {}
};

struct Material
{
	float3 albedo    { 0.8f, 0.8f, 0.8f };
	float  roughness { 0.2f };
	float3 emission  { 0.0f, 0.0f, 0.0f };
	float  metalness = 0.0f;
};

struct Sphere
{
	float rad;            // Radius
	float3 pos;           // Position
	Material mat;         // Material

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
};

__constant__ static float jitterMatrix[10] =
{
   -0.25,  0.75,
	0.75,  0.33333,
   -0.75, -0.25,
	0.25, -0.75,
	0.0f, 0.0f
};

// SCENE 9 spheres forming a Cornell box small enough to be in constant GPU memory 
__constant__ Sphere spheres[] =
{
	//{ float radius, { float3 position },      { float3 emission }, { float3 colour },       refl_type }
	  { 1e5f,{ 1e5f + 1.0f, 40.8f, 81.6f },     Material{} }, //Left
	  { 1e5f,{ -1e5f + 99.0f, 40.8f, 81.6f },   Material{} }, //Rght
	  //{ 1e5f,{ 50.0f, 40.8f, 1e5f },            Material{} }, //Back
	  { 1e5f,{ 50.0f, 40.8f, -1e5f + 600.0f },  Material{} }, //Frnt
	  { 1e5f,{ 50.0f, 1e5f, 81.6f },            Material{} }, //Botm
	  { 1e5f,{ 50.0f, -1e5f + 81.6f, 81.6f },   Material{} }, //Top
	  { 16.5f,{ 27.0f, 16.5f, 47.0f },          Material{} }, // small sphere 1
	  { 16.5f,{ 73.0f, 16.5f, 78.0f },          Material{} }, // small sphere 2
	  { 100.0f,{ 30.0f, 181.6f - 1.9f, 80.0f }, Material{} },  // Light
	  { 100.0f,{ 70.0f, 181.6f - 1.9f, 80.0f }, Material{} }  // Light
};
__constant__ Sphere spheresSimple[] =
{
	//{ float radius, { float3 position }, { Material }}
	  { 0.5f, { 0.0f, 0.0f, 0.0f },  Material{} },
	  { 0.5f, { 0.0f, -1.0f, 0.0f }, Material{ {0.0f, 0.0f, 0.0f}, 0.0f, {1.0f, 1.0f, 0.0f}, 0.0f }}
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

__device__ inline bool intersect_scene(const Ray& r, float& t, int& id)
{
	float n = sizeof(spheresSimple) / sizeof(Sphere), d, inf = t = 1e20;  // t is distance to closest intersection, initialise t to a huge number outside scene
	for (int i = int(n); i--;)                                      // Test all scene objects for intersection
	{
		if ((d = spheresSimple[i].intersect_sphere(r)) && d < t) // If newly computed intersection distance d is smaller than current closest intersection distance
		{
			t = d;  // Keep track of distance along ray to closest intersection point
			id = i; // and closest intersected object
		}
	}
	// Returns true if an intersection with the scene occurred, false when no hit
	return t < inf;
}

// Radiance function, the meat of path tracing solves the rendering equation: outgoing radiance (at a
// point) = emitted radiance + reflected radiance reflected radiance is sum (integral) of incoming
// radiance from all directions in hemisphere above point, multiplied by reflectance function of
// material (BRDF) and cosine incident angle
__device__ float3 radiance(Ray& r, unsigned int* s1, unsigned int* s2, size_t bounces) // Returns ray color
{
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	// Ray bounce loop (no Russian Roulette used)
	for (size_t b = 0; b < bounces; b++) // Iteration up to 4 bounces (replaces recursion in CPU code)
	{
		float t;           // Distance to closest intersection
		int id = 0;        // Index of closest intersected sphere

		// Test ray for intersection with scene
		if (!intersect_scene(r, t, id))
		{
			accucolor += mask * make_float3(0.1f, 0.12f, 0.2f); // If miss, return sky
		}

		// Else, we've got a hit! compute hitpoint and normal
		const Sphere& obj = spheresSimple[id];               // hitobject
		float3 x = r.origin + r.direction * t;               // hitpoint
		float3 n = normalize(x - obj.pos);                   // normal
		float3 nl = dot(n, r.direction) < 0 ? n : n * -1;    // front facing normal

		// Add emission of current sphere to accumulated
		// colour (first term in rendering equation sum)
		accucolor += mask * obj.mat.emission;

		// All spheres in the scene are diffuse diffuse material reflects light uniformly in all
		// directions generate new diffuse ray: origin = hitpoint of previous ray in path random
		// direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)

		// Create 2 random numbers
		float r1 = 2 * M_PI * getrandom(s1, s2); // Pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
		float r2 = getrandom(s1, s2);            // Pick random number for elevation
		float r2s = sqrtf(r2);

		// Compute local orthonormal basis uvw at hitpoint to use for calculation random ray
		// direction first vector = normal at hitpoint, second vector is orthogonal to first, third
		// vector is orthogonal to first two vectors
		float3 w = nl;
		float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);

		// Compute random ray direction on hemisphere using polar coordinates cosine weighted
		// importance sampling (favours ray directions closer to normal direction)
		float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

		// New ray origin is intersection point of previous ray with scene
		r.origin = x + nl * 0.05f; // offset ray origin slightly to prevent self intersection
		r.direction = d;

		mask = mask * obj.mat.albedo;     // Multiply with colour of object
		//mask *= dot(d, nl);		      // Weigh light contribution using cosine of angle between incident light and normal
		//mask *= 2;                      // Fudge factor
	}

	return accucolor;
}

// __global__ : executed on the device (GPU) and callable only from host (CPU) this kernel runs in
// parallel on all the CUDA threads
__global__ void render_kernel(float3* buf, size_t width, size_t height, float fovY, size_t samples, size_t bounces, size_t sampleIndex)
{
	// Assign a CUDA thread to every pixel (x,y) blockIdx, blockDim and threadIdx are CUDA specific
	// Keywords replaces nested outer loops in CPU code looping over image rows and image columns
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height)) return;

	// Index of current pixel (calculated using thread index)
	unsigned int i = (height - y - 1) * width + x;

	// Seeds for random number generator
	unsigned int s1 = x + sampleIndex + i;
	unsigned int s2 = y + sampleIndex + i;
	const float fov = fovY * M_PI / 180.0f;
	const float tf = std::tan(fov * 0.5f);

	// First hardcoded camera ray(origin, direction)
	Ray cam(make_float3(0.0f, 0.0f, 5.0f), normalize(make_float3(0.0f, 0.0f, -1.0f)));
	float3 cx = make_float3(width * tf / height, 0.0f, 0.0f);                    // Ray direction offset in x direction
	float3 cy = normalize(cross(cx, cam.direction)) * tf;                              // Ray direction offset in y direction (.5135 is field of view angle)
	float3 r;

	// Reset r to zero for every pixel
	r = make_float3(0.0f);

	// Samples per pixel
	for (size_t s = 0; s < samples; s++)
	{
		size_t jitterIndex = (s + sampleIndex) % 5u;
		float jitterX = 2.0 * (x + jitterMatrix[2u * jitterIndex]) / (float)width;
		float jitterY = 2.0 * (y + jitterMatrix[2u * jitterIndex + 1u]) / (float)height;

#ifndef MSAA_4X
		//jitterX = jitterY = 0.0f;
#endif // !MSAA_4X

		// Compute primary ray direction
		float3 d = cam.direction + (cx * ((.25 + x + jitterX) / width - .5) + cy * ((.25 + y + jitterY) / height - .5));

		// Create primary ray, add incoming radiance to pixelcolor
		r = r + radiance(Ray(cam.origin, normalize(d)), &s1, &s2, bounces) * (1.0 / samples);
	}   // Camera rays are pushed ^^^^^ forward to start in interior

	// Write rgb value of pixel to image buffer on the GPU
	buf[i] = r;
}

__global__ void render_kernelDebug(float3* buf, size_t width, size_t height)
{
	// Assign a CUDA thread to every pixel (x,y) blockIdx, blockDim and threadIdx are CUDA specific
	// Keywords replaces nested outer loops in CPU code looping over image rows and image columns
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height)) return;

	// Index of current pixel (calculated using thread index)
	unsigned int i = (height - y - 1) * width + x;

	float3 r;

	// Reset r to zero for every pixel
	r = make_float3(1.0f, 0.5, 0.0f);

	// Write rgb value of pixel to image buffer on the GPU
	buf[i] = r;
}

// Initialize and run the kernel
void CudaRenderer::Compute(void)
{
	float3* output_buffer_gpu;                // pointer to memory for image on the device (GPU VRAM)

	// Allocate memory on the CUDA device (GPU VRAM)
	cudaMalloc(&output_buffer_gpu, m_width * m_height * sizeof(float3));

	int tx = 8;
	int ty = 8;

	// dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 blocks(m_width / tx + 1, m_height / ty + 1, 1);
	dim3 threads(tx, ty);

	// Schedule threads on device and launch CUDA kernel from host
	//render_kernelDebug <<<blocks, threads >>> (output_buffer_gpu, m_width, m_height);
	render_kernel <<<blocks, threads >>> (output_buffer_gpu, m_width, m_height, m_fov, m_samples, m_bounces, m_sampleIndex);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaDeviceSynchronize();

	// Copy results of computation from device back to host
	//render_kernel(float3* buf, size_t width, size_t height, float fovY, size_t samples, size_t bounces, size_t sampleIndex)
	cudaMemcpy(m_accumulationBuffer, output_buffer_gpu, m_bufferSize, cudaMemcpyDeviceToHost);
	cudaFree(output_buffer_gpu);
}
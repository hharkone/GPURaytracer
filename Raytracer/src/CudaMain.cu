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

__device__ inline uint32_t ConvertToRGBA(const float4& color)
{
	float3 outColor;
	outColor.x = clamp(color.x, 0.0f, 1.0f);
	outColor.y = clamp(color.y, 0.0f, 1.0f);
	outColor.z = clamp(color.z, 0.0f, 1.0f);

	float alpha = clamp(color.w, 0.0f, 1.0f);

	outColor = powf(outColor, 0.4646464);

	uint8_t r = (uint8_t)(outColor.x * 255.0f);
	uint8_t g = (uint8_t)(outColor.y * 255.0f);
	uint8_t b = (uint8_t)(outColor.z * 255.0f);
	uint8_t a = (uint8_t)(alpha * 255.0f);

	uint32_t returnValue = (a << 24) | (b << 16) | (g << 8) | r;

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
	bool inside = false;
	float dst = FLT_MAX;
	float3 hitPoint {0.0f, 0.0f, 0.0f};
	float3 normal{ 0.0f, 0.0f, 0.0f };
	size_t materialIndex = 0u;
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
	memset(m_finalOutputBuffer, 0, m_bufferSize);

	//m_accumulationBuffer_GPU.clear();
	m_floatOutputBuffer_GPU.clear();
}

__device__ float fresnel(float cos_theta_incident, float cos_critical, float refractive_ratio)
{
	if (cos_theta_incident <= cos_critical)
		return 1.f;

	float sin_theta_incident2 = 1.f - cos_theta_incident * cos_theta_incident;
	float t = fsqrtf(1.f - sin_theta_incident2 / (refractive_ratio * refractive_ratio));
	float sqrtRs = (cos_theta_incident - refractive_ratio * t) / (cos_theta_incident + refractive_ratio * t);
	float sqrtRp = (t - refractive_ratio * cos_theta_incident) / (t + refractive_ratio * cos_theta_incident);

	return lerp(sqrtRs * sqrtRs, sqrtRp * sqrtRp, .5f);
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

__device__ float2 randomPointInCircle(uint32_t& state, float sigma)
{
	float angle = randomValue(state) * 2.0f * M_PI;
	float2 pointOnCircle = make_float2(cos(angle), sin(angle));
	return pointOnCircle * powf(fsqrtf(randomValue(state)), sigma);
}

__device__ float2 randomInUnitHex(uint32_t& state)
{
	float2 vectors[3] =
	{
		make_float2(-1.0f, 0.0f),
		make_float2(0.5f, fsqrtf(3.0f) / 2.0f),
		make_float2(0.5f, -fsqrtf(3.0f) / 2.0f)
	};

	uint16_t t = (uint16_t)randomValue(state) * 3.0f;

	float2 v1 = vectors[t];
	float2 v2 = vectors[(t + 1) % 3];

	float x = randomValue(state) * 2.0f - 1.0f;
	float y = randomValue(state) * 2.0f - 1.0f;

	return make_float2(x * v1.x + y * v2.x, x * v2.y + y * v2.y);
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

__device__ float3 randomInUnitSphere(uint32_t& state)
{
	// Thanks to https://math.stackexchange.com/a/1585996
	float x = randomValueNormalDistribution(state);
	float y = randomValueNormalDistribution(state);
	float z = randomValueNormalDistribution(state);

	float distance = randomValue(state);
	float dsqr = fsqrtf(distance);

	return normalize(make_float3(x, y, z)) * dsqr;
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

__device__ float2 toSpherical(float3 dir, const float rot)
{
	dir = normalize(dir);
	float u = (atan2f(dir.z, dir.x) / (M_PI * 2.0f)) + 0.5;
	float v = acos(-dir.y) / M_PI;

	u += rot* 0.002777777777f;
	float uselessshit;

	return make_float2(modf(u, &uselessshit), 1.0f-v);
}

__device__ float3 getEnvironmentLight(const Ray& ray, const Scene* scene, float* skyTex)
{
	switch(scene->envType)
	{
		case EnvironmentType::EnvType_ProceduralSky:
		{
			float3 sunDir = normalize(scene->sunDirection);

			float skyGradientT = powf(fmaxf(ray.direction.y, 0.0f), 0.5f);
			float groundToSkyT = powf(fmaxf(ray.direction.y, 0.0f), 0.15f);

			float3 skyGradient = lerp(srgbToLinear(scene->skyColorHorizon), srgbToLinear(scene->skyColorZenith), skyGradientT);
			float sun = powf(fmaxf(0.0f, dot(ray.direction, sunDir)), scene->sunFocus) * scene->sunIntensity;

			// Combine ground, sky, and sun
			float3 composite = lerp(srgbToLinear(scene->groundColor), skyGradient, groundToSkyT) * scene->skyBrightness + sun;

			return composite * scene->skyColor;
		}
		case EnvironmentType::EnvType_Solid:
		{
			return scene->skyColor * scene->skyBrightness;
		}
		case EnvironmentType::EnvType_HDRI:
		{
			size_t texWidth = 8192u;
			size_t texHeight = 4096u;

			float2 uv = toSpherical(ray.direction, scene->skyRotation);
			size_t row = (size_t)(uv.y * (float)texHeight) * 3;
			size_t col = (size_t)(uv.x * (float)texWidth) * 3;

			//pixel = (float*)(skyTex + row * 8000u) + 4 * col;

			//size_t pixelIndex = (col * texHeight * 3) + row;
			size_t pixelIndex = (row * texWidth) + col;

			//pixel = ((float*)skyTex + pixelIndex);
			float r = *(skyTex + pixelIndex + 0);
			float g = *(skyTex + pixelIndex + 1);
			float b = *(skyTex + pixelIndex + 2);

			float3 c = make_float3(r,g,b);

			return c * scene->skyColor * scene->skyBrightness;
		}
	}
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
		float t0 = (-b - fsqrtf(discriminant)) / (2.0f * a);
		float t1 = (-b + fsqrtf(discriminant)) / (2.0f * a);

		float dist;
		if (t0 < 0.0f)
			dist = t1;
		else
			dist = t0;

		// Ignore intersections that occur behind the ray
		if (dist > 0.0f)
		{
			hit.didHit = true;
			hit.dst = dist;
			hit.hitPoint = r.origin + r.direction * hit.dst;
			hit.normal = normalize(hit.hitPoint - s.pos);
			hit.inside = (t0 < 0.0f);
			hit.materialIndex = s.materialIndex;
		}
	}

	return hit;
}

__device__ HitInfo rayTriangleIntersect(const Ray& ray, const GPU_Mesh::Triangle* tri)
{
	float3 edgeAB = tri->pos1 - tri->pos0;
	float3 edgeAC = tri->pos2 - tri->pos0;
	float3 normalVector = cross(edgeAB, edgeAC);
	float3 ao = (ray.origin) - tri->pos0;
	float3 dao = cross(ao, ray.direction);

	float determinant = -dot(ray.direction, normalVector);
	float invDet = 1.0f / determinant;

	// Calculate dst to triangle & barycentric coordinates of intersection point
	float dst = dot(ao, normalVector) * invDet;
	float u = dot(edgeAC, dao) * invDet;
	float v = -dot(edgeAB, dao) * invDet;
	float w = 1.0f - u - v;

	float3 normal = normalize(tri->n0 * w + tri->n1 * u + tri->n2 * v);

	if (dot(normal, normal) >= 1.001f)
	{
		normal = normalize(tri->n0 + tri->n1 + tri->n2);
	}

	// Initialize hit info
	HitInfo hit;
	//hit.didHit = determinant >= 1E-6 && dst >= 0.0f && u >= 0.0f && v >= 0.0f && w >= 0.0f;
	hit.didHit = dst >= 0.0f && u >= 0.0f && v >= 0.0f && w >= 0.0f;
	hit.hitPoint = (ray.origin) + ray.direction * dst;
	hit.normal = normal;
	//hit.normal = normalize(normalVector);
	hit.dst = dst;
	hit.inside = (dot(normalVector, ray.direction) > 0.0f ? true : false);
	hit.materialIndex = (size_t)tri->uv0.x;

	return hit;
}

void __device__ IntersectTri(Ray& ray, HitInfo& hit, GPU_Mesh::Triangle* tri)
{
	float3 edge1 = tri->pos1 - tri->pos0, edge2 = tri->pos2 - tri->pos0;
	float3 h = cross(ray.direction, edge2);
	float a = dot(edge1, h);
	if (a > -0.00001f && a < 0.00001f) return; // ray parallel to triangle
	float f = 1 / a;
	float3 s = ray.origin - tri->pos0;
	float u = f * dot(s, h);
	if (u < 0 || u > 1) return;
	float3 q = cross(s, edge1);
	float v = f * dot(ray.direction, q);
	if (v < 0 || u + v > 1) return;
	float t = f * dot(edge2, q);
	if (t > 0.0001f && t < hit.dst)
	{
		hit.dst = t;
		hit.didHit = true;
	}
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

__device__ bool rayBoxIntersectionDebug(const Ray& ray, HitInfo& hit, const float3& bmin, const float3& bmax)
{
	float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));

	bool didHit (tmax >= tmin && tmin < hit.dst && tmax > 0);

	float3 c = (bmin + bmax) * 0.5f;

	hit.didHit = didHit;
	hit.dst = tmin;
	hit.hitPoint = ray.direction * tmin + ray.origin;

	float3 p = hit.hitPoint - c;
	float3 d = (bmin - bmax) * 0.5f;

	float bias = 1.0001f;

	hit.normal = normalize( make_float3(float(int(p.x / abs(d.x) * bias)),
										float(int(p.y / abs(d.y) * bias)),
										float(int(p.z / abs(d.z) * bias))) );

	//hit.hitPoint = c;

	return didHit;
}

__device__ float IntersectAABB(const Ray& ray, const HitInfo& hit, const float3 bmin, const float3 bmax)
{
	float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	if( tmax >= tmin && tmin < hit.dst && tmax > 0) return tmin; else return FLT_MAX;
}

__device__ void IntersectBVH(Ray& ray, HitInfo& hit, const GPU_Mesh* vbo, int debug)
{

#define USE_BVH 1

#if	USE_BVH == 0
	HitInfo closestHit;

	for (uint16_t i = 0; i < vbo->numTris; i++)
	{
		hit = rayTriangleIntersect(ray, &vbo->triangleBuffer[i]);

		if (hit.didHit && hit.dst < closestHit.dst)
		{
			closestHit = hit;
		}
	}

	hit = closestHit;

#endif

#if USE_BVH == 1

	GPU_Mesh::BVHNode* node = &vbo->bvhNode[0];
	GPU_Mesh::BVHNode* stack[256];

	uint32_t stackPtr = 0;

	HitInfo closestHit;

	while (1)
	{
		if (node->triCount > 0) // isLeaf()
		{
			for (uint32_t i = 0; i < node->triCount; i++)
			{
				uint32_t instPrim = vbo->triIdx[node->leftFirst + i];
				GPU_Mesh::Triangle* triangle = &vbo->triangleBuffer[instPrim];

				//if(stackPtr < debug)
				hit = rayTriangleIntersect(ray, triangle);

				if (hit.didHit && hit.dst < closestHit.dst)
				{
					closestHit = hit;
				}
			}
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}

		GPU_Mesh::BVHNode* child1 = &vbo->bvhNode[node->leftFirst];
		GPU_Mesh::BVHNode* child2 = &vbo->bvhNode[node->leftFirst +1];

		float dist1 = IntersectAABB(ray, closestHit, child1->aabbMin, child1->aabbMax);
		float dist2 = IntersectAABB(ray, closestHit, child2->aabbMin, child2->aabbMax);

		//
		//hit.dst = fminf(dist1, dist2);

		if (dist1 > dist2)
		{
			float d = dist1; dist1 = dist2; dist2 = d;
			GPU_Mesh::BVHNode* c = child1; child1 = child2; child2 = c;
		}
		if (dist1 == FLT_MAX)
		{
			if (stackPtr == 0)
			{
				break;
			}
			else
			{
				node = stack[--stackPtr];
			}
		}
		else
		{
			node = child1;
			if (dist2 != FLT_MAX) stack[stackPtr++] = child2;
		}
	}

	hit = closestHit;

#elif USE_BVH == 2

	HitInfo closestHit;

	GPU_Mesh::BVHNode* child1 = &vbo->bvhNode[debug];

	if (rayBoxIntersectionDebug(ray, closestHit, child1->aabbMin, child1->aabbMax))
	{
		hit = closestHit;
	}

#endif
}

__device__ HitInfo intersect_scene(Ray& r, const Scene* scene, const GPU_Mesh* vbo, int debug)
{
	HitInfo hit;
	HitInfo closestHit;

	for (size_t i = 0u; i < scene->sphereCount; i++)
	{
		Sphere s = scene->spheresSimple[i];
		hit = intersect_sphere(r, s);

		if (hit.didHit && hit.dst < closestHit.dst) // If newly computed intersection distance d is smaller than current closest intersection distance
		{
			closestHit = hit;
		}
	}

	IntersectBVH(r, hit, vbo, debug);

	if (hit.didHit && hit.dst < closestHit.dst)
	{
		closestHit = hit;
	}

	// Returns true if an intersection with the scene occurred, false when no hit
	return closestHit;
}

__device__ float3 refractionRay(const float3 d, const float3 n, float ior)
{
	float cosI = clamp(dot(n, d), -1.0f, 1.0f);

	float eta;
	float3 normal = n;

	if (cosI < 0.0f)
	{
		eta = 1.0f / ior;
		cosI = -cosI;
	}
	else
	{
		eta = ior;
		normal = -normal;
	}

	float k = 1.0f - eta * eta * (1.0f - cosI * cosI);

	if (k < 0.0f)
	{
		return reflect(d, normal);
		//return make_float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		return normalize(d * eta + normal * (eta * cosI - fsqrtf(k)));
	}
}

__device__ float3 radiance(Ray& r, uint32_t& s1, const Scene* scene, size_t bounces, const GPU_Mesh* vbo, int debug, float3* albedoBuf, float3* normalBuf, uint32_t i, Camera_GPU* camera, float* skyTex) // Returns ray color
{
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 accuAlbedo = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 accuNormal = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	bool inVolume = false;
	Material volumeMat;
	float volumeIor = 1.0f;
	float thickness = 0.0f;
	uint16_t surfaceCount = 0;

	for (size_t b = 0; b < bounces; b++)
	{
		// Test ray for intersection with scene
		HitInfo hit = intersect_scene(r, scene, vbo, debug);

		if (inVolume)
		{
			thickness = thickness + length(hit.hitPoint - r.origin);
		}

		if (!hit.didHit)
		{
			//absorption = { 0.0f,0.0f,0.0f };
			accucolor += mask * getEnvironmentLight(r, scene, skyTex);
			if (b <= 0)
			{
				accuAlbedo += getEnvironmentLight(r, scene, skyTex);
				accuNormal += { 0.0f, 0.0f, 1.0f };
			}
			break;
		}

		Material hitMat = scene->materials[hit.materialIndex];

		if (!inVolume)
		{
			volumeIor = 1.0f;
			volumeMat = hitMat;
		}
		//actualIor = (inVolume ? volumeMat.ior / hitMat.ior : hitMat.ior);

		// Create 2 random numbers
		float r1 = 2 * M_PI * randomValue(s1); // Pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
		float r2 = randomValue(s1);            // Pick random number for elevation
		float r2s = sqrtf(r2);

		float3 flippedNormal = (hit.inside ? -hit.normal : hit.normal);

		float ndotl = fmaxf(dot(-r.direction, flippedNormal), 0.0f);
		float F = fresnel(ndotl, 0.0f, hitMat.ior);

		float apparentRoughness = lerp(hitMat.roughness, 0.0f, F);

		bool isSpecularBounce = max(hitMat.metalness, F) >= randomValue(s1);
		bool isTransmissionBounce =  hitMat.transmission >= randomValue(s1);

		float3 diffuseDir = normalize(flippedNormal + randomDirection(s1));
		float3 specularDir = reflect(r.direction, normalize(flippedNormal + randomInUnitSphere(s1) * apparentRoughness));
		
		float3 transmissionDir = refractionRay(r.direction, normalize(hit.normal + randomInUnitSphere(s1) * hitMat.transmissionRoughness), hitMat.ior);

		float3 linearSurfColor = srgbToLinear(hitMat.albedo);
		float3 linearTransmissionColor = srgbToLinear(volumeMat.transmissionColor);

		float transmissionDistance = thickness * volumeMat.transmissionDensity * 10.0f;
		float transmissionDensity = 1.0-expf(-transmissionDistance);

		float3 absorptionColor = powf(linearTransmissionColor, transmissionDensity * (1.0-expf(-volumeMat.transmissionDensity)) * 10.0f);

		//EMISSION
		accucolor += mask * srgbToLinear(hitMat.emission) * hitMat.emissionIntensity;

		//MAIN OUTPUT
		mask = mask * lerp(
						lerp(linearSurfColor, absorptionColor, isTransmissionBounce),
						lerp(make_float3(1.0f), linearSurfColor, hitMat.metalness),
						isSpecularBounce);
		if (b == 0)
		{
			accuAlbedo += linearSurfColor + srgbToLinear(hitMat.emission) * hitMat.emissionIntensity;
			float target[4];
			float4 normalVec = make_float4(flippedNormal, 0.0f);
			vector4_matrix4_mult(&normalVec.x, &camera->viewMat[0], target);

			accuNormal += clamp(make_float3(target[0], target[1], target[2]) * 0.5 + 0.5, make_float3(0.0f), make_float3(1.0f));
		}

		r.origin = hit.hitPoint + flippedNormal * 0.0001f; // offset ray origin slightly to prevent self intersection

		//Entering a surface
		if (isTransmissionBounce && (dot(hit.normal, transmissionDir) < 0.0f))
		{
			volumeMat = hitMat;
			volumeIor = volumeMat.ior;

			r.origin = hit.hitPoint + flippedNormal * -0.0001f; // offset ray origin slightly to prevent self intersection
			surfaceCount = surfaceCount + 1;
		}
		//Exiting a surface
		else if (isTransmissionBounce && (dot(hit.normal, transmissionDir) >= 0.0f))
		{
			volumeIor = 1.0f;
			volumeMat = hitMat;
			r.origin = hit.hitPoint + flippedNormal * -0.0001f; // offset ray origin slightly to prevent self intersection
			surfaceCount = surfaceCount - 1;
		}

		inVolume = (surfaceCount >= 1);

		r.direction = normalize(lerp(lerp(diffuseDir, transmissionDir, isTransmissionBounce), specularDir, isSpecularBounce));

		float p = fmaxf(mask.x, fmaxf(mask.y, mask.z));
		if (randomValue(s1) >= p)
		{
			break;
		}
		mask *= 1.0f / p;

		//accucolor = { albedoBuf[i] };
	}

	//MAIN OUTPUT
	albedoBuf[i] = accuAlbedo;
	normalBuf[i] = normalize(accuNormal);
	return accucolor;

	// 
	//return { float(inVolume),float(inVolume),float(inVolume) };
	//return { float(surfaceCount)*0.25f, float(surfaceCount) * 0.25f, float(surfaceCount) * 0.25f };
}

__global__ void render_kernel(float4* buf, float3* albedoBuf, float3* normalBuf, uint32_t width, uint32_t height, Camera_GPU camera, const Scene* scene, int samples,
							  size_t bounces, uint32_t sampleIndex, const GPU_Mesh* vbo, float* skyTex)
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


	//float4 outputImageTempFloat4 = tex2D<float4>(skyTex, 0.0f, 0.0f);

	float2 coord = { (float)x / (float)width, (float)y / (float)height };
	coord = (coord * 2.0f) - make_float2(1.0f, 1.0f); // -1 -> 1

	float3 finalBeauty, finalNormal, finalAlbedo;

	// Reset r to zero for every pixel
	finalBeauty = make_float3(0.0f);
	finalAlbedo = make_float3(0.0f);
	finalNormal = make_float3(0.0f);

	// Calculate focus point
	float viewPointLocal[4] = { coord.x, coord.y, 1.0f, 1.0f };
	float target[4];
	vector4_matrix4_mult(&viewPointLocal[0], &camera.localToWorldMatrix[0], target);

	float3 viewPoint = make_float3(target[0], target[1], target[2]) * camera.focusDist + camera.pos;
	float3 camRight = make_float3(camera.localToWorldMatrix[0], camera.localToWorldMatrix[1], camera.localToWorldMatrix[2]);
	float3 camUp = make_float3(camera.localToWorldMatrix[4], camera.localToWorldMatrix[5], camera.localToWorldMatrix[6]);

	float2 pixelSize = make_float2(1.00f / (float)width, 1.00f / (float)height);
	float2 aspect = make_float2(1.0f, (float)width / (float)height);

	// Samples per pixel
	for (size_t s = 0; s < 1u; s++)
	{
		// Create primary ray, add incoming radiance to pixelcolor
		Ray ray = Ray(camera.pos, {0.0f, 0.0f, 0.0f});

		//DOF
		//float2 defocusJitter = randomPointInCircle(s1) * camera.aperture; //Even distribution
		//float3 edgeBiasedJitter = randomPointInCircle(s1);
		float2 defocusJitter = randomPointInCircle(s1, ((float)samples) * 0.01f) * camera.aperture; //Edge biased
		ray.origin = camera.pos + camRight * defocusJitter.x * aspect.x + camUp * defocusJitter.y * aspect.y;

		//MSAA
		float2 jitter = randomPointInCircle(s1, 1.0f) * 2.0f;
		float3 jitteredViewPoint = viewPoint + camRight * jitter.x * pixelSize.x + camUp * jitter.y * pixelSize.y;

		ray.direction = normalize(jitteredViewPoint - ray.origin);

		finalBeauty += radiance(ray, s1, scene, bounces, vbo, samples, albedoBuf, normalBuf, i, &camera, skyTex);
		finalAlbedo += albedoBuf[i];
		finalNormal += normalBuf[i];
	}

	// Write rgb value of pixel to image buffer on the GPU
	float scale = (1.0f / ((float)(sampleIndex)));
	float factor = 1.0f-scale;

	buf[i] *= factor;
	buf[i] += make_float4(finalBeauty, 1.0f) * scale;

	albedoBuf[i] *= factor;
	albedoBuf[i] += finalAlbedo * scale;

	normalBuf[i] *= factor;
	normalBuf[i] += finalNormal * scale;
}

__global__ void floatToImageData_kernel(uint32_t* outputBuffer, float4* inputBuffer, uint32_t width, uint32_t height, uint32_t sampleIndex, const Scene* scene)
{
	uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
				 

	if ((x >= width) || (y >= height))
		return;

	// Index of current pixel (calculated using thread index)
	uint32_t i = (height - y - 1) * width + x;

	outputBuffer[i] = ConvertToRGBA(inputBuffer[i]);
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

	if (m_deviceScene != nullptr)
	{
		cudaStatus = cudaMemcpy(m_deviceScene, *m_scene, sizeof(Scene), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy m_deviceScene failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
	}

	Camera_GPU camera_buffer_obj;
	memcpy(&camera_buffer_obj.invProjMat[0],		 m_invProjMat,      sizeof(float) * 16);
	memcpy(&camera_buffer_obj.invViewMat[0],		 m_invViewMat,      sizeof(float) * 16);
	memcpy(&camera_buffer_obj.viewMat[0],			 m_viewMat,         sizeof(float) * 16);
	memcpy(&camera_buffer_obj.localToWorldMatrix[0], m_localToWorldMat, sizeof(float) * 16);
	camera_buffer_obj.pos = m_cameraPos;
	camera_buffer_obj.aperture = m_aperture;
	camera_buffer_obj.focusDist = m_focusDist;

	render_kernel <<<blocks, threads>>> ((float4*)m_floatOutputBuffer_GPU.d_pointer(),
										 (float3*)m_floatAlbedoBuffer_GPU.d_pointer(),
										 (float3*)m_floatNormalBuffer_GPU.d_pointer(),
										  m_width,
										  m_height,
										  camera_buffer_obj,
										  m_deviceScene,
										  *m_samples,
										  *m_bounces,
										  *m_sampleIndex,
										  m_deviceMesh,
										  m_skyTexture);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "render_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaDeviceSynchronize();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching render_kernel!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	m_floatOutputBuffer_GPU.download(m_finalOutputBuffer, m_width * m_height * 4);
	
Error:
	printf("");
}

void CudaRenderer::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_width && height == m_height)
	{
		return;
	}

	m_bufferSize = width * height * sizeof(float4);
	m_width = width;
	m_height = height;

	cudaError_t cudaStatus = cudaErrorStartupFailure;

	cudaStatus = cudaDeviceSynchronize();

	//m_accumulationBuffer_GPU.resize(m_bufferSize);
	m_floatOutputBuffer_GPU.resize(m_bufferSize);
	m_floatAlbedoBuffer_GPU.resize(width * height * sizeof(float3));
	m_floatNormalBuffer_GPU.resize(width * height * sizeof(float3));

	m_finalOutputBuffer = new float[m_bufferSize];
	memset(m_finalOutputBuffer, 0, m_bufferSize);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda Renderer OnResize() failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	Clear();
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

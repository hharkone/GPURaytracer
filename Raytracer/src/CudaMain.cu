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
	float3 invDirection;
	//int32_t inVolumeMat = -1;

	__device__ Ray(float3 o_, float3 d_, float3 invD_ ) : origin(o_), direction(d_), invDirection(invD_) {}
};

struct HitInfo
{
	bool didHit = false;
	bool inside = false;
	float dst = FLT_MAX;
	float3 hitPoint {0.0f, 0.0f, 0.0f};
	float3 normal{ 0.0f, 0.0f, 0.0f };
	float3 geomNormal{ 0.0f, 0.0f, 0.0f };
	float3 color{ 0.0f, 0.0f, 0.0f };
	float3 tangent{ 0.0f, 0.0f, 0.0f };
	float2 uv{ 0.0f, 0.0f };
	size_t materialIndex = 0u;
	uint32_t bvhDepth = 0u;
	//uint16_t nodeID = 0u;
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

	m_floatOutputBuffer_GPU.clear();	//Final float beauty output on the device
	m_floatAlbedoBuffer_GPU.clear();	//Final float albedo output on the device
	m_floatNormalBuffer_GPU.clear();	//Final float normal output on the device
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
	float theta = 2.0f * 3.1415926f * randomValue(state);
	float rho = fsqrtf(-2.0f * log(randomValue(state)));

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

__device__ void vector4_matrix4_mult(float* vec, const float* mat, float* out)
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

__device__ float3 texture2D(const GPUImage* tex, const float2 uv)
{
	size_t texWidth = tex->width;
	size_t texHeight = tex->height;

	float2 fracuv = fracf(uv * make_float2(1.0f, -1.0f));

	size_t row = (size_t)(fracuv.y * (float)texHeight) * 4u;
	size_t col = (size_t)(fracuv.x * (float)texWidth)  * 4u;

	size_t pixelIndex = (row * texWidth) + col;

	/*
	unsigned char* ptr = (unsigned char*)tex->imageData_GPU;
	float x = float(*(ptr + pixelIndex + 0u));
	float y = float(*(ptr + pixelIndex + 1u));
	float z = float(*(ptr + pixelIndex + 2u));

	float3 c = make_float3(x / 255.0f, y / 255.0f, z / 255.0f);
	*/

	//float* ptr = (float*)tex->imageData_GPU;
	//float x = *(ptr + pixelIndex + 0u);
	//float y = *(ptr + pixelIndex + 1u);
	//float z = *(ptr + pixelIndex + 2u);

	unsigned char x = ((unsigned char*)tex->imageData_GPU)[pixelIndex + 0u];
	unsigned char y = ((unsigned char*)tex->imageData_GPU)[pixelIndex + 1u];
	unsigned char z = ((unsigned char*)tex->imageData_GPU)[pixelIndex + 2u];

	float3 c = make_float3(float(x) / 255.0f, float(y) / 255.0f, float(z) / 255.0f);

	return c;
}

__device__ float3 getEnvironmentLight(const Ray& ray, const Scene* scene, const GPUImage* skyTex)
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
			size_t texWidth = skyTex->width;
			size_t texHeight = skyTex->height;

			float2 uv = toSpherical(ray.direction, scene->skyRotation);
			size_t row = (size_t)(uv.y * (float)texHeight) * 4u;
			size_t col = (size_t)(uv.x * (float)texWidth) * 4u;

			size_t pixelIndex = (row * texWidth) + col;

			float* ptr = (float*)skyTex->imageData_GPU;
			float x = *(ptr + pixelIndex + 0u);
			float y = *(ptr + pixelIndex + 1u);
			float z = *(ptr + pixelIndex + 2u);

			float3 c = make_float3(x,y,z);

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
			hit.geomNormal = hit.normal;
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
	float3 geometricNormal = (cross(edgeAB, edgeAC));
	float3 ao = (ray.origin) - tri->pos0;
	float3 dao = cross(ao, ray.direction);

	float determinant = -dot(ray.direction, geometricNormal);
	float invDet = 1.0f / determinant;

	// Calculate dst to triangle & barycentric coordinates of intersection point
	float dst = dot(ao, geometricNormal) * invDet;
	float u = dot(edgeAC, dao) * invDet;
	float v = -dot(edgeAB, dao) * invDet;
	float w = 1.0f - u - v;

	float deltaU1 = tri->uv1.x - tri->uv0.x;
	float deltaV1 = tri->uv1.y - tri->uv0.y;
	float deltaU2 = tri->uv2.x - tri->uv0.x;
	float deltaV2 = tri->uv2.y - tri->uv0.y;

	float f = 1.0f / (deltaU1 * deltaV2 - deltaU2 * deltaV1);

	float3 tangent;// = { edgeAB.x / u1, edgeAB.y / u1, edgeAB.z / u1 };
	tangent.x = f * (deltaV2 * edgeAB.x - deltaV1 * edgeAC.x);
	tangent.y = f * (deltaV2 * edgeAB.y - deltaV1 * edgeAC.y);
	tangent.z = f * (deltaV2 * edgeAB.z - deltaV1 * edgeAC.z);

	float3 normal = normalize(tri->n0 * w + tri->n1 * u + tri->n2 * v);

	if (dot(ray.direction, -normal) < 0.0f)
	{
		normal = normalize(normal - ray.direction * 0.01f); //Bend normals toward the ray that are over-extrapolated.
		//normal = ray.direction * -1.0f;
		//normal = normalize((tri->n0 + tri->n1 + tri->n2) - ray.direction * 0.1f);
	}

	//float overExtrapolation = min(max(dot(-ray.direction, normal), 0.0f) * 50.0f, 1.0f);
	//fn = lerp(fn + -r.direction * 0.1f, fn, debug);
	//fn = normalize(fn);
	//fn = lerp(fn * r.direction, fn, debug);
	//debug = dot(-r.direction, fn) >= 0.0f;

	float3 color = tri->c0  * w + tri->c1  * u + tri->c2  * v;
	float2 uv    = tri->uv0 * w + tri->uv1 * u + tri->uv2 * v;

	// Initialize hit info
	HitInfo hit;
	//hit.didHit = determinant >= 1E-6 && dst >= 0.0f && u >= 0.0f && v >= 0.0f && w >= 0.0f;
	hit.didHit = dst >= 0.0f && u >= 0.0f && v >= 0.0f && w >= 0.0f;
	hit.hitPoint = (ray.origin) + ray.direction * dst;
	hit.normal = normal;
	hit.tangent = normalize(tangent);
	hit.color = color;
	hit.uv = uv;
	hit.geomNormal = normalize(geometricNormal);
	hit.dst = dst;
	hit.inside = (dot(geometricNormal, ray.direction) > 0.0f ? true : false);
	hit.materialIndex = tri->matID;

	return hit;
}

void __device__ IntersectTri(const Ray& ray, HitInfo& hit, const GPU_Mesh::Triangle* tri)
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

/*
__device__ bool rayBoxIntersection(const Ray& r, const float3& min, const float3& max)
{
	float t[9];
	t[1] = (min.x - r.origin.x) / r.direction.x;
	t[2] = (max.x - r.origin.x) / r.direction.x;
	t[3] = (min.y - r.origin.y) / r.direction.y;
	t[4] = (max.y - r.origin.y) / r.direction.y;
	t[5] = (min.z - r.origin.z) / r.direction.z;
	t[6] = (max.z - r.origin.z) / r.direction.z;
	t[7] = fmaxff(fmaxf(fminff(t[1], t[2]), fminff(t[3], t[4])), fminff(t[5], t[6]));
	t[8] = fminff(fminf(fmaxff(t[1], t[2]), fmaxff(t[3], t[4])), fmaxff(t[5], t[6]));
	//t[9] = (t[8] < 0 || t[7] > t[8]) ? FLT_MAX : t[7];

	return (t[8] < 0 || t[7] > t[8]);
}

__device__ bool rayBoxIntersection(const Ray& ray, HitInfo& hit, const float3& bmin, const float3& bmax)
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

	if (ray.origin.x < bmax.x && ray.origin.x > bmin.x &&
		ray.origin.y < bmax.y && ray.origin.y > bmin.y &&
		ray.origin.z < bmax.z && ray.origin.z > bmin.z)
	{
		hit.dst = tmax;
		hit.hitPoint = ray.direction * tmax + ray.origin;
		hit.inside = true;
	}

	float3 p = hit.hitPoint - c;
	float3 d = (bmin - bmax) * 0.5f;

	float bias = 1.0001f;

	hit.normal = normalize( make_float3(float(int(p.x / abs(d.x) * bias)),
										float(int(p.y / abs(d.y) * bias)),
										float(int(p.z / abs(d.z) * bias))) );

	hit.inside = (dot(hit.normal, ray.direction) > 0.0f ? true : false);
	hit.color = make_float3(1.0f, 1.0f, 1.0f);
	hit.materialIndex = 0u;
	//hit.hitPoint = c;

	return didHit;
}

*/


__device__ bool rayBoxIntersection(const Ray& ray, HitInfo& hit, const Box& box)
{
	float3 bmin = (box.pos - (box.size * 0.5f));
	float3 bmax = (box.pos + (box.size * 0.5f));

	float tx1 = (bmin.x - ray.origin.x) * ray.invDirection.x, tx2 = (bmax.x - ray.origin.x) * ray.invDirection.x;
	float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);
	float ty1 = (bmin.y - ray.origin.y) * ray.invDirection.y, ty2 = (bmax.y - ray.origin.y) * ray.invDirection.y;
	tmin = fmaxf(tmin, fminf(ty1, ty2)), tmax = fminf(tmax, fmaxf(ty1, ty2));
	float tz1 = (bmin.z - ray.origin.z) * ray.invDirection.z, tz2 = (bmax.z - ray.origin.z) * ray.invDirection.z;
	tmin = fmaxf(tmin, fminf(tz1, tz2)), tmax = fminf(tmax, fmaxf(tz1, tz2));

	bool didHit (tmax >= tmin && tmin < hit.dst && tmax > 0.0f);

	

	if (didHit)
	{
		hit.didHit = true;
		hit.dst = tmin;
		hit.hitPoint = ray.direction * tmin + ray.origin;

		if (ray.origin.x < ( bmax.x) && ray.origin.x > ( bmin.x) &&
			ray.origin.y < ( bmax.y) && ray.origin.y > ( bmin.y) &&
			ray.origin.z < ( bmax.z) && ray.origin.z > ( bmin.z))
		{
			hit.dst = tmax;
			hit.hitPoint = ray.direction * tmax + ray.origin;
			hit.inside = true;
		}

		float bias = 0.000001f;

		float3 center = (bmin + bmax) * 0.5f;
		float3 centerToPoint = hit.hitPoint - center;
		float3 halfSize = box.size * 0.5f;

		hit.normal = normalize(fsign(centerToPoint) * fstep(-bias, fabs(centerToPoint) - halfSize));
		hit.geomNormal = hit.normal;
		hit.inside = (dot(hit.normal, ray.direction) > 0.0f ? true : false);
		hit.color = {1.0f, 1.0f, 1.0f};
		hit.materialIndex = box.materialIndex;
	}

	return didHit;
}

__device__ float IntersectAABB(const Ray& ray, const HitInfo& hit, const float3 bmin, const float3 bmax)
{
	float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
	float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);
	float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
	tmin = fmaxf(tmin, fminf(ty1, ty2)), tmax = fminf(tmax, fmaxf(ty1, ty2));
	float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
	tmin = fmaxf(tmin, fminf(tz1, tz2)), tmax = fminf(tmax, fmaxf(tz1, tz2));
	if( tmax >= tmin && tmin < hit.dst && tmax > 0) return tmin; else return FLT_MAX;
}

__device__ float IntersectAABB_D(const Ray& ray, const HitInfo& hit, const float3 bmin, const float3 bmax)
{
	float tx1 = (bmin.x - ray.origin.x) * ray.invDirection.x, tx2 = (bmax.x - ray.origin.x) * ray.invDirection.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (bmin.y - ray.origin.y) * ray.invDirection.y, ty2 = (bmax.y - ray.origin.y) * ray.invDirection.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	float tz1 = (bmin.z - ray.origin.z) * ray.invDirection.z, tz2 = (bmax.z - ray.origin.z) * ray.invDirection.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	if (tmax >= tmin && tmin < hit.dst && tmax > 0) return tmin; else return FLT_MAX;
}

__device__ void IntersectBVH(const Ray& ray, HitInfo& hit, const Scene* scene, const RenderSettings* rendererSettings)
{
	uint32_t hitDepth = 0u;
	uint32_t stackPtr = 0u;

	MeshBuffer* meshBuffer = (MeshBuffer*)&scene->sceneMesh.deviceMesh;
	GPU_Mesh::BVHNode* node = (GPU_Mesh::BVHNode*)meshBuffer->bvhNode; 
	GPU_Mesh::BVHNode* stack[32];
	stack[stackPtr++] = 0u;

	HitInfo closestHit;

	while (stackPtr > 0u)
	//while (1)
	{
		if (node->triCount > 0) // isLeaf()
		{
			for (uint32_t i = 0; i < node->triCount; i++)
			{
				uint32_t triIndex = ((uint32_t*)meshBuffer->indexBuffer)[node->leftFirst + i];
				GPU_Mesh::Triangle* triangle = &((GPU_Mesh::Triangle*)meshBuffer->triangleBuffer)[triIndex];

				hit = rayTriangleIntersect(ray, triangle);

				if (hit.didHit && hit.dst < closestHit.dst)
				{
					closestHit = hit;
					closestHit.color = { float(hitDepth) * 10.0f };
				}
			}

			if (stackPtr == 0)
			{
				break;
			}

			else
			{
				node = stack[--stackPtr];
			}

			continue;
		}

		GPU_Mesh::BVHNode* child1 = &((GPU_Mesh::BVHNode*)meshBuffer->bvhNode)[node->leftFirst];
		GPU_Mesh::BVHNode* child2 = &((GPU_Mesh::BVHNode*)meshBuffer->bvhNode)[node->leftFirst + 1];

		float dist1 = IntersectAABB_D(ray, closestHit, child1->aabbMin, child1->aabbMax);
		float dist2 = IntersectAABB_D(ray, closestHit, child2->aabbMin, child2->aabbMax);
		hit.bvhDepth = fminf(dist1, dist2);

		if (dist1 >= dist2)
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
			if (dist2 != FLT_MAX)
			{
				stack[stackPtr++] = child2;
			}
		}

		hitDepth++;

	}// while()
	
	hit = closestHit;
	hit.bvhDepth = hitDepth;
}

__device__ HitInfo intersect_scene(Ray& r, const Scene* scene, const RenderSettings* rendererSettings)
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

	for (size_t i = 0u; i < scene->boxCount; i++)
	{
		Box b = scene->boxSimple[i];
		rayBoxIntersection(r, hit, b);

		if (hit.didHit && hit.dst < closestHit.dst) // If newly computed intersection distance d is smaller than current closest intersection distance
		{
			closestHit = hit;
		}
	}

	IntersectBVH(r, hit, scene, rendererSettings);

	if (hit.didHit && hit.dst < closestHit.dst)
	{
		closestHit = hit;
	}

	closestHit.dst = hit.dst;
	closestHit.bvhDepth = hit.bvhDepth;

	if (!closestHit.didHit)
	{
		closestHit.materialIndex = 0;
	}

	// Returns true if an intersection with the scene occurred, false when no hit
	return closestHit;
}

__device__ float3 refractionRay(const float3 d, const float3 n, float ior, bool& totalInternalReflection)
{
	float cosI = clamp(dot(n, d), -1.0f, 1.0f);

	float eta;
	float3 normal = n;

	if (ior == 1.0f)
	{
		totalInternalReflection = false;
		return d;
	}

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
		totalInternalReflection = true;
		return reflect(d, normal);
	}
	else
	{
		totalInternalReflection = false;
		return normalize(d * eta + normal * (eta * cosI - fsqrtf(k)));
	}
}

__device__ float3 spectrum(float x)
{
	//x = clamp(x, 0.0f, 1.0f);
	const float3 cs = make_float3(3.54541723f, 2.86670055f, 2.29421995f);
	const float3 xs = make_float3(0.69548916f, 0.49416934f, 0.28269708f);
	const float3 ys = make_float3(0.02320775f, 0.15936245f, 0.53520021f);

	float3 cs2 = (cs * (make_float3(x) - xs));

	float3 y = make_float3(1.0f, 1.0f, 1.0f) - cs2 * cs2;
	y = clamp((y - ys), 0.0f, 1.0f);

	return y * make_float3(2.0f, 2.0f, 20.0f);
}

__device__ float3 bump3(float3 x)
{
	float3 y = make_float3(1.0f) - x * x;
		   y = cfmaxf(y, make_float3(0.0f));
	return y;
}

__device__ float3 spectral_gems(float w)
{
	float x = clamp(w, 0.0f, 1.0f);

	return bump3(make_float3(4.0f * (x - 0.75f), // Red
						     4.0f * (x - 0.50f), // Green
							 4.0f * (x - 0.25f)) // Blue
	);
}

__device__ float3 hsv2rgb(float3 c)
{
	float4 K = make_float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
	float3 p = fabs(fracf(make_float3(c.x) + make_float3(K)) * 6.0f - make_float3(K.w));
	return c.z * lerp(make_float3(K.x), clamp(p - make_float3(K.x), 0.0f, 1.0f), c.y);
}

__device__ float3 rgb2hsv(float3 c)
{
	float4 K = make_float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
	float4 p = lerp(make_float4(c.z, c.y, K.w, K.z), make_float4(c.y, c.z, K.x, K.y), fstep(c.z, c.y));
	float4 q = lerp(make_float4(p.x, p.y, p.w, c.x), make_float4(c.x, p.y, p.z, p.x), fstep(p.x, c.x));

	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return make_float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

__device__ float3 radiance(Ray& r, uint32_t s1, uint32_t& s2, const Scene* scene, const RenderSettings* rendererSettings, float3& albedoOut, float3& normalOut, uint32_t i, const Camera_GPU* camera, const GPUImage* skyTex, const GPUImage* debugTex0, const GPUImage* debugTex1, const GPUImage* debugTex2) // Returns ray color
{
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 accuAlbedo = make_float3(0.0f, 0.0f, 0.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 accuNormal = make_float3(0.0f, 0.0f, 1.0f); // Accumulates ray colour with each iteration through bounce loop
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	bool totalInternalReflection = false;

	Material hitMat = ((Material*)scene->materialBufferPtr)[0];
	Material volumeMat = ((Material*)scene->materialBufferPtr)[0];

	float thickness = 0.0f;
	uint16_t transmissionCount = 0u;
	uint32_t s = 2345u;
	float bvhDepth = 0.0f;
	uint16_t matIndexMap[20u] = { 0u };
	float3 debug = make_float3(0.0f, 0.0f, 0.0f);
	HitInfo hit;
	HitInfo previousHit;

	float rt = randomValue(s1);

	//float3 spectralNormalization = make_float3(2.428571428f, 2.318181818f, 2.318181818f);
	//float3 dispersionColor = srgbToLinear(spectral_gems(rt)) * spectralNormalization;

	//dispersionColor = make_float3(rt < 0.333f ? 1.0f : 0.0f, (rt >= 0.333f && rt < 0.666f) ? 1.0f : 0.0f, (rt >= 0.666f) ? 1.0f : 0.0f) * 3.0f;

#if true
	for (size_t b = 0; b < rendererSettings->bounces; b++)
	{
		// Test ray for intersection with scene
		hit = intersect_scene(r, scene, rendererSettings);

		if (rendererSettings->bvhDebug)
		{
			bvhDepth = float(hit.bvhDepth) * 0.01f;
			//return { dispersionColor };
			return { srgbToLinear(make_float3(bvhDepth)) };
			break;
		}

		//volumeAbsorptionColor = ((transmissionCount > 0u) ? volumeAbsorptionColor : make_float3(1.0f));
		//volumeAbsorptionColor = make_float3(volumeTransmissionDensity);

		//mask = mask * volumeAbsorptionColor;

		if (!hit.didHit)
		{
			float bgMask = 1.0f;

			if (b <= 0u)
			{
				bgMask = scene->backgroundBrightness;
				accuAlbedo = mask * getEnvironmentLight(r, scene, skyTex) * bgMask;
			}

			accucolor += mask * getEnvironmentLight(r, scene, skyTex) * bgMask;

			break;
		}
		//float3 fn = (hit.inside ? -hit.normal : hit.normal);
		//float debug = dot(-r.direction, fn) > 0.0f;
		//accucolor = make_float3(debug);
		//accucolor = fn;
		//break;
		//hitMat = ((Material*)scene->materialBufferPtr)[hit.materialIndex];
		hitMat = ((Material*)scene->materialBufferPtr)[hit.materialIndex];

		if (hit.inside && transmissionCount <= 0u)
		{
			transmissionCount++;
			matIndexMap[clamp(transmissionCount, (uint16_t)0u, (uint16_t)19u)] = hit.materialIndex;
		}

		uint16_t prevMatIndex = clamp(transmissionCount, (uint16_t)0u, (uint16_t)19u);
		volumeMat = ((Material*)scene->materialBufferPtr)[matIndexMap[prevMatIndex]];

		float iorDiffF = fmaxf(fmaxf(hitMat.ior, volumeMat.ior) / fminf(hitMat.ior, volumeMat.ior), 1.0f);
		float iorDiff = hitMat.ior / volumeMat.ior;// / volumeMat.ior;


		if (hit.inside && transmissionCount <= 1u) //Interface to air
		{
			iorDiffF = hitMat.ior;
			iorDiff = hitMat.ior;//1.0f / hitMat.ior;
		}

		//if (r.inVolumeMat != -1)
		//{
			
			
			//volumeMat = scene->materials[r.inVolumeMat];
			//iorDiff = fmaxf(hitMat.ior / volumeMat.ior, 1.0f);

			//if (hit.inside && transmissionCount <= 1u) //Interface to air
			//{
				//iorDiff = hitMat.ior;
			//}

			//disperseIor = iorDiff + (1.0f - rt) * hitMat.transmissionAberration * (iorDiff - 1.0f);
		//}
		//else
		//{
		//	volumeMat = scene->air;
		//}

		float3 albedo = srgbToLinear(hitMat.albedo);
		float rough = hitMat.roughness;
		float metal = hitMat.metalness;
		float3 N = hit.normal;

		/*
		if (hit.materialIndex == 1u && debugTex0 != nullptr)
		{
			float2 uv = hit.uv;
			float3 tex0 = srgbToLinear(texture2D(debugTex0, uv));
			float3 tex1 = texture2D(debugTex1, uv);
			float3 tex2 = texture2D(debugTex2, uv) * 2.0f - 1.0f;

			albedo = hitMat.albedo * tex0;
			rough = (hitMat.roughness * hitMat.roughness) * (1.0f-tex1.y);
			metal = hitMat.metalness * tex1.z;
			float3 bT = normalize(cross(hit.normal, hit.tangent));
			N = normalize(hit.tangent) * tex2.x + normalize(bT) * tex2.y + normalize(hit.normal) * tex2.z;
			N = normalize(lerp(hit.normal, N, hitMat.transmissionInscatter));
		}
		*/

		float3 flippedNormal = (hit.inside ? -N : N);
		float3 flippedGeometricNormal = (hit.inside ? -hit.geomNormal : hit.geomNormal);

		//float disperseIorDiff = fmaxf(disperseIor / 1.0f, 1.0f);

		// Create 2 random numbers
		float r1 = 2 * M_PI * randomValue(s1); // Pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
		float r2 = randomValue(s1);            // Pick random number for elevation
		float r2s = sqrtf(r2);

		float ndotv = fmaxf(dot(-r.direction, flippedNormal), 0.0f);
		float F = fresnel(ndotv, 0.0f, iorDiffF);
		float F82 = clamp(fresnel(ndotv, 0.0f, 1.5f) * 3.0f - 1.0f, 0.0f, 1.0f); //Hacking desaturated edges for metals

		float3 transmissionDir = refractionRay(normalize(r.direction + randomInUnitSphere(s1) * hitMat.transmissionRoughness * hitMat.transmissionRoughness), N, iorDiff, totalInternalReflection);
		float3 inscatterDir = normalize(randomInUnitSphere(s1));

		float apparentRoughness = lerp(lerp(rough, 0.0f, F), hitMat.transmissionRoughness, totalInternalReflection);

		bool isSpecularBounce = fmaxf(metal, F) >= randomValue(s1);
			 isSpecularBounce = (isSpecularBounce || totalInternalReflection);

		bool isTransmissionBounce = (hitMat.transmission * (float)!isSpecularBounce) > randomValue(s1);
			 isTransmissionBounce *= !totalInternalReflection;
		bool isInscatterBounce = (hitMat.transmissionInscatter * isTransmissionBounce) > randomValue(s1);

		thickness = length(hit.hitPoint - r.origin);
		//float3 linearTransmissionColor = srgbToLinear(hitMat.transmissionColor);
		//float transmissionDistance = thickness * hitMat.transmissionDensity * 10.0f;
		//float transmissionDensity = 1.0 - expf(-transmissionDistance);
		float volumeTransmissionDistance = thickness * volumeMat.transmissionDensity * 10.0f;
		//float volumeTransmissionDensity = 1.0 - expf(-volumeTransmissionDistance);

		//float3 absorptionColor = powf(linearTransmissionColor, transmissionDensity * (1.0-expf(-hitMat.transmissionDensity)) * 10.0f);
		//absorptionColor = (inVolume ? absorptionColor : make_float3(1.0f));

		float3 linearVolumeTransmissionColor = hsv2rgb(rgb2hsv(make_float3(1.0f) - srgbToLinear(volumeMat.transmissionColor)) + make_float3(0.5f, 0.0f, 0.0f));
		float3 volumeAbsorptionColor = lerp(make_float3(1.0f), powf(linearVolumeTransmissionColor, volumeTransmissionDistance), (transmissionCount >= 1u));

		float3 diffuseDir = normalize(flippedNormal + randomDirection(s1));
		float3 specularDir = reflect(r.direction, normalize(flippedNormal + randomInUnitSphere(s1) * apparentRoughness));

		//float chromaticAberration = fmaxff(hitMat.ior + (aberration * 2.0f - 1.0f) * hitMat.transmissionAberration * (hitMat.ior-1.0f), 1.0f);
		//float3 cromaticColor = spectrum(1.0f-aberration) * make_float3(0.5f, 0.5f, 0.098f) * 2.83067f;
		//float3 cromaticColor = hsv2rgb(make_float3(1.0f-aberration, 0.5f, 1.0f)) * 2.93067f;

		float3 linearVertexColor = srgbToLinear(lerp(make_float3(1.0f, 1.0f, 1.0f), hit.color, hitMat.vcolor));
		float3 linearSurfColor = albedo * linearVertexColor;

		//EMISSION
		accucolor += mask * srgbToLinear(hitMat.emission) * hitMat.emissionIntensity * volumeAbsorptionColor;

		//MAIN OUTPUT
		//mask = mask * lerp(
		//				lerp(linearSurfColor, absorptionColor, isTransmissionBounce),
		//				lerp(make_float3(1.0f), lerp(linearSurfColor, make_float3(1.0f), F82), metal),
		//				isSpecularBounce) * volumeAbsorptionColor;

		//MAIN OUTPUT
		float3 specularColor = lerp(linearSurfColor, make_float3(1.0f), F82);
			   specularColor = lerp(make_float3(1.0f), specularColor, metal);
		float3 maskColor = lerp(lerp(linearSurfColor, specularColor, isSpecularBounce), make_float3(1.0f), isTransmissionBounce);
			   maskColor = maskColor * volumeAbsorptionColor;

		mask = mask * maskColor;

		/*
		mask = mask * lerp(
						lerp(linearSurfColor,
							lerp(make_float3(1.0f),
								lerp(linearSurfColor,
									make_float3(1.0f),
									0.0f),
								metal),
							isSpecularBounce),
						make_float3(1.0f),
					isTransmissionBounce);

		*/
		if (b <= 0)
		{
			float target[4];
			float4 normalVec = make_float4(flippedNormal, 0.0f);
			vector4_matrix4_mult(&normalVec.x, &camera->viewMat[0], target);

			accuNormal = clamp(make_float3(target[0], target[1], target[2]) * 0.5f + 0.5f, make_float3(-1.0f), make_float3(1.0f));
			accuAlbedo = mask * lerp(linearSurfColor, srgbToLinear(hitMat.transmissionColor), hitMat.transmission);
			accuAlbedo += mask * srgbToLinear(hitMat.emission) * hitMat.emissionIntensity;
		}

		float p = fmaxf(mask.x, fmaxf(mask.y, mask.z));
		if (randomValue(s1) >= p)
		{
			break;
		}

		mask *= 1.0f / p;

		r.origin = hit.hitPoint + flippedGeometricNormal * 0.0001f;
		r.direction = normalize(lerp(diffuseDir, specularDir, isSpecularBounce));

		if (isTransmissionBounce)
		{
			// Entering a surface
			if (dot(hit.geomNormal, transmissionDir) < 0.0f)
			{
				transmissionCount++;
				transmissionCount = min(transmissionCount, 19u);
				matIndexMap[transmissionCount] = hit.materialIndex;
				//mask = mask * volumeAbsorptionColor;

				r.origin = hit.hitPoint + flippedGeometricNormal * -0.0001f;
				r.direction = normalize(transmissionDir);
				/*
				if (transmissionCount > 0u)
				{
					float inscatterLength = (isInscatterBounce ? randomValue(s1) : 0.0f);
					r.origin = lerp(r.origin, hit.hitPoint, inscatterLength);
					r.direction = normalize(r.direction + normalize(inscatterDir + r.direction * hitMat.transmissionInscatterAnisotropy) * hitMat.transmissionInscatter * isInscatterBounce);
				}
				*/
			}
			//Exiting a surface
			else if(dot(hit.geomNormal, transmissionDir) >= 0.0f)
			{
				transmissionCount--;
				transmissionCount = max(transmissionCount, 0u);
				//mask = mask * volumeAbsorptionColor;
				//matIndexMap[transmissionCount] = 0;
				r.origin = hit.hitPoint + flippedGeometricNormal * -0.001f; // offset ray origin slightly to prevent self intersection
				r.direction = normalize(transmissionDir);
			}

			//r.inVolumeMat = ((transmissionCount <= 0u) ? -1 : matIndexMap[transmissionCount]);
		}

		r.invDirection = make_float3(1.0f, 1.0f, 1.0f) / r.direction;
	}
#endif

	//MAIN OUTPUT
	albedoOut = accuAlbedo;
	normalOut = normalize(accuNormal);

	return accucolor;// *dispersionColor;

}

__global__ void render_kernel(float4* buf, float3* albedoBuf, float3* normalBuf, uint32_t width, uint32_t height, const Camera_GPU camera, const Scene* scene,
							   const RenderSettings* rendererSettings, uint32_t sampleIndex, const GPUImage skyTex, const GPUImage debugTex0, const GPUImage debugTex1, const GPUImage debugTex2)
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
	uint32_t s2 = i;

	float2 coord = { (float)x / (float)width, (float)y / (float)height };
	coord = (coord * 2.0f) - make_float2(1.0f, 1.0f); // -1 -> 1

	float3 finalBeauty, finalNormal, finalAlbedo;

	// Reset r to zero for every pixel
	finalBeauty = make_float3(0.0f);
	finalAlbedo = make_float3(0.0f);
	finalNormal = make_float3(0.0f);

	float2 aspect = make_float2(1.0f, (float)width / (float)height);
	float2 pixelSize = make_float2(1.00f / (float)width, 1.00f / (float)height);

	float3 camRight = make_float3(camera.localToWorldMatrix[0], camera.localToWorldMatrix[1], camera.localToWorldMatrix[2]);
	float3 camUp = make_float3(camera.localToWorldMatrix[4], camera.localToWorldMatrix[5], camera.localToWorldMatrix[6]);

	//DOF
	float2 defocusJitter = randomPointInCircle(s1, 0.8f) * camera.aperture; //Edge biased
	//MSAA
	float2 jitter = make_float2(randomValue(s1) - 0.5f, randomValue(s1) - 0.5f) * pixelSize * 2.0f;

	// Calculate focus point
	float viewPointLocal[4] = { coord.x, coord.y, 1.0f, 1.0f };
	viewPointLocal[0] += jitter.x;
	viewPointLocal[1] += jitter.y;

	float viewPointWorld[4];

	vector4_matrix4_mult(&viewPointLocal[0], &camera.localToWorldMatrix[0], viewPointWorld);

	//float3 viewPoint = make_float3(viewPointWorld[0], viewPointWorld[1], viewPointWorld[2]);
	float3 viewPoint = make_float3(viewPointWorld[0], viewPointWorld[1], viewPointWorld[2]) * camera.focusDist + camera.pos;

	//viewPoint = viewPoint + camRight * jitter.x + camUp * jitter.y;
	//viewPoint = viewPoint + camRight * jitter.x + camUp * jitter.y;

	float3 cameraPos = camera.pos + camRight * defocusJitter.x * aspect.x + camUp * defocusJitter.y * aspect.y;

	// Create primary ray, add incoming radiance to pixelcolor
	Ray ray = Ray(cameraPos, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });

	//ray.origin = cameraPos + camRight * defocusJitter.x * aspect.x + camUp * defocusJitter.y * aspect.y;

	ray.direction = normalize(viewPoint - cameraPos);
	ray.invDirection = make_float3(1.0f, 1.0f, 1.0f) / ray.direction;

	finalBeauty += radiance(ray, s1, s2, scene, rendererSettings, finalAlbedo, finalNormal, i, &camera, &skyTex, &debugTex0, &debugTex1, &debugTex2);

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

	if (m_scene == NULL || m_rendererSettings == NULL)
	{
		return;
	}

	m_deviceScene.upload(m_scene, 1u);
	m_deviceSettings.upload(m_rendererSettings, 1u);

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
										  (Scene*)m_deviceScene.d_pointer(),
										  (RenderSettings*)m_deviceSettings.d_pointer(),
										  *m_sampleIndex,
										  m_imgLoaderEnv.gpuImage,
										  m_imgLoaderTestTexture0.gpuImage,
										  m_imgLoaderTestTexture1.gpuImage,
										  m_imgLoaderTestTexture2.gpuImage);


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

void CudaRenderer::SetHDRI(std::string path)
{
	cudaDeviceSynchronize();
	m_imgLoaderEnv.LoadImage_EXR(path);
}

void CudaRenderer::SetScene(const Scene* scene)
{
	m_scene = scene;
	m_deviceScene.upload(scene, 1u);
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

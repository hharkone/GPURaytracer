#include <execution>
#include "Renderer.h"
#include "Walnut/Random.h"

namespace Utils
{
	static uint32_t ConvertToRGBA(const glm::vec4& color)
	{
		uint8_t r = (uint8_t)(color.r * 255.0f);
		uint8_t g = (uint8_t)(color.g * 255.0f);
		uint8_t b = (uint8_t)(color.b * 255.0f);
		uint8_t a = (uint8_t)(color.a * 255.0f);

		uint32_t returnValue = (a << 24) | (b << 16) | (g << 8 ) | r;

		return returnValue;
	}

	static uint32_t PCG_Hash(uint32_t input)
	{
		uint32_t state = input * 747796405u + 2891336453u;
		uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		return (word >> 22u) ^ word;
	}

	static float RandmFloat(uint32_t& seed)
	{
		seed = PCG_Hash(seed);
		return (float)seed / (float)std::numeric_limits<uint32_t>::max();
	}

	static glm::vec3 InUnitSphere(uint32_t& seed)
	{
		return glm::normalize(glm::vec3(RandmFloat(seed) * 2.0f - 1.0f, RandmFloat(seed) * 2.0f - 1.0f, RandmFloat(seed) * 2.0f - 1.0f));
	}
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_finalImage)
	{
		if (m_finalImage->GetWidth() == width && m_finalImage->GetHeight() == height)
			return;

		m_finalImage->Resize(width, height);
	}
	else
	{
		m_finalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}

	delete[] m_imageData;
	m_imageData = new uint32_t[width * height];

	delete[] m_accumulationData;
	m_accumulationData = new glm::vec4[width * height];

	m_imgHorizontalIterator.resize(width);
	m_imgVerticalIterator.resize(height);

	for (uint32_t i = 0; i < width; i++)
		m_imgHorizontalIterator[i] = i;

	for (uint32_t i = 0; i < height; i++)
		m_imgVerticalIterator[i] = i;
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_activeScene = &scene;
	m_activeCamera = &camera;

	if (m_frameIndex == 1)
		memset(m_accumulationData, 0, m_finalImage->GetWidth() * m_finalImage->GetHeight() * sizeof(glm::vec4));


#if 1
	std::for_each(std::execution::par, m_imgVerticalIterator.begin(), m_imgVerticalIterator.end(),
		[this](uint32_t y)
		{
			std::for_each(std::execution::par, m_imgHorizontalIterator.begin(), m_imgHorizontalIterator.end(),
				[this, y](uint32_t x)
				{
					glm::vec4 color = PerPixel(x, y);
					m_accumulationData[x + y * m_finalImage->GetWidth()] += color;

					glm::vec4 accumulatedColor = m_accumulationData[x + y * m_finalImage->GetWidth()];
					accumulatedColor /= (float)m_frameIndex;

					accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0));
					//accumulatedColor = glm::pow(accumulatedColor, glm::vec4(0.46464f));
					m_imageData[x + y * m_finalImage->GetWidth()] = Utils::ConvertToRGBA(accumulatedColor);
				});
		});
#else
	std::for_each(m_imgVerticalIterator.begin(), m_imgVerticalIterator.end(),
		[this](uint32_t y)
		{
			std::for_each( m_imgHorizontalIterator.begin(), m_imgHorizontalIterator.end(),
			[this, y](uint32_t x)
				{
					glm::vec4 color = PerPixel(x, y);
					m_accumulationData[x + y * m_finalImage->GetWidth()] += color;

					glm::vec4 accumulatedColor = m_accumulationData[x + y * m_finalImage->GetWidth()];
					accumulatedColor /= (float)m_frameIndex;

					accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0));
					m_imageData[x + y * m_finalImage->GetWidth()] = Utils::ConvertToRGBA(accumulatedColor);
				});
		});
#endif

	m_finalImage->SetData(m_imageData);

	if (m_settings.accumulate)
	{
		m_frameIndex++;
	}
	else
	{
		m_frameIndex = 1;
	}
}

Renderer::Hit Renderer::rayTriangleIntersect(const Ray& ray, const Mesh::Triangle& tri, const glm::vec3& origin)
{
	Hit hit;

	// compute the plane's normal
	glm::vec3 v0v1 = tri.v1.pos - tri.v0.pos;
	glm::vec3 v0v2 = tri.v2.pos - tri.v0.pos;
	glm::vec3 N = glm::cross(v0v1, v0v2);
	//float area2 = N.length();
	//N = glm::normalize(N);

	float hitDistance = std::numeric_limits<float>::max();

	// Step 1: finding P
	// check if the ray and plane are parallel.
	float NdotRayDirection = glm::dot(N, ray.direction);
	if (fabs(NdotRayDirection) < FLT_EPSILON) // almost 0
		return Miss(ray, true); // they are parallel, so they don't intersect! 

	// compute d parameter using equation 2
	float d = glm::dot(-N, tri.v0.pos);

	// compute t (equation 3)
	hitDistance = -(glm::dot(N, origin) + d) / NdotRayDirection;

	// check if the triangle is behind the ray
	if (hitDistance < 0) return Miss(ray, true); // the triangle is behind

	// compute the intersection point using equation 1
	glm::vec3 P = origin + ray.direction * hitDistance;

	// Step 2: inside-outside test
	glm::vec3 C; // vector perpendicular to triangle's plane

	// edge 0
	glm::vec3 edge0 = tri.v1.pos - tri.v0.pos;
	glm::vec3 vp0 = P - tri.v0.pos;
	C = glm::cross(edge0, vp0);
	if (glm::dot(N, C) < 0) return Miss(ray, true); // P is on the right side

	// edge 1
	glm::vec3 edge1 = tri.v2.pos - tri.v1.pos;
	glm::vec3 vp1 = P - tri.v1.pos;
	C = glm::cross(edge1, vp1);
	if (glm::dot(N, C) < 0)  return Miss(ray, true); // P is on the right side

	// edge 2
	glm::vec3 edge2 = tri.v0.pos - tri.v2.pos;
	glm::vec3 vp2 = P - tri.v2.pos;
	C = glm::cross(edge2, vp2);
	if (glm::dot(N, C) < 0) return Miss(ray, true); // P is on the right side;

	//hit.worldNormal = glm::vec3(0.0, 0.0, 1.0);
	//hit.worldPosition = P;
	hit.hitDistance = hitDistance;

	return hit; // this ray hits the triangle
}

Renderer::Hit Renderer::TraceRay(const Ray& ray)
{
	int closestTriIndex = -1;
	float closestHit = std::numeric_limits<float>::max();
	int closestObjectIndex = -1;

	Hit hit;

	//spheres
	/*
	for (size_t i = 0u; i < m_activeScene->spheres.size(); i++)
	{
		const Sphere& sphere = m_activeScene->spheres[i];

		glm::vec3 origin = ray.origin - sphere.position;

		float a = glm::dot(ray.direction, ray.direction);
		float b = 2.0f * glm::dot(origin, ray.direction);
		float c = glm::dot(origin, origin) - sphere.radius * sphere.radius;

		float discriminant = b * b - 4.0f * a * c;

		if (discriminant < 0.0f)
			continue;

		float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a);
		float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);

		if (closestT > 0.0f && closestT < hitDistance)
		{
			hitDistance = closestT;
			closestSphere = (int)i;
		}
	}

	if (closestSphere < 0)
		return  Miss(ray);

	return ClosestHit(ray, hitDistance, closestSphere);
	*/

	//triangles
	for (size_t i = 0u; i < m_activeScene->meshes.size(); i++)
	{
		const Mesh::MeshData& mesh = m_activeScene->meshes[i];
		glm::vec3 origin = ray.origin - mesh.Transform;

		for (size_t t = 0u; t < mesh.tris.size(); t++)
		{
			hit = rayTriangleIntersect(ray, mesh.tris[t], origin);

			if (hit.hitDistance > 0.0f && hit.hitDistance < closestHit)
			{
				closestHit = hit.hitDistance;
				closestObjectIndex = (int)i;
				closestTriIndex = (int)t;
				hit.worldPosition + mesh.Transform;
			}
		}
	}

	if (closestTriIndex < 0 && closestObjectIndex)
		return  Miss(ray, false);

	return ClosestHit(ray, closestHit, closestObjectIndex, closestTriIndex);

	//return hit;
}

glm::vec4 Renderer::PerPixel(uint32_t x, uint32_t y)
{
	Ray ray;
	ray.origin = m_activeCamera->GetPosition();
	ray.direction = m_activeCamera->GetRayDirections()[x + y * m_finalImage->GetWidth()];

	glm::vec3 light(0.0f);
	glm::vec3 contribution(1.0);
	glm::vec3 previousContribution(1.0);

	uint32_t seed = x + y * m_finalImage->GetWidth();
	seed *= m_frameIndex;

	int bounces = 5;
	for (int i = 0; i < bounces; i++)
	{
		seed += i;
		Renderer::Hit hit = TraceRay(ray);

		if (hit.hitDistance < 0.0f)
		{
			light += m_activeScene->m_skyColor * m_activeScene->m_skyBrightness * contribution;
			break;
		}

		const Mesh::MeshData& mesh = m_activeScene->meshes[hit.objectIndex];
		const Mesh::Triangle& tri = mesh.tris[hit.primIndex];
		const Material& mat = m_activeScene->materials[mesh.materialIndex];

		glm::vec3 nvec = glm::normalize(glm::cross(tri.v1.pos - tri.v0.pos, tri.v2.pos - tri.v0.pos));

		contribution *= mat.albedo * previousContribution;
		light += mat.GetEmission() * previousContribution;
		//light = hit.worldPosition;

		ray.origin = hit.worldPosition + hit.worldNormal * 0.0001f;
		glm::vec3 randomDirection = nvec + Utils::InUnitSphere(seed);
		randomDirection = glm::normalize(randomDirection);

		ray.direction = randomDirection;
		previousContribution *= mat.albedo;
	}

	return glm::vec4(light, 1.0f);
}

Renderer::Hit Renderer::ClosestHit(const Ray& ray, float hitDistance, int objectIndex, int triIndex)
{
	Renderer::Hit hit;
	hit.hitDistance = hitDistance;
	hit.objectIndex = objectIndex;
	hit.primIndex = triIndex;

	const Mesh::MeshData& mesh     = m_activeScene->meshes[objectIndex];
	const Mesh::Triangle& triangle = mesh.tris[triIndex];

	glm::vec3 origin = ray.origin - mesh.Transform;
	hit.worldPosition = origin + ray.direction * hitDistance;
	hit.worldNormal = glm::normalize(triangle.v0.normal);
	hit.worldPosition += mesh.Transform;

	return hit;
}

Renderer::Hit Renderer::ClosestHit(const Ray& ray, float hitDistance, int objectIndex)
{
	Renderer::Hit hit;
	hit.hitDistance = hitDistance;
	hit.objectIndex = objectIndex;

	const Sphere& closestSphere = m_activeScene->spheres[objectIndex];

	glm::vec3 origin = ray.origin - closestSphere.position;
	hit.worldPosition = origin + ray.direction * hitDistance;
	hit.worldNormal = glm::normalize(hit.worldPosition);

	hit.worldPosition += closestSphere.position;

	return hit;
}

Renderer::Hit Renderer::Miss(const Ray& ray, const bool missDistMax)
{
	Renderer::Hit hit;
	if (missDistMax) { hit.hitDistance = std::numeric_limits<float>::max(); }
	else { hit.hitDistance = -1.0f; }
	hit.objectIndex = -1;
	hit.primIndex = -1;

	return hit;
}
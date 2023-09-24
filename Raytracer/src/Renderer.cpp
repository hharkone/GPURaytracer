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

	static float RandomFloat(uint32_t& seed)
	{
		seed = PCG_Hash(seed);
		return (float)seed / (float)std::numeric_limits<uint32_t>::max();
	}

	static glm::vec3 InUnitSphere(uint32_t& seed)
	{
		return glm::normalize(glm::vec3(RandomFloat(seed) * 2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f));
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


#if true
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

bool RayBoundingBox(const Ray& ray, const Mesh& mesh)
{
	glm::vec3 boxMin = mesh.bbox.min + mesh.Transform;
	glm::vec3 boxMax = mesh.bbox.max + mesh.Transform;

	glm::vec3 invDir = 1.0f / ray.direction;
	glm::vec3 tMin = (boxMin - ray.origin) * invDir;
	glm::vec3 tMax = (boxMax - ray.origin) * invDir;
	glm::vec3 t1 = glm::min(tMin, tMax);
	glm::vec3 t2 = glm::max(tMin, tMax);

	float tNear = glm::max(glm::max(t1.x, t1.y), t1.z);
	float tFar = glm::min(glm::min(t2.x, t2.y), t2.z);

	return tNear <= tFar;
};

Renderer::Hit Renderer::rayTriangleIntersect(const Ray& ray, const Mesh::Triangle& tri, const glm::vec3& origin)
{
	glm::vec3 edgeAB = tri.v1.pos - tri.v0.pos;
	glm::vec3 edgeAC = tri.v2.pos - tri.v0.pos;
	glm::vec3 normalVector = glm::cross(edgeAB, edgeAC);
	glm::vec3 ao = (ray.origin - origin) - tri.v0.pos;
	glm::vec3 dao = glm::cross(ao, ray.direction);

	float determinant = -glm::dot(ray.direction, normalVector);
	float invDet = 1.0f / determinant;

	// Calculate dst to triangle & barycentric coordinates of intersection point
	float dst = glm::dot(ao, normalVector) * invDet;
	float u = glm::dot(edgeAC, dao) * invDet;
	float v = -glm::dot(edgeAB, dao) * invDet;
	float w = 1.0f - u - v;

	// Initialize hit info
	Hit hitInfo;
	hitInfo.didHit = determinant >= 1E-6 && dst >= 0.0f && u >= 0.0f && v >= 0.0f && w >= 0.0f;
	hitInfo.worldPosition = (ray.origin) + ray.direction * dst;
	hitInfo.worldNormal = normalize(tri.v0.normal * w + tri.v1.normal * u + tri.v2.normal * v);
	hitInfo.hitDistance = dst;

	return hitInfo;
}

Renderer::Hit Renderer::CalculateRayCollision(const Ray& ray)
{
	Hit closestHit;

	//Meshes
	for (size_t i = 0u; i < m_activeScene->meshes.size(); i++)
	{
		const Mesh& mesh = m_activeScene->meshes[i];
		glm::vec3 origin = mesh.Transform;


		if (!RayBoundingBox(ray, mesh))
		{
			closestHit.debugHit = 1;
			continue;
		}

		for (size_t t = 0u; t < mesh.tris.size(); t++)
		{
			Hit hit = rayTriangleIntersect(ray, mesh.tris[t], origin);

			if (hit.didHit && hit.hitDistance < closestHit.hitDistance)
			{
				closestHit = hit;
				closestHit.objectIndex = (int)i;
				closestHit.primIndex = (int)t;
				closestHit.worldPosition + mesh.Transform;
				closestHit.materialIndex = mesh.materialIndex;
			}
		}
	}

	return closestHit;
}

glm::vec3 Renderer::GetEnvironmentLight(Ray& ray)
{
	glm::vec3 sunDir = glm::vec3(1.0f);
	sunDir = glm::normalize(sunDir);

	float skyGradientT = glm::pow(glm::smoothstep(0.0f, 0.4f, ray.direction.y), 0.35f);
	float groundToSkyT = glm::smoothstep(-0.01f, 0.0f, ray.direction.y);
	glm::vec3 skyGradient = glm::mix(m_activeScene->m_skyColorHorizon, m_activeScene->m_skyColorZenith, skyGradientT);
	float sun = glm::pow(glm::max(0.0f, glm::dot(ray.direction, sunDir)), m_activeScene->m_sunFocus) * m_activeScene->m_sunIntensity;
	// Combine ground, sky, and sun
	glm::vec3 composite = glm::mix(m_activeScene->m_groundColor, skyGradient, groundToSkyT) + sun * (groundToSkyT >= 1);

	return composite;
}

float fresnel_schlick_ratio(float cos_theta_incident, float power)
{
	float p = 1.f - cos_theta_incident;
	return pow(p, power);
}

float fresnel_schlick(float F0, float cos_theta_incident)
{
	return glm::mix(F0, 1.f, fresnel_schlick_ratio(cos_theta_incident, 1.0f));
}

glm::vec3 Renderer::TraceRay(Ray& ray, uint32_t& seed)
{
	glm::vec3 incomingLight = glm::vec3(0.0f);
	glm::vec3 rayColor = glm::vec3(1.0f);

	size_t bounces = 5;
	for (size_t b = 0; b <= bounces; b++)
	{
		Hit hit = CalculateRayCollision(ray);
		seed += (uint32_t)b;

		if (hit.didHit)
		{
			const Material& mat = m_activeScene->materials[hit.materialIndex];

			float F0 = glm::mix(0.02f, 1.0f, mat.metalness);
			float ndotv = glm::max(glm::dot(hit.worldNormal, -ray.direction), 0.0f);
			float F = fresnel_schlick(F0, ndotv);

			// Figure out new ray position and direction
			bool isSpecularBounce = F >= Utils::RandomFloat(seed);

			ray.origin = hit.worldPosition;
			glm::vec3 diffuseContribution = glm::mix(mat.albedo, glm::vec3(0.0f), mat.metalness);
			glm::vec3 diffuseDir = glm::normalize(hit.worldNormal + Utils::InUnitSphere(seed));
			glm::vec3 specularDir = glm::reflect(ray.direction, hit.worldNormal);
			ray.direction = glm::normalize(glm::mix(diffuseDir, specularDir, (1.0f-mat.roughness) * isSpecularBounce));

			// Update light calculations
			glm::vec3 emittedLight = mat.emissionColor * mat.emissionPower;
			incomingLight += emittedLight * rayColor;
			glm::vec3 specularColor = glm::mix(glm::vec3(0.1f), mat.albedo, mat.metalness);
			rayColor *= glm::mix(mat.albedo * diffuseContribution, specularColor, isSpecularBounce);

			// Random early exit if ray colour is nearly 0 (can't contribute much to final result)
			float p = glm::max(rayColor.r, glm::max(rayColor.g, rayColor.b));
			if (Utils::RandomFloat(seed) >= p)
			{
				break;
			}

			rayColor *= 1.0f / p;
		}
		else
		{
			incomingLight += GetEnvironmentLight(ray) * m_activeScene->m_skyColor * m_activeScene->m_skyBrightness * rayColor;
			break;
			//if (hit.debugHit == 0 && b <= 1)
		}
	}

	return incomingLight;
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

	glm::vec3 totalIncomingLight = glm::vec3(0.0f);
	totalIncomingLight += TraceRay(ray, seed);


	return glm::vec4(totalIncomingLight, 1.0f);
}

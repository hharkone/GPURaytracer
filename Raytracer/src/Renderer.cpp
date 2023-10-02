#include <execution>
#include <glm/gtc/type_ptr.hpp>

#include "Renderer.h"
#include "Walnut/Random.h"

namespace Utils
{
	float4 vec4Tofloat4(glm::vec4& v)
	{
		return make_float4(v.x, v.y, v.z, v.w);
	}
}

void Renderer::ResetFrameIndex()
{
	m_frameIndex = 1;

	if(m_cudaRenderer)
		m_cudaRenderer->Clear();
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_finalImage)
	{
		if (m_finalImage->GetWidth() == width && m_finalImage->GetHeight() == height)
			return;

		m_finalImage->Resize(width, height);
		ResetFrameIndex();
	}
	else
	{
		m_finalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}

	delete[] m_imageData;
	m_imageData = new uint32_t[width * height];

	m_imgHorizontalIterator.resize(width);
	m_imgVerticalIterator.resize(height);

	for (uint32_t i = 0; i < width; i++)
		m_imgHorizontalIterator[i] = i;

	for (uint32_t i = 0; i < height; i++)
		m_imgVerticalIterator[i] = i;

	m_cudaRenderer = std::shared_ptr<CudaRenderer>(new CudaRenderer(width, height, &m_activeScene, &m_frameIndex, 1u, &m_settings.bounces));
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_activeScene = &scene;
	m_activeCamera = &camera;

	if (m_cudaRenderer && m_settings.accumulate)
	{
		glm::vec3 glmPos = m_activeCamera->GetPosition();
		float3 pos = make_float3(glmPos.x, glmPos.y, glmPos.z);
		glm::vec3 glmDir = m_activeCamera->GetDirection();
		float3 dir = make_float3(glmDir.x, glmDir.y, glmDir.z);

		glm::mat4x4 invView = m_activeCamera->GetInverseView();
		float4 x1 = Utils::vec4Tofloat4(invView[0]);
		float4 y1 = Utils::vec4Tofloat4(invView[1]);
		float4 z1 = Utils::vec4Tofloat4(invView[2]);
		float4 w1 = Utils::vec4Tofloat4(invView[3]);

		glm::mat4x4 invProj = m_activeCamera->GetInverseProjection();
		float4 x2 = Utils::vec4Tofloat4(invProj[0]);
		float4 y2 = Utils::vec4Tofloat4(invProj[1]);
		float4 z2 = Utils::vec4Tofloat4(invProj[2]);
		float4 w2 = Utils::vec4Tofloat4(invProj[3]);

		glm::mat4x4 viewMat = m_activeCamera->GetView();
		float4 x3 = Utils::vec4Tofloat4(viewMat[0]);
		float4 y3 = Utils::vec4Tofloat4(viewMat[1]);
		float4 z3 = Utils::vec4Tofloat4(viewMat[2]);
		float4 w3 = Utils::vec4Tofloat4(viewMat[3]);

		glm::mat4x4 localToWorldMat = m_activeCamera->GetLocalToWorld();
		float4 x4 = Utils::vec4Tofloat4(localToWorldMat[0]);
		float4 y4 = Utils::vec4Tofloat4(localToWorldMat[1]);
		float4 z4 = Utils::vec4Tofloat4(localToWorldMat[2]);
		float4 w4 = Utils::vec4Tofloat4(localToWorldMat[3]);

		m_cudaRenderer->SetCamera(pos, dir, m_activeCamera->m_aperture, m_activeCamera->m_focusDistance);
		m_cudaRenderer->SetInvViewMat(x1, y1, z1, w1);
		m_cudaRenderer->SetInvProjMat(x2, y2, z2, w2);
		m_cudaRenderer->SetViewMat(x3, y3, z3, w3);
		m_cudaRenderer->SetLocalToWorldMat(x4, y4, z4, w4);

		//m_cudaData = m_cudaRenderer->getOutputData();
		m_cudaRenderer->Compute();

		m_imageData = m_cudaRenderer->getImageData();

		if (m_imageData)
		{
			m_finalImage->SetData(m_imageData);
		}

		m_frameIndex++;
	}
/*
#if true
	std::for_each(std::execution::par, m_imgVerticalIterator.begin(), m_imgVerticalIterator.end(),
		[this](uint32_t y)
		{
			std::for_each(std::execution::par, m_imgHorizontalIterator.begin(), m_imgHorizontalIterator.end(),
				[this, y](uint32_t x)
				{
					size_t index = x + y * m_finalImage->GetWidth();
					glm::vec4 color;

					color.r = m_cudaData[index * 3u + 0u];
					color.g = m_cudaData[index * 3u + 1u];
					color.b = m_cudaData[index * 3u + 2u];
					color.a = 1.0f;

					color /= (float)m_frameIndex;

					color.a = 1.0f;

					color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0));
					color = glm::pow(color, glm::vec4(0.46464f));
					m_imageData[index] = Utils::ConvertToRGBA(color);
				});
		});
#else
	std::for_each(m_imgVerticalIterator.begin(), m_imgVerticalIterator.end(),
		[this](uint32_t y)
		{
			std::for_each( m_imgHorizontalIterator.begin(), m_imgHorizontalIterator.end(),
			[this, y](uint32_t x)
				{
					size_t index = x + y * m_finalImage->GetWidth();
					glm::vec4 color;

					color.r = m_cudaData[index * 3u + 0u];
					color.g = m_cudaData[index * 3u + 1u];
					color.b = m_cudaData[index * 3u + 2u];
					color.a = 1.0f;

					color /= (float)m_frameIndex;

					color.a = 1.0f;

					color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0));
					//accumulatedColor = glm::pow(accumulatedColor, glm::vec4(0.46464f));
					m_imageData[index] = Utils::ConvertToRGBA(color);
				});
		});
#endif
*/
}
/*
bool RayBoundingBox(const RayCPU& ray, const Mesh& mesh)
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

Renderer::Hit Renderer::rayTriangleIntersect(const RayCPU& ray, const Mesh::Triangle& tri, const glm::vec3& origin)
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

// Calculate the intersection of a ray with a sphere
Renderer::Hit Renderer::raySphere(const RayCPU& ray, const Sphere& sphere)
{
	Hit hitInfo;

	glm::vec3 offsetRayOrigin = ray.origin - sphere.position;
	// From the equation: sqrLength(rayOrigin + rayDir * dst) = radius^2
	// Solving for dst results in a quadratic equation with coefficients:
	float a = glm::dot(ray.direction, ray.direction); // a = 1 (assuming unit vector)
	float b = 2.0f * glm::dot(offsetRayOrigin, ray.direction);
	float c = dot(offsetRayOrigin, offsetRayOrigin) - sphere.radius * sphere.radius;

	// Quadratic discriminant
	float discriminant = b * b - 4.0f * a * c;

	// No solution when d < 0 (ray misses sphere)
	if (discriminant >= 0.0f)
	{
		// Distance to nearest intersection point (from quadratic formula)
		float dst = (-b - sqrt(discriminant)) / (2 * a);

		// Ignore intersections that occur behind the ray
		if (dst >= 0) {
			hitInfo.didHit = true;
			hitInfo.hitDistance = dst;
			hitInfo.worldPosition = ray.origin + ray.direction * dst;
			hitInfo.worldNormal = glm::normalize(hitInfo.worldPosition - sphere.position);
		}
	}
	return hitInfo;
}

Renderer::Hit Renderer::CalculateRayCollision(const RayCPU& ray)
{
	Hit closestHit;

	//Spheres
	for (int i = 0; i < m_activeScene->spheres.size(); i++)
	{
		Sphere sphere = m_activeScene->spheres[i];
		Hit hitInfo = raySphere(ray, sphere);

		if (hitInfo.didHit && hitInfo.hitDistance < closestHit.hitDistance)
		{
			closestHit = hitInfo;
			closestHit.objectIndex = (int)i;
			closestHit.materialIndex = sphere.materialIndex;
		}
	}

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

glm::vec3 Renderer::GetEnvironmentLight(RayCPU& ray)
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
	return glm::mix(F0, 1.f, fresnel_schlick_ratio(cos_theta_incident, 2.0f));
}

glm::vec3 Renderer::TraceRay(RayCPU& ray, uint32_t& seed)
{
	glm::vec3 incomingLight = glm::vec3(0.0f);
	glm::vec3 rayColor = glm::vec3(1.0f);

	for (size_t b = 0; b <= GetSettings().bounces; b++)
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
			bool isSpecularBounce = 0.5f >= Utils::RandomFloat(seed);

			ray.origin = hit.worldPosition;
			glm::vec3 diffuseContribution = glm::mix(mat.albedo, glm::vec3(0.0f), mat.metalness);
			glm::vec3 diffuseDir = glm::normalize(hit.worldNormal + Utils::RandomDirection(seed));
			glm::vec3 specularDir = glm::reflect(ray.direction, hit.worldNormal);
			ray.direction = glm::normalize(glm::mix(diffuseDir, specularDir, (1.0f-mat.roughness) * isSpecularBounce));

			// Update light calculations
			glm::vec3 emittedLight = mat.emissionColor * mat.emissionPower;
			incomingLight += emittedLight * rayColor;
			glm::vec3 specularColor = glm::mix(glm::vec3(F), mat.albedo, mat.metalness);
			rayColor *= glm::mix(mat.albedo * diffuseContribution, specularColor, isSpecularBounce);

			// Random early exit if ray colour is nearly 0 (can't contribute much to final result)
			float p = glm::max(rayColor.r, glm::max(rayColor.g, rayColor.b));
			if (Utils::RandomFloat(seed) >= p)
			{
				break;
			}

			rayColor *= 1.0f / p;

			//rayColor = glm::vec3(specularColor);
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
	RayCPU ray;
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
*/
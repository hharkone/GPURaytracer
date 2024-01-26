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
		m_finalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA32F);
	}

	if (!m_cudaRenderer)
	{
		m_cudaRenderer = std::shared_ptr<CudaRenderer>(new CudaRenderer(width, height, &m_activeScene, &m_frameIndex, &GetSettings().samples, &m_settings.bounces));
	}
	else
	{
		m_cudaRenderer->OnResize(width, height);
	}

	m_denoiser.InitOptix((void*)m_cudaRenderer->getFloatOutputDataDevice()->d_pointer(),
						 (void*)m_cudaRenderer->getFloatAlbedoOutputDataDevice()->d_pointer(),
						 (void*)m_cudaRenderer->getFloatNormalOutputDataDevice()->d_pointer(),
						 width, height);
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_activeScene = &scene;
	m_activeCamera = &camera;

	if (m_cudaRenderer)
	{
		if (m_settings.accumulate)
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

			m_cudaRenderer->Compute();

			m_frameIndex++;
		}

		m_denoiser.Denoise(m_activeScene, m_settings.denoise);

		if (m_denoiser.GetDenoisedBuffer())
		{
			m_finalImage->SetData(m_denoiser.GetDenoisedBuffer());
		}
	}
}

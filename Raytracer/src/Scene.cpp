#pragma once

#include "Scene.h"

void Scene::UploadMaterials()
{
    materialBuffer.upload(materials, materialCount);
    materialBufferPtr = materialBuffer.d_pointer();
}

Scene::Scene()
{
    //Upload default material buffer to GPU.
    materialBuffer.alloc_and_upload(materials, materialCount);
    materialBufferPtr = materialBuffer.d_pointer();

    ImportMesh("meshes/dragon2.obj");
}

Scene::~Scene()
{
    materialBuffer.free();
    delete[] materials;
}

void Scene::ImportMesh(std::string path)
{
    //sceneMesh = new GPU_Mesh();
    sceneMesh.LoadOBJFile(path);

    ResizeAndAddMaterials(sceneMesh.meshInfoBuffer->materialCount);
}

void Scene::ImportMesh(std::string path, size_t overrideMaterial)
{
	//sceneMesh = new GPU_Mesh();
	sceneMesh.LoadOBJFile(path, overrideMaterial);
}

void Scene::AddNewDefaultMaterial(size_t index)
{
    materials[index] = Material{ { 0.8f, 0.8f,  0.8f  }, 0.0f, 0.21f, { 0.0f, 0.0f, 0.0f }, 0.0f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, { 0.95f, 0.75f, 0.4f }, 0.0f };
}

void Scene::ResizeAndAddMaterials(size_t size)
{
    materialBuffer.free();

    Material* newArr = new Material[materialCount + size];
    memcpy(newArr, &materials[0], materialCount * sizeof(Material));
    delete[] materials;
    materials = newArr;

    for (int i = materialCount - 1u; i < materialCount + size; i++)
    {
        AddNewDefaultMaterial(i);
    }

    materialCount += size;

    materialBuffer.alloc_and_upload(materials, materialCount);
    materialBufferPtr = materialBuffer.d_pointer();
}
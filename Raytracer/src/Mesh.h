#pragma once

#include <glm/glm.hpp>
#include <vector>

class Mesh
{

public:

    struct Vertex
    {
        Vertex(glm::vec3 pos, glm::vec3 normal, glm::vec2 uv) : pos(pos), normal(normal), uv(uv) {}

        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    struct Triangle
    {
        Triangle(Vertex v0, Vertex v1, Vertex v2) : v0(v0), v1(v1), v2(v2) {}

        Vertex v0 = v0;
        Vertex v1 = v1;
        Vertex v2 = v2;
    };

    struct MeshData
    {
        std::vector<glm::vec3> verts;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> uvs;
        std::vector<Triangle>  tris;
        glm::vec3 Transform;
        int materialIndex;
        std::string name;
    };

    MeshData LoadOBJFile(const std::string& path);
};
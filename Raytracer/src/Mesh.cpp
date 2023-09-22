//#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "Mesh.h"

Mesh::MeshData Mesh::LoadOBJFile(const std::string& path)
{
    Mesh::MeshData mesh;
    std::ifstream infile(path, std::ifstream::in);
    std::string line;

    float x, y, z;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9;
    std::string s;

    while (std::getline(infile, line))
    {
        std::string test = line.substr(0, 2);
        if (test == "v ")
        {
            if(sscanf_s(line.c_str(), "v %f %f %f\n", &x, &y, &z) == 3)
                mesh.verts.push_back(glm::vec3(x, y, z));
        }
        else if (test == "vn")
        {
            if (sscanf_s(line.c_str(), "vn %f %f %f\n", &x, &y, &z) == 3)
                mesh.normals.push_back(glm::vec3(x, y, z));
        }
        else if (test == "vt")
        {
            if (sscanf_s(line.c_str(), "vt %f %f\n", &x, &y) == 2)
                mesh.uvs.push_back(glm::vec2(x, y));
        }
        else if (test == "f ")
        {
            if (sscanf_s(line.c_str(), "f %i/%i/%i %i/%i/%i %i/%i/%i\n", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9) == 9)
            {
                Mesh::Vertex v0 = { mesh.verts[f1 - 1], mesh.normals[f3 - 1], mesh.uvs[f2 - 1] };
                Mesh::Vertex v1 = { mesh.verts[f4 - 1], mesh.normals[f6 - 1], mesh.uvs[f5 - 1] };
                Mesh::Vertex v2 = { mesh.verts[f7 - 1], mesh.normals[f9 - 1], mesh.uvs[f8 - 1] };

                mesh.tris.push_back(Mesh::Triangle(v0, v1, v2));
            }
                
        }

    }

    return mesh;
}
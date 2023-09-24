//#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "Mesh.h"

void Mesh::CalculateBbox(Mesh& mdata)
{
    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = std::numeric_limits<float>::max();
    max_x = max_y = max_z = std::numeric_limits<float>::min();

    for (size_t i = 0u; i < mdata.verts.size(); i++)
    {
        if (mdata.verts[i].x < min_x) min_x = mdata.verts[i].x;
        if (mdata.verts[i].x > max_x) max_x = mdata.verts[i].x;
        if (mdata.verts[i].y < min_y) min_y = mdata.verts[i].y;
        if (mdata.verts[i].y > max_y) max_y = mdata.verts[i].y;
        if (mdata.verts[i].z < min_z) min_z = mdata.verts[i].z;
        if (mdata.verts[i].z > max_z) max_z = mdata.verts[i].z;
    }

    mdata.bbox.min = glm::vec3(min_x, min_y, min_z) + Transform;
    mdata.bbox.max = glm::vec3(max_x, max_y, max_z) + Transform;
}

Mesh::Bbox Mesh::GetMeshBoundingBox(const Mesh& mdata)
{
    Mesh::Bbox bbox;
    bbox.max = mdata.bbox.max + mdata.Transform;
    bbox.min = mdata.bbox.min + mdata.Transform;

    return bbox;
}

Mesh Mesh::LoadOBJFile(const std::string& path)
{
    Mesh mesh;
    std::ifstream infile(path, std::ifstream::in);
    std::string line;

    float x, y, z;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9;
    std::string s;

    while (std::getline(infile, line))
    {
        std::string test = line.substr(0, 2);
        if (test == "o ")
        {
            mesh.name = line.substr(2, line.length()-2);
        }
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

    CalculateBbox(mesh);

    return mesh;
}
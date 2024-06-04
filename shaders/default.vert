#version 330 core

//--------------------------------------------

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTexCoord;
layout (location=2) in vec3 vertexNormal;

out vec2 fragmentTexCoord;
out vec3 fragmentPosition;
out vec3 fragmentNormal;

uniform mat4 model;
uniform mat4 projections;
uniform mat4 view;

//--------------------------------------------

void main()
{
    gl_Position = projections * view * model * vec4(vertexPos, 1.0);
    fragmentTexCoord = vertexTexCoord;
    fragmentNormal = mat3(model) * vertexNormal;
    fragmentPosition = (model * vec4(vertexPos,1.0)).xyz;
}

//--------------------------------------------
#version 330 core

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

out vec4 color;

uniform sampler2D imageTexture;
uniform vec3 tint;

void main()
{
    vec4 baseTexture = texture(imageTexture, fragmentTexCoord);
    color = vec4(tint,1) * baseTexture;
}

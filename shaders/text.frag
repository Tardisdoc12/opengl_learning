#version 330 core

in vec2 TexCoords;
out vec4 color;
uniform sampler2D screenTexture;
uniform vec4 textColor;

void main()
{             
    color = textColor * texture(screenTexture, TexCoords);
}
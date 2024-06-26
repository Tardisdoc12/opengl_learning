#version 330 core

in vec2 fragmentTexCoord;

out vec4 fragmentColor;

uniform sampler2D material;

const vec2 curvature = vec2(3.0, 3.0);
const float opacity = 0.5;
const float gamma = 2.2;

vec2 curveRemapUV(){
    vec2 uv = vec2(2.0) * fragmentTexCoord - vec2(1.0);
    vec2 offset = abs(uv.xy)/curvature;
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

vec4 scanLineIntensity(float uv, float resolution){
    float intensity = sin(uv * resolution * 3.1415926 * 2.0);
    intensity = ((0.5 * intensity) + 0.5) * 0.9 + 0.1;
    return vec4(vec3(pow(intensity, opacity)),1.0);
}

vec4 gammaCorrection(vec4 color){
    return vec4(pow(color.rgb, vec3(1.0/gamma)), 1.0);
}

void main(){
    vec2 uv = curveRemapUV();
    fragmentColor = texture(material, uv);
    fragmentColor *= scanLineIntensity(uv.x, 1024);
    fragmentColor *= scanLineIntensity(uv.y, 1024);
    fragmentColor = gammaCorrection(fragmentColor);
}
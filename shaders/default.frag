#version 330 core

//------------------------------------------------------------------------------------------

struct PointLight{
    vec3 position;
    vec3 color;
    float strength;
};

struct Materials{
    sampler2D albedo;
    sampler2D ao;
    sampler2D normal;
    sampler2D specular;
};

//------------------------------------------------------------------------------------------

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

out vec4 color;

vec3 calculatePointLight(PointLight light,vec3 fragmentPosition, vec3 fragmentNormal);

uniform PointLight Lights[8];
uniform vec3 cameraPosition;
uniform Materials material;

//------------------------------------------------------------------------------------------

void main()
{
    vec4 baseTexture = texture(material.albedo, fragmentTexCoord);
    vec3 temp = vec3(0.0);

    //ambient
    temp += 0.2 * baseTexture.rgb;

    for(int i = 0; i < 8; i++){
        temp += calculatePointLight(Lights[i],fragmentPosition, fragmentNormal);
    }
    color = vec4(temp, baseTexture.a);
}


//------------------------------------------------------------------------------------------
vec3 calculatePointLight(PointLight light,vec3 fragmentPosition, vec3 fragmentNormal){
    vec3 result = vec3(0.0);
    vec3 baseTexture = texture(material.albedo, fragmentTexCoord).rgb;
    vec3 specularTexture = texture(material.specular, fragmentTexCoord).rgb;

    //geometric data
    vec3 fragLight = light.position - fragmentPosition;
    float distance = length(fragLight);
    fragLight = normalize(fragLight);
    vec3 fragCamera = normalize(cameraPosition - fragmentPosition);
    vec3 halfVec = normalize(fragLight + fragCamera);

    

    //diffuse
    result += light.color * light.strength * max(0.0, dot(fragmentNormal, fragLight)) / (distance * distance) * baseTexture;

    //specular
    result += light.color * light.strength * pow(max(0.0, dot(fragmentNormal, halfVec)),32) / (distance * distance) * specularTexture;

    return result;
}

//------------------------------------------------------------------------------------------
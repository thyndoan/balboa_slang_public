StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;
uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix; // inverse transpose of the top-left 3x3 above
uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;
uniform float3 Ks;
uniform float exponent;
Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    float4 p = float4(vertices[vid], 1.0);
    output.position = mul(MVP, p);
    output.world_position = mul(model_matrix, p).xyz;
    output.world_normal = mul(normal_matrix, vertex_normals[vid]);
    output.uv = vertex_uvs[vid];
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    float3 n = normalize(input.world_normal);
    float3 l = normalize(light_dir);
    float3 v = normalize(camera_position - input.world_position);

    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;

    float3 Ca = 0.2 * Kd;

    float n_dot_l = dot(n, l);
    float3 Cd = Kd * light_color * max(0.0, n_dot_l);

    float3 Cs = float3(0.0);
    if (n_dot_l > 0.0) {
        float3 r = reflect(-l, n);
        Cs = Ks * light_color * pow(max(0.0, dot(r, v)), exponent);
    }

    return float4(Ca + Cd + Cs, 1.0);
}
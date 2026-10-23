float cross2(float2 a, float2 b, float2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool valid_triangle(float3 v0, float3 v1, float3 v2) {
    float total = cross2(v0.xy, v1.xy, v2.xy);
    if (!(abs(total) > 0.0))
        return false;
    if (!(v0.z > 0.0) || !(v1.z > 0.0) || !(v2.z > 0.0))
        return false;
    return true;
}

bool triangle_bbox(float3 v0, float3 v1, float3 v2,
                   uint width, uint height,
                   out int2 bmin, out int2 bmax) {
    bmin = int2(0, 0);
    bmax = int2(-1, -1);

    float minx = floor(min(min(v0.x, v1.x), v2.x));
    float miny = floor(min(min(v0.y, v1.y), v2.y));
    float maxx = ceil(max(max(v0.x, v1.x), v2.x));
    float maxy = ceil(max(max(v0.y, v1.y), v2.y));

    if (maxx < 0.0 || maxy < 0.0 ||
        minx > float(width) || miny > float(height))
        return false;

    minx = clamp(minx, 0.0, float(width - 1));
    maxx = clamp(maxx, 0.0, float(width - 1));
    miny = clamp(miny, 0.0, float(height - 1));
    maxy = clamp(maxy, 0.0, float(height - 1));

    bmin = int2(int(minx), int(miny));
    bmax = int2(int(maxx), int(maxy));
    return true;
}

bool projected_barycentric(float2 p, float3 v0, float3 v1, float3 v2,
                           out float3 bp) {
    float total = cross2(v0.xy, v1.xy, v2.xy);
    float inv_total = 1.0 / total;
    bp.x = cross2(p, v1.xy, v2.xy) * inv_total;
    bp.y = cross2(v0.xy, p, v2.xy) * inv_total;
    bp.z = cross2(v0.xy, v1.xy, p) * inv_total;
    return bp.x >= 0.0 && bp.y >= 0.0 && bp.z >= 0.0;
}

float interpolate_depth(float3 bp, float3 v0, float3 v1, float3 v2) {
    float inv_depth = bp.x * v0.z + bp.y * v1.z + bp.z * v2.z;
    return 1.0 / inv_depth;
}

float3 perspective_barycentric(float3 bp, float3 v0, float3 v1, float3 v2) {
    float3 w = float3(bp.x * v0.z, bp.y * v1.z, bp.z * v2.z);
    return w / (w.x + w.y + w.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
uniform uint num_pixels,
RWStructuredBuffer<uint> depth_buffer,
RWStructuredBuffer<uint> triangle_id_buffer) {
if (tid >= num_pixels)
return;

depth_buffer[tid] = 0xffffffff;
triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
uint tid : SV_DispatchThreadID,
uniform uint num_vertices,
uniform uint vertex_offset,
uniform float4x4 MVP,
StructuredBuffer<float3> vertices,
RWStructuredBuffer<float3> projected_vertices)
{
if (tid >= num_vertices)
return;

uint i = vertex_offset + tid;
float4 p = mul(MVP, float4(vertices[i], 1.0));
projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
uint tid : SV_DispatchThreadID,
uniform uint num_triangles,
uniform uint width,
uniform uint height,
uniform float s,
uniform float a,
StructuredBuffer<uint3> faces,
// A projected vertex store the screen space coordinates
// in xy, and the inverse depth in z.
StructuredBuffer<float3> projected_vertices,
RWStructuredBuffer<uint> depth_buffer) {
// HW 2.5 TODO
    if (tid >= num_triangles)
        return;

    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    if (!valid_triangle(v0, v1, v2))
        return;

    int2 bmin, bmax;
    if (!triangle_bbox(v0, v1, v2, width, height, bmin, bmax))
        return;

    for (int y = bmin.y; y <= bmax.y; y++) {
        for (int x = bmin.x; x <= bmax.x; x++) {
            float2 p = float2(x, y) + float2(0.5, 0.5);
            float3 bp;
            if (!projected_barycentric(p, v0, v1, v2, bp))
                continue;

            float d = interpolate_depth(bp, v0, v1, v2);
            uint depth_bits = asuint(d);
            InterlockedMin(depth_buffer[y * width + x], depth_bits);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
uint tid : SV_DispatchThreadID,
uniform uint num_triangles,
uniform uint width,
uniform uint height,
uniform float s,
uniform float a,
StructuredBuffer<uint3> faces,
StructuredBuffer<float3> projected_vertices,
RWStructuredBuffer<uint> depth_buffer,
RWStructuredBuffer<uint> triangle_id_buffer) {
// HW 2.5 TODO
    if (tid >= num_triangles)
        return;

    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    if (!valid_triangle(v0, v1, v2))
        return;

    int2 bmin, bmax;
    if (!triangle_bbox(v0, v1, v2, width, height, bmin, bmax))
        return;

    for (int y = bmin.y; y <= bmax.y; y++) {
        for (int x = bmin.x; x <= bmax.x; x++) {
            float2 p = float2(x, y) + float2(0.5, 0.5);
            float3 bp;
            if (!projected_barycentric(p, v0, v1, v2, bp))
                continue;

            float d = interpolate_depth(bp, v0, v1, v2);
            uint idx = y * width + x;
            if (asuint(d) == depth_buffer[idx]) {
                InterlockedMin(triangle_id_buffer[idx], tid);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
uint2 tid : SV_DispatchThreadID,
uniform uint width,
uniform uint height,
uniform float s,
uniform float a,
StructuredBuffer<uint3> faces,
StructuredBuffer<float3> vertex_colors,
StructuredBuffer<float3> projected_vertices,
StructuredBuffer<uint> triangle_id_buffer,
RWTexture2D<float4> image) {
if (tid.x >= width || tid.y >= height) {
return;
    }

// HW 2.5 TODO
    uint face_id = triangle_id_buffer[tid.y * width + tid.x];
    if (face_id == 0xffffffff)
        return;

    uint3 f = faces[face_id];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    if (!valid_triangle(v0, v1, v2))
        return;

    float2 p = float2(tid) + float2(0.5, 0.5);
    float3 bp;
    projected_barycentric(p, v0, v1, v2, bp);

    float3 b = perspective_barycentric(bp, v0, v1, v2);

    float3 color = b.x * vertex_colors[f.x]
                 + b.y * vertex_colors[f.y]
                 + b.z * vertex_colors[f.z];

    image[tid] = float4(color, 1.0);
}

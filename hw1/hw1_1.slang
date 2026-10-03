[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    float2 pixel_center = float2(
    tid.x + 0.5,
    height - (tid.y + 0.5)  
    );


    // Vector from the circle center to the pixel center
    float2 difference = pixel_center - center;

    // Squared distance from pixel center to circle center
    float distance_squared = dot(difference, difference);

    // If the pixel center is inside the circle, use the circle color
    if (distance_squared <= radius * radius) {
    image[tid] = float4(color, 1.0);
    }
}


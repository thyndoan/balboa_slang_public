[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // pixel center
    float2 p = float2(ipixel) + float2(0.5, 0.5);

    // calculate the three edges
    // edge 1, from sp0-sp1
    float2 e01 = sp1 - sp0;
    float2 n01 = float2(-e01.y, e01.x);
    float E01 = dot((p - sp0), n01);

    // edge 2, from sp1-sp2
    float2 e12 = sp2 - sp1;
    float2 n12 = float2(-e12.y, e12.x);
    float E12 = dot((p - sp1), n12);

    // edge 3, from sp2-sp0
    float2 e20 = sp0 - sp2;
    float2 n20 = float2(-e20.y, e20.x);
    float E20 = dot((p - sp2), n20);

    // Check to see if it's inside the triangle
    bool all_positive = (E01 >= 0.0) && (E12 >= 0.0) && (E20 >= 0.0);
    bool all_negative = (E01 <= 0.0) && (E12 <= 0.0) && (E20 <= 0.0);

    // If either all positive or all negative => inside the triangle
    if (all_positive || all_negative) {
        image[ipixel] = float4(color, 1.0);
    }
}

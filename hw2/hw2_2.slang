// area helper to calculate area of 3 points
float area(float2 a, float2 b, float2 c) {
    float2 ab = b - a;
    float2 ac = c - a;

    return abs(ab.x * ac.y - ab.y * ac.x) * 0.5;
}

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

    // pixel center (p_screen)
    float2 p_screen = float2(ipixel) + float2(0.5, 0.5);

    // calculate the three edges
    // edge 1, from sp0-sp1
    float2 e01 = sp1 - sp0;
    float2 n01 = float2(-e01.y, e01.x);
    float E01 = dot((p_screen - sp0), n01);

    // edge 2, from sp1-sp2
    float2 e12 = sp2 - sp1;
    float2 n12 = float2(-e12.y, e12.x);
    float E12 = dot((p_screen - sp1), n12);

    // edge 3, from sp2-sp0
    float2 e20 = sp0 - sp2;
    float2 n20 = float2(-e20.y, e20.x);
    float E20 = dot((p_screen - sp2), n20);

    // Check to see if it's inside the triangle
    bool all_positive = (E01 >= 0.0) && (E12 >= 0.0) && (E20 >= 0.0);
    bool all_negative = (E01 <= 0.0) && (E12 <= 0.0) && (E20 <= 0.0);

    // If either all positive or all negative => inside the triangle
    bool inside;
    if (all_positive || all_negative) {
        inside = true;
    }
    else {
        inside = false;
    }
    if (!inside) {
        return;
    }

    // (a) convert screen point back to projected camera space
    float x = (p_screen.x / width) * (2.0 * s * a) - s * a;
    float y = s - (p_screen.y / height) * (2.0 * s);
    float2 p = float2(x,y);

    // (b) Projected barycentric coordinates (equation 6)
    float total_area = area(sp0, sp1, sp2);
    float b0p = area(p_screen, sp1, sp2) / total_area;
    float b1p = area(sp0, p_screen, sp2) / total_area;
    float b2p = area(sp0, sp1, p_screen) / total_area;

    // (c) perspective correction barycentric coordinates (equation 15)
    float w0 = b0p * inv_depth0;
    float w1 = b1p * inv_depth1;
    float w2 = b2p * inv_depth2;

    float sum = w0 + w1 + w2;

    float b0 = w0 / sum;
    float b1 = w1 / sum;
    float b2 = w2 / sum;

    // (d) obtain depth/ z value (equation 16)
    // (d) obtain depth using projected barycentric coordinates
    float inv_depth = b0p * inv_depth0
                    + b1p * inv_depth1
                    + b2p * inv_depth2;
    
    float depth = 1.0 / inv_depth;

    // depth test
    if (depth < depth_image[ipixel]) {
        depth_image[ipixel] = depth;

        // (e) color interpolation (equation 17)
        float3 color = b0 * c0 + b1 * c1 + b2 * c2;

        color_image[ipixel] = float4(color, 1.0);
    }
}

// This tells Slang that this function is a compute shader.
// A compute shader performs calculations in parallel on the GPU.
[shader("compute")]

// Each thread group contains 8 threads in the x direction
// and 8 threads in the y direction.
// Therefore, one thread group contains 8 * 8 = 64 threads.
[numthreads(8, 8, 1)]

// This is the main function executed by the GPU.
// Each thread is responsible for processing one pixel.
void compute_main(
    // tid = thread ID.
    // tid.x is the pixel's horizontal coordinate.
    // tid.y is the pixel's vertical coordinate.
    uint2 tid : SV_DispatchThreadID,

    // The center coordinate of the circle.
    uniform float2 center,

    // The radius of the circle.
    uniform float radius,

    // The RGB color used to draw the circle.
    uniform float3 color,

    // The output image.
    // float4 represents RGBA: red, green, blue, and alpha.
    RWTexture2D<float4> image
) {

    // Get the width and height of the output image.
    uint width, height;
    image.GetDimensions(width, height);

    // Check whether this thread is outside the image boundaries.
    // If it is outside, stop this thread to avoid accessing
    // an invalid pixel.
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Set the background color to gray.
    // The first three values represent RGB.
    // The last value, 1.0, means fully opaque.
    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    // Calculate the center position of the current pixel.
    // We add 0.5 because we want to test the center of the pixel,
    // rather than its top-left corner.
    //
    // height - (tid.y + 0.5) flips the y-coordinate.
    // This makes the coordinate system start at the bottom-left,
    // which matches the scene's coordinate convention.
    float2 pixel_center = float2(
        tid.x + 0.5,
        height - (tid.y + 0.5)
    );

    // Calculate the vector from the circle center to the current pixel center.
    //
    // difference contains the horizontal and vertical distances.
    float2 difference = pixel_center - center;

    // Calculate the squared distance between the pixel center
    // and the circle center.
    //
    // dot(difference, difference) is equivalent to: difference.x * difference.x + difference.y * difference.y
    //
    // We use squared distance so we do not need to calculate the square root.
    float distance_squared = dot(difference, difference);

    // A point is inside the circle if its distance from the
    // center is less than or equal to the radius.
    //
    // We compare squared distance with radius squared.
    if (distance_squared <= radius * radius) {

        // If the pixel is inside the circle,
        // replace the gray background with the circle color.
        image[tid] = float4(color, 1.0);
    }
}
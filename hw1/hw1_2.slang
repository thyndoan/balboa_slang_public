/*Initialize the background and find the current pixel center.

Determine whether the pixel belongs to the stroke.

Determine whether the pixel belongs to the fill.

Draw the fill first and the stroke second.
*/

// Import shared definitions and utilities.
import hw1_common;

// Declare this function as a compute shader.
[shader("compute")]

// Each thread group contains 8 x 8 threads.
[numthreads(8, 8, 1)]

void compute_main(
    // The ID of the current GPU thread.
    // Each thread handles one pixel.
    uint2 tid : SV_DispatchThreadID,

    // An array containing all the polyline points.
    StructuredBuffer<float2> polyline,

    // Number of points in the polyline.
    uniform int num_polyline,

    // Whether the polyline connects its last point back to its first point.
    uniform bool is_closed,

    // Whether the polyline should have a fill color.
    uniform bool use_fill_color,

    // RGB color used for filling.
    uniform float3 fill_color,

    // Whether the polyline should have a stroke.
    uniform bool use_stroke_color,

    // RGB color used for the stroke.
    uniform float3 stroke_color,

    // Thickness of the stroke.
    uniform float stroke_width,

    // Output image.
    RWTexture2D<float4> image
) {

    // Get the output image dimensions.
    uint width, height;
    image.GetDimensions(width, height);

    // Ignore threads outside the image.
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Initialize the pixel with a gray background.
    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    // Calculate the center of the current pixel.
    // Flip y so the coordinate origin is at the bottom-left.
    float2 p = float2(
        tid.x + 0.5,
        height - (tid.y + 0.5)
    );

    // Initially assume this pixel is not part of the stroke.
    bool in_stroke = false;

    // Only perform stroke calculations if stroke is enabled.
    if (use_stroke_color) {

        // Loop through every pair of consecutive points.
        // Each pair forms one line segment.
        for (int i = 0; i < num_polyline - 1; i++) {

            // Start point of the current segment.
            float2 A = polyline[i];

            // End point of the current segment.
            float2 B = polyline[i + 1];

            // Vector representing the line segment from A to B.
            float2 AB = B - A;

            // Vector from A to the current pixel center.
            float2 AP = p - A;

            // Squared length of the segment.
            // This is the dot product of AB with itself.
            float denom = dot(AB, AB);

            // Avoid dividing by zero if A and B are the same point.
            if (denom > 0.0) {

                // Project AP onto AB.
                // t tells us where the closest point lies
                // along the infinite line through A and B.
                float t = dot(AP, AB) / denom;

<<<<<<< HEAD
                // Only accept the projection if it falls
                // between A and B.
                //
                // t = 0 means A.
                // t = 1 means B.
                // Values between 0 and 1 are on the segment.
                if (t >= 0.0 && t <= 1.0) {

                    // Calculate the closest point on the segment.
                    float2 closest = A + t * AB;

                    // Calculate the distance from the pixel center
                    // to that closest point.
                    float distance = length(p - closest);

                    // The pixel belongs to the stroke if its
                    // distance is no greater than half the width.
=======
                // Count pixels that have closest point lies on the segment
                if (t >= 0.0 && t <= 1.0) {
                    float2 closest = A + t * AB;
                    float distance = length(p - closest);

                    // Stroke extends half the stroke width on each side
>>>>>>> main
                    if (distance <= stroke_width * 0.5) {
                        in_stroke = true;
                    }
                }
            }
        }

        // If the polyline is closed, also check the segment
        // connecting the last point back to the first point.
        if (is_closed && num_polyline > 1) {

            float2 A = polyline[num_polyline - 1];
            float2 B = polyline[0];

            float2 AB = B - A;
            float2 AP = p - A;

            float denom = dot(AB, AB);

            if (denom > 0.0) {

                float t = dot(AP, AB) / denom;

                if (t >= 0.0 && t <= 1.0) {
<<<<<<< HEAD

                    float2 closest = A + t * AB;

                    float distance = length(p - closest);

=======
                    float2 closest = A + t * AB;

                    float distance = length(p - closest);

>>>>>>> main
                    if (distance <= stroke_width * 0.5) {
                        in_stroke = true;
                    }
                }
            }
        }

        // Check interior vertices to create round joins.
        // These are the points where two line segments meet.
        for (int i = 1; i < num_polyline - 1; i++) {

            float2 vertex = polyline[i];

            // If the pixel is close enough to the vertex,
            // include it in the stroke.
            float distance = length(p - vertex);

            if (distance <= stroke_width * 0.5) {
                in_stroke = true;
            }
        }

        // For a closed polyline, the first and last points also meet.
        // Check those points for round joins.
        if (is_closed && num_polyline > 1) {

            float2 first_point = polyline[0];
            float2 last_point = polyline[num_polyline - 1];

            float distance_first = length(p - first_point);
            float distance_last = length(p - last_point);

            if (distance_first <= stroke_width * 0.5) {
                in_stroke = true;
            }

            if (distance_last <= stroke_width * 0.5) {
                in_stroke = true;
            }
        }
    }

    // Initially assume the pixel is not inside the filled region.
    bool in_fill = false;

    // A fill is only possible when the polyline is closed
    // and the user has enabled the fill color.
    if (is_closed && use_fill_color) {

        // Winding number counts how many times the polygon's
        // edges cross a horizontal ray extending from the pixel.
        int winding = 0;

        // Check every edge, including the edge from the
        // last point back to the first point.
        for (int i = 0; i < num_polyline; i++) {

            float2 A = polyline[i];

            // Modulo (%) wraps the last index back to zero.
            // This makes the last point connect to the first.
            float2 B = polyline[(i + 1) % num_polyline];

            // Check whether the edge crosses the horizontal ray
            // in the upward direction.
            if (A.y <= p.y) {

                if (B.y > p.y) {

                    // Cross product determines which side of
                    // the directed edge the pixel is on.
                    float cross =
                        (B.x - A.x) * (p.y - A.y)
                        - (p.x - A.x) * (B.y - A.y);

                    // Positive cross means this upward crossing
                    // contributes +1 to the winding number.
                    if (cross > 0.0) {
                        winding++;
                    }
                }
            }

            // Otherwise, check whether the edge crosses
            // the horizontal ray in the downward direction.
            else {

                if (B.y <= p.y) {

                    float cross =
                        (B.x - A.x) * (p.y - A.y)
                        - (p.x - A.x) * (B.y - A.y);

                    // Negative cross means this downward crossing
                    // contributes -1 to the winding number.
                    if (cross < 0.0) {
                        winding--;
                    }
                }
            }
        }

        // If the winding number is not zero,
        // the pixel is considered inside the polygon.
        if (winding != 0) {
            in_fill = true;
        }
    }

    // Draw the fill first.
    // This sets the pixel to the fill color if it is inside.
    if (in_fill && use_fill_color) {
        image[tid] = float4(fill_color, 1.0);
    }

    // Draw the stroke second.
    // This lets the stroke overwrite the fill where they overlap.
    if (in_stroke && use_stroke_color) {
        image[tid] = float4(stroke_color, 1.0);
    }
}
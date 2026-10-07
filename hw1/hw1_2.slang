import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }
    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    // p is the pixel center
    float2 p = float2(
    tid.x + 0.5,
    height - (tid.y + 0.5)  
    );


    bool in_stroke = false;

    if (use_stroke_color) {

        // Check every line segment between consecutive points
        for (int i = 0; i < num_polyline - 1; i++) {

            float2 A = polyline[i];
            float2 B = polyline[i + 1];

            float2 AB = B - A;
            float2 AP = p - A;

            float denom = dot(AB, AB);

            if (denom > 0.0) {

                // Find the closest point on the line segment
                float t = dot(AP, AB) / denom;

                // Count pixels that have closest point lies on the segment
                if (t >= 0.0 && t <= 1.0) {
                    float2 closest = A + t * AB;
                    float distance = length(p - closest);

                    // Stroke extends half the stroke width on each side
                    if (distance <= stroke_width * 0.5) {
                        in_stroke = true;
                    }
                }
            }
        }


        // If the polyline is closed, connect the last point
        // back to the first point
        if (is_closed && num_polyline > 1) {

            float2 A = polyline[num_polyline - 1];
            float2 B = polyline[0];

            float2 AB = B - A;
            float2 AP = p - A;

            float denom = dot(AB, AB);

            if (denom > 0.0) {

                float t = dot(AP, AB) / denom;

                if (t >= 0.0 && t <= 1.0) {
                    float2 closest = A + t * AB;

                    float distance = length(p - closest);

                    if (distance <= stroke_width * 0.5) {
                        in_stroke = true;
                    }
                }
            }
        }


        // Round joins at the interior vertices
        for (int i = 1; i < num_polyline - 1; i++) {

            float2 vertex = polyline[i];

            float distance = length(p - vertex);

            if (distance <= stroke_width * 0.5) {
                in_stroke = true;
            }
        }


        // For a closed polyline, the first and last points
        // are also connected, so give that corner a round join.
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

    bool in_fill = false;

    // Only closed polylines can be filled
    if (is_closed && use_fill_color) {

        // Non-zero winding number
        int winding = 0;

        // Check every edge, including the last point -> first point
        for (int i = 0; i < num_polyline; i++) {

            float2 A = polyline[i];
            float2 B = polyline[(i + 1) % num_polyline];


            // Edge goes upward across the horizontal ray
            if (A.y <= p.y) {

                if (B.y > p.y) {

                    float cross =
                        (B.x - A.x) * (p.y - A.y)
                        - (p.x - A.x) * (B.y - A.y);

                    if (cross > 0.0) {
                        winding++;
                    }
                }
            }


            // Edge goes downward across the horizontal ray
            else {

                if (B.y <= p.y) {

                    float cross =
                        (B.x - A.x) * (p.y - A.y)
                        - (p.x - A.x) * (B.y - A.y);

                    if (cross < 0.0) {
                        winding--;
                    }
                }
            }
        }


        // Non-zero winding number means the pixel is inside
        if (winding != 0) {
            in_fill = true;
        }
    }


    // Draw fill first
    if (in_fill && use_fill_color) {
        image[tid] = float4(fill_color, 1.0);
    }

    // Draw stroke second so the stroke appears on top
    if (in_stroke && use_stroke_color) {
        image[tid] = float4(stroke_color, 1.0);
    }
}
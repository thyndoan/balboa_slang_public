import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    image[tid] = float4(background, 1.0);
        float2 p = float2(
        tid.x + 0.5,
        height - (tid.y + 0.5)
    );

    for (int s = num_shapes - 1; s >= 0; s--){

        Shape shape = shapes[s];

        // Circle portion
        if (shape.type == ShapeType.Circle) {

            Circle c = circles[shape.index];

            float d = length(p - c.center);

            if (shape.use_fill &&
                d <= c.radius) {

                image[tid] = shape.fill_rgba;
            }

            if (shape.use_stroke &&
                abs(d - c.radius) <= shape.stroke_width * 0.5) {

                image[tid] = shape.stroke_rgba;
            }
        }

        // responsible for the Polyline
        else if (shape.type == ShapeType.Polyline) {

            Polyline line = polylines[shape.index];

            // Fill
            if (shape.use_fill && line.closed) {

                int winding = 0;

                for (int i = line.pts_begin;
                     i < line.pts_end;
                     i++) {

                    int j = i + 1;

                    if (j >= line.pts_end) {
                        j = line.pts_begin;
                    }

                    float2 a = points[i];
                    float2 b = points[j];

                    if (a.y <= p.y && b.y > p.y) {

                        float cross =
                            (b.x - a.x) * (p.y - a.y)
                            - (p.x - a.x) * (b.y - a.y);

                        if (cross > 0) {
                            winding++;
                        }

                    } else if (a.y > p.y && b.y <= p.y) {

                        float cross =
                            (b.x - a.x) * (p.y - a.y)
                            - (p.x - a.x) * (b.y - a.y);

                        if (cross < 0) {
                            winding--;
                        }
                    }
                }

                if (winding != 0) {
                    image[tid] = shape.fill_rgba;
                }
            }


            // Stroke part
            if (shape.use_stroke) {

                float r = shape.stroke_width * 0.5;
                bool hit = false;

                // responsible for th Line segments
                for (int i = line.pts_begin;
                     i < line.pts_end - 1;
                     i++) {

                    float2 a = points[i];
                    float2 b = points[i + 1];

                    float2 ab = b - a;
                    float t = dot(p - a, ab) / dot(ab, ab);

                    if (t >= 0.0 && t <= 1.0) {
                        float2 closest = a + t * ab;

                        if (length(p - closest) <= r) {
                            hit = true;
                        }
                    }
                }

                // Last segment back to first
                if (line.closed) {

                    float2 a = points[line.pts_end - 1];
                    float2 b = points[line.pts_begin];

                    float2 ab = b - a;
                    float t = dot(p - a, ab) / dot(ab, ab);

                    if (t >= 0.0 && t <= 1.0) {
                        float2 closest = a + t * ab;

                        if (length(p - closest) <= r) {
                            hit = true;
                        }   
                    }
                }

                // Round joins
                for (int i = line.pts_begin + 1;
                     i < line.pts_end - 1;
                     i++) {

                    if (length(p - points[i]) <= r) {
                        hit = true;
                    }
                }

                // Round joins at first/last point for closed shapes
                if (line.closed) {

                    if (length(p - points[line.pts_begin]) <= r) {
                        hit = true;
                    }

                    if (length(p - points[line.pts_end - 1]) <= r) {
                        hit = true;
                    }
                }

                if (hit) {
                    image[tid] = shape.stroke_rgba;
                }
            }
        }
    }
}

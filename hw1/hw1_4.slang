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

    for (int s = num_shapes - 1; s >= 0; s--) {

        Shape shape = shapes[s];

        float3 object_point = mul(
            shape.world_to_obj,
            float3(p.x, p.y, 1.0)
        );

        float2 object_p = object_point.xy;

        if (shape.type == ShapeType.Circle) {

            Circle circle = circles[shape.index];

            float distance = length(object_p - circle.center);

            if (shape.use_fill && distance <= circle.radius) {
                image[tid] = shape.fill_rgba;
            }

            if (shape.use_stroke &&
                abs(distance - circle.radius) <= shape.stroke_width * 0.5) {
                image[tid] = shape.stroke_rgba;
            }
        }

        else if (shape.type == ShapeType.Polyline) {

            Polyline polyline = polylines[shape.index];

            bool in_fill = false;
            bool in_stroke = false;

            if (shape.use_fill && polyline.closed) {

                int winding = 0;

                for (int i = polyline.pts_begin;
                     i < polyline.pts_end;
                     i++) {

                    int next_i = i + 1;

                    if (next_i >= polyline.pts_end) {
                        next_i = polyline.pts_begin;
                    }

                    float2 A = points[i];
                    float2 B = points[next_i];

                    if (A.y <= object_p.y && B.y > object_p.y) {

                        float cross =
                            (B.x - A.x) * (object_p.y - A.y)
                            - (object_p.x - A.x) * (B.y - A.y);

                        if (cross > 0.0) {
                            winding++;
                        }
                    }

                    else if (A.y > object_p.y && B.y <= object_p.y) {

                        float cross =
                            (B.x - A.x) * (object_p.y - A.y)
                            - (object_p.x - A.x) * (B.y - A.y);

                        if (cross < 0.0) {
                            winding--;
                        }
                    }
                }

                if (winding != 0) {
                    in_fill = true;
                }
            }

            if (shape.use_stroke) {

                float half_width = shape.stroke_width * 0.5;

                for (int i = polyline.pts_begin;
                     i < polyline.pts_end - 1;
                     i++) {

                    float2 A = points[i];
                    float2 B = points[i + 1];

                    float2 AB = B - A;
                    float2 AP = object_p - A;

                    float denom = dot(AB, AB);

                    if (denom > 0.0) {

                        float t = dot(AP, AB) / denom;

                        if (t >= 0.0 && t <= 1.0) {
                            float2 closest = A + t * AB;
                            float distance = length(object_p - closest);

                            if (distance <= half_width) {
                                in_stroke = true;
                            }
                        }
                    }
                }

                if (polyline.closed &&
                    polyline.pts_end - polyline.pts_begin > 1) {

                    float2 A = points[polyline.pts_end - 1];
                    float2 B = points[polyline.pts_begin];

                    float2 AB = B - A;
                    float2 AP = object_p - A;

                    float denom = dot(AB, AB);

                    if (denom > 0.0) {

                        float t = dot(AP, AB) / denom;

                        if (t >= 0.0 && t <= 1.0) {
                            float2 closest = A + t * AB;
                            float distance = length(object_p - closest);

                            if (distance <= half_width) {
                                in_stroke = true;
                            }
                        }
                    }
                }

                for (int i = polyline.pts_begin + 1;
                     i < polyline.pts_end - 1;
                     i++) {

                    float2 vertex = points[i];

                    if (length(object_p - vertex) <= half_width) {
                        in_stroke = true;
                    }
                }

                if (polyline.closed &&
                    polyline.pts_end - polyline.pts_begin > 1) {

                    float2 first_point = points[polyline.pts_begin];
                    float2 last_point = points[polyline.pts_end - 1];

                    if (length(object_p - first_point) <= half_width) {
                        in_stroke = true;
                    }

                    if (length(object_p - last_point) <= half_width) {
                        in_stroke = true;
                    }
                }
            }

            if (in_fill) {
                image[tid] = shape.fill_rgba;
            }

            if (in_stroke) {
                image[tid] = shape.stroke_rgba;
            }
        }
    }
}
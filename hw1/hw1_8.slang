import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> quadratic_beziers,
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // TODO: your code here

    //Change to accumulated color to sample 
    float3 accumulated_color = float3(0.0);
                       
    // Loop through all the samples in the pixel 
    for (int sx = 0; sx < samples_per_axis; sx++) {
        for (int sy = 0; sy < samples_per_axis; sy++) {
            float4 sample_color = float4(background, 1.0);

            // calculate center of each subpixel
            float2 p = float2(
                tid.x + (sx + 0.5) / samples_per_axis,
                height - (tid.y + (sy + 0.5) / samples_per_axis)
            );

            //Loop through all the shapes 
            for (int s = num_shapes - 1; s >= 0; s--) {

                Shape shape = shapes[s];

                float3 object_point = mul(
                    shape.world_to_obj,
                    float3(p.x, p.y, 1.0)
                );

                float2 object_p = object_point.xy;

                // Circle portion
                if (shape.type == ShapeType.Circle) {

                    Circle circle = circles[shape.index];

                    float distance = length(object_p - circle.center);

                    if (shape.use_fill && distance <= circle.radius) {
                        
                        //get the alpha and blend the colors
                        float alpha = shape.fill_rgba.a;
                        sample_color.rgb = alpha * shape.fill_rgba.rgb + (1.0 - alpha) * sample_color.rgb;
                    }

                    if (shape.use_stroke &&
                        abs(distance - circle.radius) <= shape.stroke_width * 0.5) {
                        
                        //get the alpha and blend the colors
                        float alpha = shape.stroke_rgba.a;
                        sample_color.rgb = alpha * shape.stroke_rgba.rgb + (1.0 - alpha) * sample_color.rgb;
                    }
                }

                // Polyline portion
                else if (shape.type == ShapeType.Polyline) {

                    Polyline polyline = polylines[shape.index];

                    bool in_fill = false;
                    bool in_stroke = false;

                    // Fill
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

                    // Stroke
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

                        // Closing segment, connect last point to first point if polyline is closed
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
                        
                        // Check vertices for round joins
                        for (int i = polyline.pts_begin + 1;
                            i < polyline.pts_end - 1;
                            i++) {

                            float2 vertex = points[i];

                            if (length(object_p - vertex) <= half_width) {
                                in_stroke = true;
                            }
                        }

                        // round joins when the polyline is closed
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

                    // Calculate the sample color based on fill and stroke
                    if (in_fill) {
                        //get the alpha and blend the colors
                        float alpha = shape.fill_rgba.a;
                        sample_color.rgb = alpha * shape.fill_rgba.rgb + (1.0 - alpha) * sample_color.rgb;
                    }

                    if (in_stroke) {
                        //get the alpha and blend the colors
                        float alpha = shape.stroke_rgba.a;
                        sample_color.rgb = alpha * shape.stroke_rgba.rgb + (1.0 - alpha) * sample_color.rgb;
                    }
                }

                //Quadratic Bezier portion
                else if (shape.type == ShapeType.QuadraticBezier) {
                    QuadraticBezier curve = quadratic_beziers[shape.index];
                    
                    bool in_fill = false;
                    bool in_stroke = false;

                    //Fill 

                    //Stroke

                    //Blend the fill and stroke colors with the sample color
                    // Calculate the sample color based on fill and stroke
                    if (in_fill) {
                        //get the alpha and blend the colors
                        float alpha = shape.fill_rgba.a;
                        sample_color.rgb = alpha * shape.fill_rgba.rgb + (1.0 - alpha) * sample_color.rgb;
                    }

                    if (in_stroke) {
                        //get the alpha and blend the colors
                        float alpha = shape.stroke_rgba.a;
                        sample_color.rgb = alpha * shape.stroke_rgba.rgb + (1.0 - alpha) * sample_color.rgb;
                    }
                }
            }
            // Accumulate the sample color 
            accumulated_color += sample_color.rgb;
        }
    }

    // get the average color of all the samples 
    float samples_count = float(samples_per_axis * samples_per_axis);
    accumulated_color /= samples_count;

    //paint the pixel 
    image[tid] = float4(accumulated_color, 1.0);
}

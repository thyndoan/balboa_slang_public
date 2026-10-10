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
                    if (shape.use_fill && curve.closed) {

                        int winding = 0;

                        // Walk through all the quadratic bezier segments
                        for (int i = curve.pts_begin; i < curve.pts_end; i += 2){
                            float2 p0 = points[i];
                            float2 p1 = points[i + 1];

                            int i2 = (i + 2);
                            if (i2 >= curve.pts_end) {
                                i2 = curve.pts_begin;
                            }
                            float2 p2 = points[i2]; // wrap around to the first point if closed

                            // Solve the quadratic equation where q_y(t) = object_p.y
                            float a = p0.y - 2 * p1.y + p2.y;
                            float b = 2 * (p1.y - p0.y);
                            float c = p0.y - object_p.y;

                            // roots returned by the quadratic solver
                            float2 roots;

                            // if we find the roots, we check if they are in range [0,1]
                            if (solve_quadratic(a, b, c, roots)) {
                                for (int j = 0; j < 2; j++) {
                                    
                                    // skip duplicate root
                                    if (j == 0 && roots.x == roots.y) {
                                        continue; 
                                    }

                                    // check both roots values
                                    float t = (j == 0) ? roots.x : roots.y;

                                    //if the root is not in the range [0,1], we skip it
                                    if (t < 0.0 || t > 1.0) {
                                        continue;
                                    }

                                    // calculate the curve at t value to get the intersection point
                                    float2 q = (1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2;
                                    
                                    // only check the intersection if it's on the right side of the point
                                    if (q.x > object_p.x) {
                                        // calculate the derivative of the bezier curve at t to determine the direction of the curve
                                        float dy = 2*(1 - t) * (p1.y - p0.y) + 2 * t * (p2.y - p1.y);

                                        // upward crossing
                                        if (dy > 0) {
                                            winding++;
                                        }
                                        //downward crossing 
                                        else if (dy < 0) {
                                            winding--;
                                        }
                                    }
                                }
                            }
                        }
                        if (winding != 0) {
                            in_fill = true;
                        } 
                    }

                    //Stroke
                    if (shape.use_stroke) {
                        float half_width = shape.stroke_width * 0.5;

                        // Number of curve pieces: an open path with n points has (n - 1) / 2,
                        // and a closed path has n / 2 

                        int n = curve.pts_end - curve.pts_begin;
                        int num_pieces = curve.closed ? n / 2 : (n - 1) / 2;
                        
                        // Walk through all the quadratic bezier segments
                        for (int k = 0; k < num_pieces; k++) {

                            int i = curve.pts_begin + 2 * k;

                            // The third point wraps to the first point at the end of a closed path
                            int i2 = i + 2;
                            if (i2 >= curve.pts_end) {
                                i2 = curve.pts_begin;
                            }

                            float2 p0 = points[i];
                            float2 p1 = points[i + 1];
                            float2 p2 = points[i2];

                            // q(t) = A t^2 + B t + C
                            float2 A = p0 - 2.0 * p1 + p2;
                            float2 B = 2.0 * (p1 - p0);
                            float2 C = p0;
                            float2 D = C - object_p;   
                            
                            // the coefficients of (q(t) - object_p) * q'(t) = 0, the closest point on the curve to the object point
                            float c3 = 2 * dot(A, A);
                            float c2 = 3 * dot(A, B);
                            float c1 = dot(B, B) + 2 * dot(A, D);
                            float c0 = dot(B, D);

                            float3 roots;
                            int count = solve_cubic(c3, c2, c1, c0, roots);

                            for (int r = 0; r < count; r++) {

                                float t = roots[r];

                                // skip t unless it is in [0, 1]
                                if (t < 0.0 || t > 1.0) {
                                    continue;
                                }

                                // Calculate the point on the curve at t
                                float2 q = A * t * t + B * t + C;

                                // if the distance from object_p to q is at most half_width,
                                // the sample is in the stroke
                                if (length(object_p - q) <= half_width) {
                                    in_stroke = true;
                                }
                            }
                        }
                        // Round joins
                        for (int i = 0; i < num_pieces; i++) {
                            //leave the free end flat
                            if (!curve.closed && i == 0) {
                                continue;
                            }
                            float2 joint = points[curve.pts_begin + 2 * i];
                            if (length(object_p - joint) <= half_width) {
                                in_stroke = true;
                            }
                        }
                    }
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

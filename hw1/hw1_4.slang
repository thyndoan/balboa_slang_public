// Import shared shape and geometry definitions.
import hw1_common;

// Declare a compute shader.
[shader("compute")]

// Each thread group contains 8 x 8 threads.
[numthreads(8, 8, 1)]

void compute_main(
    // Current GPU thread ID; each thread handles one pixel.
    uint2 tid : SV_DispatchThreadID,

    // Scene background color.
    uniform float3 background,

    // Number of samples per axis.
    // Currently ignored by this implementation.
    uniform int /*samples_per_axis*/,

    // Shape descriptions.
    StructuredBuffer<Shape> shapes,

    // Total number of shapes.
    uniform int num_shapes,

    // Circle geometry.
    StructuredBuffer<Circle> circles,

    // Polyline geometry.
    StructuredBuffer<Polyline> polylines,

    // Quadratic Bezier geometry, currently ignored.
    StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/,

    // Shared polyline point buffer.
    StructuredBuffer<float2> points,

    // Output image.
    RWTexture2D<float4> image
) {

    // Get the image dimensions.
    uint width, height;
    image.GetDimensions(width, height);

    // Ignore threads outside the image.
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Set the initial pixel color to the scene background.
    image[tid] = float4(background, 1.0);
<<<<<<< HEAD

    // Calculate the pixel center and flip the y-coordinate.
=======
        
    // TODO: your code here
>>>>>>> main
    float2 p = float2(
        tid.x + 0.5,
        height - (tid.y + 0.5)
    );

    // Process all shapes in reverse order.
    for (int s = num_shapes - 1; s >= 0; s--) {

        // Get the current shape.
        Shape shape = shapes[s];

        // -----------------------------------------
        // Transformation
        // -----------------------------------------

        // Convert the pixel coordinate into homogeneous
        // coordinates by adding a third component equal to 1.
        //
        // Multiply by world_to_obj to convert the point
        // from world space into the shape's object space.
        //
        // world_to_obj is the inverse of the transformation
        // applied to the original shape.
        float3 object_point = mul(
            shape.world_to_obj,
            float3(p.x, p.y, 1.0)
        );

        // We only need the x and y coordinates for
        // the circle and polyline calculations.
        float2 object_p = object_point.xy;

<<<<<<< HEAD
        // -----------------------------------------
        // Circle rendering
        // -----------------------------------------

=======
        // Circle portion
>>>>>>> main
        if (shape.type == ShapeType.Circle) {

            // Retrieve the circle associated with this shape.
            Circle circle = circles[shape.index];

            // Calculate the distance from the transformed pixel
            // to the circle center.
            float distance = length(object_p - circle.center);

            // Check whether the transformed pixel is inside
            // the circle.
            if (shape.use_fill && distance <= circle.radius) {
                image[tid] = shape.fill_rgba;
            }

            // Check whether the transformed pixel is close
            // enough to the circle boundary to belong to its stroke.
            if (shape.use_stroke &&
                abs(distance - circle.radius) <= shape.stroke_width * 0.5) {
                image[tid] = shape.stroke_rgba;
            }
        }

<<<<<<< HEAD
        // -----------------------------------------
        // Polyline rendering
        // -----------------------------------------

=======
        // Polyline portion
>>>>>>> main
        else if (shape.type == ShapeType.Polyline) {

            // Retrieve the polyline associated with this shape.
            Polyline polyline = polylines[shape.index];

            // These variables track whether the transformed pixel
            // belongs to the fill or stroke.
            bool in_fill = false;
            bool in_stroke = false;

<<<<<<< HEAD
            // -----------------------------------------
            // Fill
            // -----------------------------------------

            // Only closed polylines with fill enabled can be filled.
=======
            // Fill
>>>>>>> main
            if (shape.use_fill && polyline.closed) {

                int winding = 0;

                // Check all edges of the polygon.
                for (int i = polyline.pts_begin;
                     i < polyline.pts_end;
                     i++) {

                    // Find the next point.
                    int next_i = i + 1;

                    // Connect the final point back to the first.
                    if (next_i >= polyline.pts_end) {
                        next_i = polyline.pts_begin;
                    }

                    // Get the edge endpoints.
                    float2 A = points[i];
                    float2 B = points[next_i];

                    // Check for an upward crossing of the
                    // horizontal ray through object_p.
                    if (A.y <= object_p.y && B.y > object_p.y) {

                        // Calculate the cross product.
                        float cross =
                            (B.x - A.x) * (object_p.y - A.y)
                            - (object_p.x - A.x) * (B.y - A.y);

                        if (cross > 0.0) {
                            winding++;
                        }
                    }

                    // Check for a downward crossing.
                    else if (A.y > object_p.y && B.y <= object_p.y) {

                        float cross =
                            (B.x - A.x) * (object_p.y - A.y)
                            - (object_p.x - A.x) * (B.y - A.y);

                        if (cross < 0.0) {
                            winding--;
                        }
                    }
                }

                // Nonzero winding means the pixel is inside.
                if (winding != 0) {
                    in_fill = true;
                }
            }

<<<<<<< HEAD
            // -----------------------------------------
            // Stroke
            // -----------------------------------------

=======
            // Stroke
>>>>>>> main
            if (shape.use_stroke) {

                // Half of the stroke width.
                float half_width = shape.stroke_width * 0.5;

                // Initially assume the pixel is not on the stroke.
                bool in_stroke = false;

                // Check each segment between consecutive points.
                for (int i = polyline.pts_begin;
                     i < polyline.pts_end - 1;
                     i++) {

                    float2 A = points[i];
                    float2 B = points[i + 1];

                    // Segment direction vector.
                    float2 AB = B - A;

                    // Vector from the segment start to the pixel.
                    float2 AP = object_p - A;

                    // Squared length of the segment.
                    float denom = dot(AB, AB);

                    // Avoid division by zero for a zero-length segment.
                    if (denom > 0.0) {

                        // Project the pixel onto the line segment.
                        float t = dot(AP, AB) / denom;

                        // Only use projections between the endpoints.
                        if (t >= 0.0 && t <= 1.0) {

                            // Closest point on the segment.
                            float2 closest = A + t * AB;

                            // Distance from the pixel to the segment.
                            float distance = length(object_p - closest);

                            // If close enough, this pixel belongs
                            // to the stroke.
                            if (distance <= half_width) {
                                in_stroke = true;
                            }
                        }
                    }
                }

<<<<<<< HEAD
                // If the polyline is closed, check the final
                // segment connecting the last point to the first.
=======
                // Closing segment, connect last point to first point if polyline is closed
>>>>>>> main
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

<<<<<<< HEAD
                // Check interior vertices to produce round joins.
=======
                // Check vertices for round joins
>>>>>>> main
                for (int i = polyline.pts_begin + 1;
                     i < polyline.pts_end - 1;
                     i++) {

                    float2 vertex = points[i];

                    if (length(object_p - vertex) <= half_width) {
                        in_stroke = true;
                    }
                }

<<<<<<< HEAD
                // Check the first and last vertices for
                // round joins on closed polylines.
=======
                // round joins when the polyline is closed
>>>>>>> main
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

<<<<<<< HEAD
            // -----------------------------------------
            // Painting
            // -----------------------------------------

            // Draw the fill first.
=======
            // Painting 
>>>>>>> main
            if (in_fill) {
                image[tid] = shape.fill_rgba;
            }

            // Draw the stroke second so it appears above the fill.
            if (in_stroke) {
                image[tid] = shape.stroke_rgba;
            }
        }
    }
}
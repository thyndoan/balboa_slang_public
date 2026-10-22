// Import the shared definitions of Shape, Circle, Polyline,
// QuadraticBezier, and ShapeType.
import hw1_common;

// Declare a compute shader.
[shader("compute")]

// Each thread group contains 8 x 8 threads.
[numthreads(8, 8, 1)]

void compute_main(
    // Each GPU thread processes one pixel.
    uint2 tid : SV_DispatchThreadID,

    // Background RGB color of the scene.
    uniform float3 background,

    // Number of samples per axis.
    // This parameter is currently ignored in this implementation.
    uniform int /*samples_per_axis*/,

    // Buffer containing the descriptions of all shapes.
    StructuredBuffer<Shape> shapes,

    // Total number of shapes in the scene.
    uniform int num_shapes,

    // Buffer containing the actual circle geometry.
    StructuredBuffer<Circle> circles,

    // Buffer containing the actual polyline geometry.
    StructuredBuffer<Polyline> polylines,

    // Buffer for quadratic Bezier curves.
    // These are currently ignored.
    StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/,

    // Shared buffer containing the points used by polylines.
    StructuredBuffer<float2> points,

    // Output image.
    RWTexture2D<float4> image
) {

    // Get the dimensions of the output image.
    uint width, height;
    image.GetDimensions(width, height);

    // Stop threads that are outside the image.
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Initialize this pixel using the scene background color.
    image[tid] = float4(background, 1.0);

    // Calculate the pixel center.
    // Flip the y-coordinate to use a bottom-left origin.
    float2 p = float2(
        tid.x + 0.5,
        height - (tid.y + 0.5)
    );

    // Process shapes in reverse order.
    //
    // The scene stores shapes in their original order.
    // Rendering in reverse allows earlier shapes in the scene
    // to be drawn on top of later shapes.
    for (int s = num_shapes - 1; s >= 0; s--) {

        // Retrieve the current shape description.
        Shape shape = shapes[s];


        // !!CIRCLE RENDERING!!
        // Check whether this shape is a circle.
        if (shape.type == ShapeType.Circle) {

            // Use shape.index to find the actual circle data.
            Circle c = circles[shape.index];

            // Calculate the distance from the pixel center
            // to the circle center.
            float d = length(p - c.center);

            // Check whether the pixel is inside the circle.
            // If fill is enabled, use the fill color.
            if (shape.use_fill &&
                d <= c.radius) {

                image[tid] = shape.fill_rgba;
            }

            // Check whether the pixel is close enough to
            // the circle boundary to belong to the stroke.
            //
            // abs(d - radius) measures the distance from
            // the pixel to the circle's circumference.
            if (shape.use_stroke && abs(d - c.radius) <= shape.stroke_width * 0.5) {

                image[tid] = shape.stroke_rgba;
            }
        }

        // -----------------------------
        // Polyline rendering
        // -----------------------------

        // If the shape is not a circle, check whether it is a polyline.
        else if (shape.type == ShapeType.Polyline) {

            // Retrieve the polyline information.
            // Its index identifies the correct entry
            // in the polylines buffer.
            Polyline line = polylines[shape.index];

            // -----------------------------
            // Fill
            // -----------------------------

            // Only closed polylines can be filled.
            if (shape.use_fill && line.closed) {

                // Tracks upward and downward crossings.
                int winding = 0;

                // Iterate through the points belonging
                // to this particular polyline.
                for (int i = line.pts_begin; i < line.pts_end; i++) {

                    // Get the next point's index.
                    int j = i + 1;

                    // If this is the last point,
                    // wrap around to the first point.
                    if (j >= line.pts_end) {
                        j = line.pts_begin;
                    }

                    // Get the current edge's endpoints.
                    float2 a = points[i];
                    float2 b = points[j];

                    // Check whether this edge crosses
                    // the horizontal ray upward.
                    if (a.y <= p.y && b.y > p.y) {

                        // Use the cross product to determine which side of the edge the pixel is on.
                        float cross =
                            (b.x - a.x) * (p.y - a.y)
                            - (p.x - a.x) * (b.y - a.y);

                        // Positive cross means an upward crossing
                        // contributes +1.
                        if (cross > 0) {
                            winding++;
                        }

                    // Check whether this edge crosses
                    // the horizontal ray downward.
                    } else if (a.y > p.y && b.y <= p.y) {

                        float cross =
                            (b.x - a.x) * (p.y - a.y)
                            - (p.x - a.x) * (b.y - a.y);

                        // Negative cross means a downward crossing
                        // contributes -1.
                        if (cross < 0) {
                            winding--;
                        }
                    }
                }

                // A nonzero winding number means
                // the pixel is inside the polygon.
                if (winding != 0) {
                    image[tid] = shape.fill_rgba;
                }
            }

            // -----------------------------
            // Stroke
            // -----------------------------

            // Only calculate the stroke if it is enabled.
            if (shape.use_stroke) {

                // The stroke extends half its width
                // on each side of the center line.
                float r = shape.stroke_width * 0.5;

                // Initially assume the pixel is not part
                // of the stroke.
                bool hit = false;

                // Check each segment between consecutive points.
                for (int i = line.pts_begin; i < line.pts_end - 1; i++) {

                    // Get the two endpoints of the segment.
                    float2 a = points[i];
                    float2 b = points[i + 1];

                    // Vector representing the segment.
                    float2 ab = b - a;

                    // Project the pixel onto the segment.
                    // The denominator is the squared segment length.
                    float t = dot(p - a, ab) / dot(ab, ab);

                    // Make sure the projected point is
                    // between the two endpoints.
                    if (t >= 0.0 && t <= 1.0) {

                        // Calculate the closest point.
                        float2 closest = a + t * ab;

                        // If the pixel is within half the stroke width of the segment,
                        // it is part of the stroke.
                        if (length(p - closest) <= r) {
                            hit = true;
                        }
                    }
                }

                // If the polyline is closed, also check
                // the segment connecting the last point to the first.
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

                // Check interior vertices to create round joins.
                for (int i = line.pts_begin + 1; i < line.pts_end - 1; i++) {

                    // If the pixel is within radius r of a vertex,
                    // include it in the stroke.
                    if (length(p - points[i]) <= r) {
                        hit = true;
                    }
                }

                // Closed polylines also have a join
                // where the last and first points meet.
                if (line.closed) {

                    if (length(p - points[line.pts_begin]) <= r) {
                        hit = true;
                    }

                    if (length(p - points[line.pts_end - 1]) <= r) {
                        hit = true;
                    }
                }

                // If any segment or join contains the pixel,
                // draw the stroke color.
                if (hit) {
                    image[tid] = shape.stroke_rgba;
                }
            }
        }
    }
}
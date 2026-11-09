import hw3_common;

// Check whether a ray intersects a sphere.
// Return the closest valid intersection between t_min and t_max.
bool intersect_sphere(
    float3 ray_origin,
    float3 ray_dir,
    Sphere sphere,
    float t_min,
    float t_max,
    out float t_hit
) {
    float3 oc = ray_origin - sphere.center;

    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(ray_dir, oc);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float2 roots;
    t_hit = t_max;

    if (!solve_quadratic(a, b, c, roots)) {
        return false;
    }

    // Check both roots and choose the closest valid one.
    if (roots.x > t_min && roots.x < t_hit) {
        t_hit = roots.x;
    }

    if (roots.y > t_min && roots.y < t_hit) {
        t_hit = roots.y;
    }

    return t_hit < t_max;
}

// Check whether a ray intersects a shape
// Move the ray into the shape's object space with world_to_obj
// Intersect there with the plain sphere.
bool intersect_shape(
    float3 ray_origin,
    float3 ray_dir,
    Shape shape,
    StructuredBuffer<Sphere> spheres,
    //add meshes, vertices, triangles for the triangle mesh intersection
    StructuredBuffer<TriangleMesh> meshes,     
    StructuredBuffer<float3> vertices,         
    StructuredBuffer<uint3> triangles,     
    StructuredBuffer<BVHNode> bvh_nodes,    
    float t_min,
    float t_max,
    out float t_hit,
    out uint hit_tri,
    out float3 hit_bary
) {
    t_hit = t_max;
    hit_tri = 0;
    hit_bary = float3(0.0, 0.0, 0.0);

    float3 obj_origin = xform_pt(shape.world_to_obj, ray_origin);  
    float3 obj_dir = xform_vec(shape.world_to_obj, ray_dir);  

    // if it's a sphere, part 4
    if (shape.type == ShapeType.Sphere) {
        return intersect_sphere(obj_origin, obj_dir, spheres[shape.index],
                            t_min, t_max, t_hit);
    }
    
    // if it's a triangle mesh: use the BVH instead of testing every triangle
    TriangleMesh mesh = meshes[shape.index];
    bool found = false;

    uint node_index = mesh.bvh_root_index;
    uint end_index = bvh_nodes[mesh.bvh_root_index].next_index;  // end of this mesh's nodes

    while (node_index < end_index) {
        BVHNode node = bvh_nodes[node_index];

        // Missed the box -> Skip everything inside it.
        if (!intersect_aabb(node.bbox_min, node.bbox_max,
                            obj_origin, obj_dir, t_min, t_hit)) {
            node_index = node.next_index;
            continue;
        }

        if (node.is_leaf) {
            // Each leaf box hold 1-4 triangles. Loop through them and test for intersection 
            for (uint j = node.triangle_begin; j < node.triangle_end; j++) {
                uint3 tri = triangles[j] + mesh.vertex_offset;
                float3 b;
                float t;

                // test for intersection with the triangle
                if (intersect_tri(vertices[tri.x], vertices[tri.y], vertices[tri.z],
                                  obj_origin, obj_dir, t_min, t_hit, b, t)) {
                    t_hit = t;
                    hit_tri = j;
                    hit_bary = b;
                    found = true;
                }
            }
            node_index = node.next_index;   // done with this leaf, traverse to the next node
        } else {
            node_index = node_index + 1;    // hit an inner box -> look inside
        }
    }
    return found; 
}

// Check whether another sphere blocks the light.
// Using transformed shapes instead  
bool is_in_shadow(
    float3 hit_point,
    float3 light_dir,
    float light_distance,
    StructuredBuffer<Shape> shapes,
    int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,  
    StructuredBuffer<float3> vertices,       
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<BVHNode> bvh_nodes  
) {
    for (int i = 0; i < num_shapes; i++) {

        float shadow_t;
        uint unused_tri;        
        float3 unused_bary;

        if (intersect_shape(
            hit_point,
            light_dir,
            shapes[i],
            spheres,
            meshes,
            vertices,
            triangles,
            bvh_nodes,
            EPSILON,
            light_distance - EPSILON,
            shadow_t,
            unused_tri, unused_bary
        )) {
            return true;
        }
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Calculate the aspect ratio
    float aspect = float(width) / float(height);

    // Get image plane size from fov_x, calculate half height and half width
    float half_width = tan(radians(camera.fov_x) / 2.0);
    float half_height = half_width / aspect;

    // Calculate the pixel center on the image plane, scale to the new plane size
    float x = (2.0 * (float(tid.x) + 0.5) / float(width) - 1.0) * half_width;
    float y = (1.0 - 2.0 * (float(tid.y) + 0.5) / float(height)) * half_height;

    // Initial camera ray, move the camera ray from the camera space to world space
    float3 ray_origin = xform_pt(camera.cam_to_world, float3(0.0, 0.0, 0.0));
    float3 ray_dir = normalize(xform_vec(camera.cam_to_world, float3(x, y, -1.0)));

    // Accumulated color and reflection throughput
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    // Minimum valid intersection distance
    float ray_t_min = 0.0;

    // Trace multiple reflection bounces
    for (int b = 0; b < max_depth; b++) {

        // Find the closest sphere intersection
        float closest_t = 1e30;
        int hit_shape = -1;
        uint hit_tri = 0;
        float3 hit_bary = float3(0.0, 0.0, 0.0);

        for (int i = 0; i < num_shapes; i++) {

            float t;
            uint tri;
            float3 bary;

            if (intersect_shape(ray_origin, ray_dir, shapes[i], spheres, meshes, vertices, triangles, bvh_nodes,
                                ray_t_min, closest_t, t, tri, bary)) {
                closest_t = t;
                hit_shape = i;
                hit_tri= tri;
                hit_bary = bary;
            }
        }

        // If the ray misses, use the background color
        if (hit_shape < 0) {
            color += throughput * background;
            break;
        }

        // Normal
        Shape shape = shapes[hit_shape];
        
        // Calculate intersection point and normal
        float3 hit_point = ray_origin + closest_t * ray_dir;

        float3 obj_normal;

        // same as part 4
        if (shape.type == ShapeType.Sphere) {
            Sphere sphere = spheres[shape.index];
            
            // Compute the plain sphere normal in object space.
            // Bring it to world space with the inverse transpose of obj_to_world
            float3 obj_hit = xform_pt(shape.world_to_obj, hit_point);
            obj_normal = obj_hit - sphere.center;
        }

        // else it's the triangle mesh
        else {
            TriangleMesh mesh = meshes[shape.index];
            uint3 tri = triangles[hit_tri] + mesh.vertex_offset;
            // Compute the interpolated normal in object space using barycentric coordinates
            obj_normal = hit_bary.x * vertex_normals[tri.x]
                        + hit_bary.y * vertex_normals[tri.y]
                        + hit_bary.z * vertex_normals[tri.z];
        }
    
        //then normalize it.
        float3 normal = normalize(xform_vec(transpose(shape.world_to_obj), obj_normal));

        // Get material properties
        float3 Kd = shape.Kd;
        float3 Kr = shape.Kr;

        // Calculate diffuse lighting
        float3 contribution = float3(0.0, 0.0, 0.0);

        for (int i = 0; i < num_lights; i++) {

            PointLight light = lights[i];

            // Direction and distance to the light
            float3 to_light = light.position - hit_point;
            float light_distance = length(to_light);
            float3 light_dir = to_light / light_distance;

            // Check for shadows
            bool shadowed = is_in_shadow(
                hit_point,
                light_dir,
                light_distance,
                shapes,
                num_shapes,
                spheres, meshes, vertices, triangles, bvh_nodes
            );

            if (shadowed) {
                continue;
            }

            // Lambertian diffuse lighting
            float NdotL = max(dot(normal, light_dir), 0.0);

            // Point light attenuation
            contribution += Kd * light.intensity * NdotL /
                            (light_distance * light_distance);
        }

        // Accumulate this bounce's contribution
        color += throughput * contribution;

        // Update throughput for the next reflection
        throughput *= Kr;

        // No more reflection contribution
        if (max(throughput.x, max(throughput.y, throughput.z)) < 1e-6) {
            break;
        }

        // Calculate mirror reflection direction
        ray_dir = normalize(reflect(ray_dir, normal));

        // Start the next ray at the current intersection
        ray_origin = hit_point;

        // Avoid self intersection
        ray_t_min = EPSILON;
    }

    image[tid] = float4(color, 1.0);
}
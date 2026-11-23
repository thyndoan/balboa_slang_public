struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}


[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }

    float3 x = positions[tid];

    // The closest object so far, and its contact plane
    float best_dist = 1e30;
    float4 best_plane = float4(0.0, 0.0, 0.0, 0.0);

    // Check all planes
    for (uint i = 0; i < num_planes; i++) {
        float3 n = planes[i].xyz;
        float d = planes[i].w;
        float dist = dot(n, x) - d;    // (+) means above the plane, (-) means below the plane
        if (dist < best_dist) {
            best_dist = dist;
            best_plane = float4(n, d);
        }
    }

    // Check all spheres
    for (uint i = 0; i < num_spheres; i++) {
        float3 c = spheres[i].xyz;
        float r = spheres[i].w;
        float3 to_x = x - c;
        float len = length(to_x);
        float dist = len - r;          // (+) means outside, (-) means inside
        if (dist < best_dist && len > 1e-6) {
            float3 n = to_x / len;     // from center toward the point
            best_dist = dist;
            best_plane = float4(n, dot(n, c) + r);  // flat plane touching the ball
        }
    }

    // Check the mesh using the BVH.
    // Only look for triangles closer than this distance.
    float max_dist = contact_margin + collision_distance;

    // The closest triangle we found (as a plane)
    bool mesh_found = false;
    float mesh_dist = 0.0;
    float4 mesh_plane = float4(0.0, 0.0, 0.0, 0.0);

    int i = 0;
    while (i >= 0 && i < int(num_bvh_nodes)) {
        BVHNode node = bvh_nodes[i];

        // If box is too far, then skip this whole branch
        if (point_aabb_distance(x, node.bbox_min, node.bbox_max) > max_dist) {
            i = node.next_index;
            continue;
        }

        // If it's not a leaf, then go into the first child (the next node in the list)
        if (!node.is_leaf) {
            i = i + 1;
            continue;
        }

        // If it's a leaf, then check its triangles
        for (uint t = node.triangle_begin; t < node.triangle_end; t++) {
            // The 3 corners of the triangle
            uint3 tri = mesh_triangles[t];
            float3 p0 = mesh_vertices[tri.x];
            float3 p1 = mesh_vertices[tri.y];
            float3 p2 = mesh_vertices[tri.z];

            // Closest point on this triangle to the particle
            ClosestPoint cp = closest_point_triangle(x, p0, p1, p2);
            float unsigned_dist = length(x - cp.position);

            // Farther than the closest triangle so far, then skip
            // (1e-5 lets two triangles that share an edge count as a tie)
            if (unsigned_dist > max_dist + 1e-5) {
                continue;
            }

            // Face normal 
            float3 ng = normalize(cross(p1 - p0, p2 - p0));

            // Smooth normal at the closest point, from the 3 vertex normals
            float3 ns = normalize(cp.bary.x * mesh_vertex_normals[tri.x] +
                                  cp.bary.y * mesh_vertex_normals[tri.y] +
                                  cp.bary.z * mesh_vertex_normals[tri.z]);

            // Flip the face normal so it points outside, like the smooth normal
            if (dot(ng, ns) < 0.0) {
                ng = -ng;
            }

            // Treat the triangle like an infinite plane
            float d = dot(ng, cp.position);
            float dist = dot(ng, x) - d;   // (+) outside, (-) inside

            // Keep this triangle if it's the first one, if it's closer,
            // or if it's a tie and it says more outside
            bool closer = unsigned_dist < max_dist - 1e-5;
            if (!mesh_found || closer || dist > mesh_dist) {
                mesh_found = true;
                mesh_dist = dist;
                mesh_plane = float4(ng, d);
            }
            max_dist = min(max_dist, unsigned_dist);
        }

        // Done with this leaf, go to the next branch
        i = node.next_index;
    }

    // Now compare the closest triangle with the planes and spheres
    if (mesh_found && mesh_dist < best_dist) {
        best_dist = mesh_dist;
        best_plane = mesh_plane;
    }

    // If it's close enough, then save the contact plane (shifted out a little)
    // If it's too far, then save zeros, this means no contact
    if (best_dist < contact_margin + collision_distance) {
        best_plane.w += collision_distance;
        contact_planes[tid] = best_plane;
    } else {
        contact_planes[tid] = float4(0.0, 0.0, 0.0, 0.0);
    }
}

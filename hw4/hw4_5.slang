[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated 
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below 
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
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

    // If it's close enough, then save the contact plane (shifted out a little)
    // If it's too far, then save zeros, this means no contact
    if (best_dist < contact_margin + collision_distance) {
        best_plane.w += collision_distance;
        contact_planes[tid] = best_plane;
    } else {
        contact_planes[tid] = float4(0.0, 0.0, 0.0, 0.0);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         // Contact plane: dot(n, x) = d, with unit normal n
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
    if (tid >= num_particles) {
        return;
    }
    // Current position and inverse mass of this particle
    float3 position = positions_in[tid];
    float wi = inv_masses[tid];

    // Pinned point: don't move it, but still copy it to the output
    if (wi == 0.0) {
        positions_out[tid] = position;
        return;
    }

    float3 total_correction = float3(0.0, 0.0, 0.0);
    int neighbor_count = 0;

    // CSR lookup, go through all neighbors of this particle
    for (uint k = row_ptr[tid]; k < row_ptr[tid + 1]; k++) {
        uint neighbor_tid = neighbors[k];
        float3 d = positions_in[neighbor_tid] - position;
        float distance = length(d);
        float wj = inv_masses[neighbor_tid];

        // The distance we want between this particle and its neighbor
        float rest_length = neighbor_rest[k];

        // Distance constraint correction (Eq. 1):
        // Move toward or away from the neighbor
        // Skip if the particles overlap (can't divide by zero)
        if (distance > 1e-6 && wi + wj > 0.0) {
            total_correction += (wi / (wi + wj)) * (distance - rest_length) * (d / distance);
        }

        neighbor_count++;
    }

    // Jacobi update. Move by the average of all edge corrections
    if (neighbor_count > 0) {
        position += total_correction / float(neighbor_count);
    }

    // Contact with an object (either plane or sphere)
    float4 plane = contact_planes[tid];
    float3 n = plane.xyz;

    // n = (0,0,0) means no contact
    if (dot(n, n) > 0.0) {
        // How far inside the contact plane this particle is (negative = inside)
        float penetration = dot(n, position) - plane.w;

        // If it's inside, then push it out and apply friction
        if (penetration < 0.0) {
            float depth = -penetration; // make it a positive number  

            // Push out to the surface
            position += depth * n;

            // Apply friction: how much did this particle slide sideways
            float3 move = position - old_positions[tid];
            float3 tangent = move - dot(move, n) * n;
            float tangent_len = length(tangent);

            if (tangent_len <= static_friction * depth) {
                // If it's small slide => this is static friction, stop it completely
                position -= tangent;
            } else {
                // If it's big slide => this is kinetic friction, slow it down a bit
                position -= (tangent / tangent_len) * (kinetic_friction * depth);
            }
        }
    }

    // Save the new position
    positions_out[tid] = position;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // Pinned point never moves, so its speed stays 0, we skip it
    if (inv_masses[tid] == 0.0) {
        return;
    }

    // Speed = how far the particle moved / time
    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}

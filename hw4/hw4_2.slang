
[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    float3 position = positions[tid];
    old_positions[tid] = position;

    if (inv_masses[tid] == 0.0) {
        return;
    }

    float3 gravity = float3(0.0, -9.8, 0.0);
    float3 velocity = velocities[tid];
    velocity += gravity * dt;
    position += velocity * dt;

    positions[tid] = position;
    velocities[tid] = velocity;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }

    float3 position = positions_in[tid];
    float wi = inv_masses[tid];

    if (wi == 0.0) {
        positions_out[tid] = position;
        return;
    }

    float3 total_correction = float3(0.0, 0.0, 0.0);
    uint neighbor_count = 0;

    if (tid > 0) {
        uint neighbor = tid - 1;
        float3 d = positions_in[neighbor] - position;
        float distance = length(d);
        float wj = inv_masses[neighbor];

        if (distance > 1e-6 && wi + wj > 0.0) {
            total_correction += (wi / (wi + wj)) * (distance - rest_lengths[neighbor]) * (d / distance);
        }
        neighbor_count++;
    }

    if (tid + 1 < num_particles) {
        uint neighbor = tid + 1;
        float3 d = positions_in[neighbor] - position;
        float distance = length(d);
        float wj = inv_masses[neighbor];

        if (distance > 1e-6 && wi + wj > 0.0) {
            total_correction += (wi / (wi + wj)) * (distance - rest_lengths[tid]) * (d / distance);
        }
        neighbor_count++;
    }

    if (neighbor_count > 0) {
        position += total_correction / float(neighbor_count);
    }

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
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    if (inv_masses[tid] == 0.0) {
        return;
    }

    float3 position = positions[tid];

    if (position.y < -0.97) {
        position.y = -0.97;
    }

    positions[tid] = position;
    velocities[tid] = (position - old_positions[tid]) / dt;
}

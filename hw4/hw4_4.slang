[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
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

    // Save the new position
    positions_out[tid] = position;
}

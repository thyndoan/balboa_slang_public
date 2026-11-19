
[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }

    int2 grid_position = int2(int(tid % uint(grid_size.x)), int(tid / uint(grid_size.x)));
    float3 position = positions_in[tid];
    float wi = inv_masses[tid];

    if (wi == 0.0) {
        positions_out[tid] = position;
        return;
    }

    float3 total_correction = float3(0.0, 0.0, 0.0);
    int neighbor_count = 0;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            bool structural = (abs(dx) == 1 && dy == 0) || (dx == 0 && abs(dy) == 1);
            bool shearing = abs(dx) == 1 && abs(dy) == 1;
            bool bending = (abs(dx) == 2 && dy == 0) || (dx == 0 && abs(dy) == 2);

            if (!structural && !shearing && !bending) {
                continue;
            }

            int2 neighbor_grid_position = grid_position + int2(dx, dy);
            if (neighbor_grid_position.x < 0 || neighbor_grid_position.x >= grid_size.x ||
                neighbor_grid_position.y < 0 || neighbor_grid_position.y >= grid_size.y) {
                continue;
            }

            uint neighbor_tid = uint(neighbor_grid_position.y * grid_size.x + neighbor_grid_position.x);
            float3 d = positions_in[neighbor_tid] - position;
            float distance = length(d);
            float wj = inv_masses[neighbor_tid];

            float rest_length;
            if (structural) {
                rest_length = (dx != 0) ? spacing.x : spacing.y;
            } else if (shearing) {
                rest_length = length(spacing);
            } else {
                rest_length = (dx != 0) ? 2.0 * spacing.x : 2.0 * spacing.y;
            }

            if (distance > 1e-6 && wi + wj > 0.0) {
                total_correction += (wi / (wi + wj)) * (distance - rest_length) * (d / distance);
            }

            neighbor_count++;
        }
    }

    if (neighbor_count > 0) {
        position += total_correction / float(neighbor_count);
    }

    positions_out[tid] = position;
}

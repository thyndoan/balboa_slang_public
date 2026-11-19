
[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    float3 gravity = float3(0.0, -9.8, 0.0);
    float3 velocity = velocities[tid];
    float3 position = positions[tid];

    velocity += gravity * dt;
    position += velocity * dt;

    if (position.y < -0.97) {
        position.y = -0.97;
        velocity.y = -velocity.y;
    }

    positions[tid] = position;
    velocities[tid] = velocity;
}

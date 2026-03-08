extern "C" __global__ void hmc_stdnormal_leapfrog_diag(
    const double* q_in,
    const double* p_in,
    const double* inv_mass,
    const double eps,
    const int n,
    double* q_out,
    double* p_out
) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }

    const double q0 = q_in[idx];
    const double p0 = p_in[idx];
    const double inv_m = inv_mass[idx];

    const double p_half = p0 - 0.5 * eps * q0;
    const double q1 = q0 + eps * inv_m * p_half;
    const double p1 = p_half - 0.5 * eps * q1;

    q_out[idx] = q1;
    p_out[idx] = p1;
}

extern "C" __global__ void hmc_stdnormal_log_joint_diag(
    const double* q,
    const double* p,
    const double* inv_mass,
    const int n,
    double* block_out
) {
    extern __shared__ double scratch[];

    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    double local = 0.0;
    if (idx < n) {
        const double qi = q[idx];
        const double pi = p[idx];
        local = qi * qi + pi * pi * inv_mass[idx];
    }

    scratch[threadIdx.x] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_out[blockIdx.x] = -0.5 * scratch[0];
    }
}

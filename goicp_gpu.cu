// =============================================================================
// goicp_gpu.cu  —  Method A CUDA implementation
//
// Three kernels:
//   1. rotate_points_kernel     — 9-MAD rotation for K×N points
//   2. eval_trans_children      — wavefront: expand + evaluate 8 children
//                                  per active trans node per outer node
//   3. collect_results_kernel   — gather ub/best_t per outer node
//
// Wavefront loop (host-side):
//   while any outer node has active trans nodes:
//     launch eval_trans_children  → produces child (ub, lb) for 8×active nodes
//     update d_ub_best, d_lb_best atomically within kernel
//     swap cur/nxt pools
//
// Synchronization minimized:
//   - cudaDeviceSynchronize only after eval_trans_children (mandatory: next
//     iteration reads results)
//   - check_any_active replaced by reading d_active_count_nxt on host after
//     zero_next_counts + eval_trans_children; the sync after eval already
//     guarantees counts are visible
//   - d_flag eliminated from the steady-state loop
//   - d_R pre-allocated in GoICPGpu (no malloc/free per batch)
// =============================================================================

#include "goicp_gpu.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(1);                                                     \
    }                                                                \
} while(0)

// ---------------------------------------------------------------------------
// Constant memory for DT descriptor (fits easily: struct is ~24 bytes)
// Must be declared at file scope, not inside a function.
// ---------------------------------------------------------------------------
__constant__ GpuDT c_dt;

// ---------------------------------------------------------------------------
// DT lookup — mirrors DT3D::Distance() exactly
//
// CPU code:
//   x = ROUND((_x - xMin) * scale)   clamp to [0, SIZE-1]
//   return A.data[z][y][x].distance + extra_if_oob
//
// We reproduce the same clamped-nearest-cell lookup.
// For out-of-bounds: add sqrt(a^2+b^2+c^2)/scale exactly as CPU does.
// This makes GPU and CPU return identical values for identical inputs.
// ---------------------------------------------------------------------------
__device__ __forceinline__
float dt_lookup(float qx, float qy, float qz)
{
    // Convert world coords to grid indices (nearest cell)
    float fx = (qx - c_dt.xMin) * c_dt.scale;
    float fy = (qy - c_dt.yMin) * c_dt.scale;
    float fz = (qz - c_dt.zMin) * c_dt.scale;

    int ix = __float2int_rn(fx);  // ROUND = round-to-nearest
    int iy = __float2int_rn(fy);
    int iz = __float2int_rn(fz);

    // Track out-of-bounds displacement (mirrors CPU extrapolation)
    float ex = 0.f, ey = 0.f, ez = 0.f;

    if (ix < 0)            { ex = (float)ix;              ix = 0; }
    else if (ix >= c_dt.size) { ex = (float)(ix - c_dt.size + 1); ix = c_dt.size - 1; }

    if (iy < 0)            { ey = (float)iy;              iy = 0; }
    else if (iy >= c_dt.size) { ey = (float)(iy - c_dt.size + 1); iy = c_dt.size - 1; }

    if (iz < 0)            { ez = (float)iz;              iz = 0; }
    else if (iz >= c_dt.size) { ez = (float)(iz - c_dt.size + 1); iz = c_dt.size - 1; }

    // Use hardware texture lookup (point filtering acts as round-to-nearest)
    // Note: unnormalized coordinates access texel at floor(x), so x+0.5f = round(x)
    float dist = tex3D<float>(c_dt.tex_dist, ix + 0.5f, iy + 0.5f, iz + 0.5f);

    // Add OOB penalty exactly as CPU: sqrt(ex^2+ey^2+ez^2) / scale
    float oob = ex*ex + ey*ey + ez*ez;
    if (oob > 0.f)
        dist += sqrtf(oob) / c_dt.scale;

    return dist;
}

// ---------------------------------------------------------------------------
// Warp reduction — sum of 32 floats via shuffle
// ---------------------------------------------------------------------------
__device__ __forceinline__
float warp_reduce_sum(float v)
{
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ---------------------------------------------------------------------------
// Kernel 1: rotate_points_kernel
//
// Grid:  (K, ceil(Nd/256))   — K outer nodes × point tiles
// Block: (256, 1)
//
// Reads:  d_pData[i*3 + coord]
// Writes: d_pDataTemp[k*Nd*3 + i*3 + coord]   — coalesced in i
// ---------------------------------------------------------------------------
__global__ void rotate_points_kernel(
    const float* __restrict__ d_pData,   // [Nd*3]
    const float* __restrict__ d_R,       // [K*9]  row-major per node
    float*       d_pDataTemp,            // [K*Nd*3]
    int Nd)
{
    int k = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= Nd) return;

    const float* R = d_R + k * 9;
    float px = d_pData[i*3 + 0];
    float py = d_pData[i*3 + 1];
    float pz = d_pData[i*3 + 2];

    float* outX = d_pDataTemp + k * Nd * 3 + i;
    float* outY = outX + Nd;
    float* outZ = outY + Nd;
    *outX = R[0]*px + R[1]*py + R[2]*pz;
    *outY = R[3]*px + R[4]*py + R[5]*pz;
    *outZ = R[6]*px + R[7]*py + R[8]*pz;
}

// ---------------------------------------------------------------------------
// Kernel 2: eval_trans_children
//
// For each (outer_node k, active_trans_slot s), expands 8 children and
// evaluates ub/lb for each child via parallel reduction over N data points.
//
// Grid:  (K, MAX_ACTIVE_THIS_LEVEL, 8)
//         — k=outer node, s=parent trans slot, j=child index
// Block: (BLOCK_N, 1)  where BLOCK_N = min(Nd, 512)
//
// Shared memory: 2 * (BLOCK_N/warpSize) floats for ub and lb partial sums
//
// After reduction, lane 0 of warp 0 in each block:
//   - updates d_ub_best[k] atomically via atomicMin bit-cast trick
//   - records translation if new best ub found
//   - writes child to d_nxt[k][...] if lb < optError_snap
//
// Per-node rotation level:
//   d_rot_levels[k] is used to index d_maxRotDis for the lb pass.
//   For the ub pass, d_maxRotDis is not used (is_ub_pass=true).
//
// Atomicity of UB update:
//   atomicMin on int bit-cast is valid for non-negative IEEE-754 floats.
//   The translation recorded after atomicMin may come from a racing thread
//   (not strictly the winner), but any translation with ub <= optError_snap
//   is a valid ICP starting point — correctness is maintained.
// ---------------------------------------------------------------------------

#define EVAL_BLOCK_N  256

__global__ void eval_trans_children(
    const float* __restrict__ d_pDataTemp,   // [K*Nd*3]
    const float* __restrict__ d_maxRotDis,   // [MAXROTLEVEL*Nd] — may be nullptr for ub pass
    const int*   __restrict__ d_rot_levels,  // [K] — rotation level per outer node
    const GpuTransNode* __restrict__ d_cur,  // [K*MAX_TRANS] current level nodes
    const GpuTask* __restrict__ d_tasks,     // [M] compacted task array
    int            M,                        // total number of active tasks
    // output: written atomically
    GpuTransNode*  d_nxt,             // [K*MAX_TRANS] next level survivors
    int*           d_active_count_nxt,// [K] — atomicAdd
    float*         d_ub_best,         // [K] — atomicMin (bit-cast trick)
    float*         d_ub_global_best,  // [1] — running global best ub in the batch
    float*         d_best_tx,         // [K]
    float*         d_best_ty,
    float*         d_best_tz,
    float*         d_best_tw,
    float          optError_snap,
    int            Nd,
    int            inlierNum,
    bool           is_ub_pass,        // true = UB computation (ignore maxRotDis)
    int            max_trans_slots)   // GPU_MAX_TRANS
{
    // Block maps to a specific compacted task
    int task_idx = blockIdx.x;
    if (task_idx >= M) return;

    int k = d_tasks[task_idx].k;
    int s = d_tasks[task_idx].s;

    GpuTransNode parent = d_cur[k * max_trans_slots + s];

    // Compute child translation cube corner and center (mirrors CPU bit-decomposition)
    float child_w = parent.w * 0.5f;
    float half_w = child_w * 0.5f;
    float maxTransDis = GPU_SQRT3 * 0.5f * child_w;

    // Warp w (0..7) evaluates child w
    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    float cx = parent.x + (float)(w & 1) * child_w;
    float cy = parent.y + (float)((w >> 1) & 1) * child_w;
    float cz = parent.z + (float)((w >> 2) & 1) * child_w;
    float tx = cx + half_w;
    float ty = cy + half_w;
    float tz = cz + half_w;

    // Base pointers for this outer node's rotated data in SoA layout [3][K][Nd]
    const float* pTempX = d_pDataTemp + k * Nd * 3;
    const float* pTempY = pTempX + Nd;
    const float* pTempZ = pTempY + Nd;

    // Base pointer for maxRotDis at this rotation level (lb pass only)
    const float* rotDis = nullptr;
    if (!is_ub_pass) {
        int my_level = d_rot_levels[k];
        rotDis = d_maxRotDis + my_level * Nd;
    }

    float local_ub = 0.f;
    float local_lb = 0.f;

    // Each thread in the warp processes a subset of the points
    for (int i = lane; i < inlierNum; i += 32)
    {
        // Coalesced load of rotated point once per loop iteration
        float px = pTempX[i] + tx;
        float py = pTempY[i] + ty;
        float pz = pTempZ[i] + tz;

        float d = dt_lookup(px, py, pz);

        if (!is_ub_pass) {
            d -= rotDis[i];
        }
        if (d < 0.f) d = 0.f;

        local_ub += d * d;

        float dlb = d - maxTransDis;
        if (dlb > 0.f) local_lb += dlb * dlb;
    }

    // Warp-level reduction
    local_ub = warp_reduce_sum(local_ub);
    local_lb = warp_reduce_sum(local_lb);

    // Final reduction and update bounds/queues
    if (lane == 0) {
        // --- UB update (atomicMin via int bit-cast, valid for non-negative floats) ---
        if (local_ub < optError_snap) {
            int* ub_int = (int*)&d_ub_best[k];
            int new_int = __float_as_int(local_ub);
            // atomicMin on int is monotone for positive IEEE-754 floats.
            // Returns the old value; if we installed a new minimum, record
            // the translation.
            int old_int = atomicMin(ub_int, new_int);
            if (old_int > new_int) {
                d_best_tx[k] = cx;
                d_best_ty[k] = cy;
                d_best_tz[k] = cz;
                d_best_tw[k] = child_w;
            }
            // Update global best only during UB pass
            if (is_ub_pass) {
                int* global_ub_int = (int*)d_ub_global_best;
                atomicMin(global_ub_int, new_int);
            }
        }

        // --- Push child to next level if lb < global best UB ---
        float current_global_ub = *d_ub_global_best;
        if (local_lb < current_global_ub) {
            int slot = atomicAdd(&d_active_count_nxt[k], 1);
            if (slot < max_trans_slots) {
                GpuTransNode child;
                child.x  = cx;
                child.y  = cy;
                child.z  = cz;
                child.w  = child_w;
                child.lb_parent = local_lb;
                d_nxt[k * max_trans_slots + slot] = child;
            }
            // slot >= max_trans_slots: overflow silently truncated.
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: init_batch_state
// Resets per-outer-node state before each wavefront pass.
// Initialises d_ub_best to optError_snap (so the first valid child wins),
// resets active counts to 1 (one root trans node), and sets initial best_t*.
// ---------------------------------------------------------------------------
__global__ void init_batch_state(
    float* d_ub_best,
    float* d_best_tx, float* d_best_ty, float* d_best_tz, float* d_best_tw,
    int*   d_active_count,
    float* d_ub_global_best,
    float  optError_snap,
    float  init_tx, float init_ty, float init_tz, float init_tw,
    int    K,
    bool   is_ub_pass)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k == 0 && is_ub_pass) {
        *d_ub_global_best = optError_snap;
    }
    if (k >= K) return;
    d_ub_best[k]      = optError_snap;
    d_best_tx[k]      = init_tx;
    d_best_ty[k]      = init_ty;
    d_best_tz[k]      = init_tz;
    d_best_tw[k]      = init_tw;
    d_active_count[k] = 1;  // one root trans node per outer node
}

// ---------------------------------------------------------------------------
// Kernel 4: init_trans_pool
// Writes the single root translation node into slot 0 of each outer node.
// ---------------------------------------------------------------------------
__global__ void init_trans_pool(
    GpuTransNode* d_pool,
    float init_tx, float init_ty, float init_tz, float init_tw,
    int max_trans_slots, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    GpuTransNode root;
    root.x  = init_tx;
    root.y  = init_ty;
    root.z  = init_tz;
    root.w  = init_tw;
    root.lb_parent = 0.f;
    d_pool[k * max_trans_slots + 0] = root;
}

// ---------------------------------------------------------------------------
// Kernel 5: zero_next_counts
// Clears d_active_count_nxt before each wavefront expansion.
// ---------------------------------------------------------------------------
__global__ void zero_next_counts(int* d_active_count_nxt, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    d_active_count_nxt[k] = 0;
}

// ---------------------------------------------------------------------------
// collect_results_kernel
// Writes final (ub, best_t*) into GpuRotResult array after UB pass.
// The lb field is filled by the host from the d_ub_best snapshot after
// the LB wavefront pass (see GpuRunBatch).
// ---------------------------------------------------------------------------
__global__ void collect_results_kernel(
    const float* d_ub_best,
    const float* d_best_tx, const float* d_best_ty,
    const float* d_best_tz, const float* d_best_tw,
    GpuRotResult* d_results, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    d_results[k].ub       = d_ub_best[k];
    d_results[k].best_tx  = d_best_tx[k];
    d_results[k].best_ty  = d_best_ty[k];
    d_results[k].best_tz  = d_best_tz[k];
    d_results[k].best_tw  = d_best_tw[k];
    d_results[k].lb       = 0.f;  // placeholder; filled after LB pass in host
}

// =============================================================================
// Host-side API implementation
// =============================================================================

extern "C" void GpuInit(
    GoICPGpu*     gpu,
    int           Nd,
    const float*  pData_xyz,
    float**       maxRotDis_cpu,
    const float*  dt_dist,
    int           dt_size,
    float         dt_xMin, float dt_yMin, float dt_zMin,
    float         dt_scale)
{
    memset(gpu, 0, sizeof(*gpu));
    gpu->Nd = Nd;
    gpu->K  = GPU_BATCH_K;

    // --- Upload point cloud ---
    CUDA_CHECK(cudaMalloc(&gpu->d_pData, sizeof(float) * Nd * 3));
    CUDA_CHECK(cudaMemcpy(gpu->d_pData, pData_xyz,
                          sizeof(float) * Nd * 3, cudaMemcpyHostToDevice));

    // --- Upload maxRotDis: flatten [MAXROTLEVEL][Nd] → [MAXROTLEVEL*Nd] ---
    {
        float* flat = (float*)malloc(sizeof(float) * GPU_MAXROTLEVEL * Nd);
        for (int l = 0; l < GPU_MAXROTLEVEL; l++)
            memcpy(flat + l * Nd, maxRotDis_cpu[l], sizeof(float) * Nd);
        CUDA_CHECK(cudaMalloc(&gpu->d_maxRotDis,
                              sizeof(float) * GPU_MAXROTLEVEL * Nd));
        CUDA_CHECK(cudaMemcpy(gpu->d_maxRotDis, flat,
                              sizeof(float) * GPU_MAXROTLEVEL * Nd,
                              cudaMemcpyHostToDevice));
        free(flat);
    }

    // --- Upload DT distance array to CUDA Array and bind Texture Object ---
    cudaExtent extent = make_cudaExtent(dt_size, dt_size, dt_size);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray(&gpu->d_distArray, &channelDesc, extent));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)dt_dist, dt_size * sizeof(float), dt_size, dt_size);
    copyParams.dstArray = gpu->d_distArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gpu->d_distArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; // Match CPU exactly
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_CHECK(cudaCreateTextureObject(&gpu->tex_dist, &resDesc, &texDesc, nullptr));
    
    GpuDT dt_host;
    dt_host.tex_dist = gpu->tex_dist;
    dt_host.size  = dt_size;
    dt_host.xMin  = dt_xMin;
    dt_host.yMin  = dt_yMin;
    dt_host.zMin  = dt_zMin;
    dt_host.scale = dt_scale;
    gpu->dt = dt_host;

    // Copy DT descriptor to constant memory (pointer + scalars together)
    CUDA_CHECK(cudaMemcpyToSymbol(c_dt, &dt_host, sizeof(GpuDT)));

    // --- Allocate pDataTemp [3][K][Nd] ---
    CUDA_CHECK(cudaMalloc(&gpu->d_pDataTemp,
                          sizeof(float) * 3 * GPU_BATCH_K * Nd));

    // --- Pre-allocate rotation matrix buffer [K*9] ---
    CUDA_CHECK(cudaMalloc(&gpu->d_R, sizeof(float) * GPU_BATCH_K * 9));

    // --- Per-batch transfer buffers ---
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_batch,
                          sizeof(GpuRotBatch) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_result,
                          sizeof(GpuRotResult) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_levels,
                          sizeof(int) * GPU_BATCH_K));

    // --- Wavefront working memory ---
    size_t pool_sz = sizeof(GpuTransNode) * GPU_BATCH_K * GPU_MAX_TRANS;
    CUDA_CHECK(cudaMalloc(&gpu->d_trans_pool_cur, pool_sz));
    CUDA_CHECK(cudaMalloc(&gpu->d_trans_pool_nxt, pool_sz));
    CUDA_CHECK(cudaMalloc(&gpu->d_active_count,
                          sizeof(int) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_active_count_nxt,
                          sizeof(int) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_ub_best,  sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_ub_global_best, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_lb_best,  sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_tx,  sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_ty,  sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_tz,  sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_tw,  sizeof(float) * GPU_BATCH_K));

    // --- Persistent termination flag (avoids malloc/free in wavefront loop) ---
    CUDA_CHECK(cudaMalloc(&gpu->d_flag, sizeof(int)));

    // --- Pinned host buffers for async H2D/D2H ---
    CUDA_CHECK(cudaMallocHost(&gpu->h_rot_batch,
                              sizeof(GpuRotBatch) * GPU_BATCH_K));
    CUDA_CHECK(cudaMallocHost(&gpu->h_rot_result,
                              sizeof(GpuRotResult) * GPU_BATCH_K));

    // --- Task compaction buffers ---
    CUDA_CHECK(cudaMalloc(&gpu->d_tasks, sizeof(GpuTask) * GPU_BATCH_K * GPU_MAX_TRANS));
    CUDA_CHECK(cudaMallocHost(&gpu->h_tasks, sizeof(GpuTask) * GPU_BATCH_K * GPU_MAX_TRANS));

    printf("[GPU] Init complete. Nd=%d, DT=%dx%dx%d, "
           "pDataTemp=%.1f MB, TransPool=%.1f MB\n",
           Nd, dt_size, dt_size, dt_size,
           3 * GPU_BATCH_K * Nd * sizeof(float) / 1e6f,
           2 * pool_sz / 1e6f);
}

// ---------------------------------------------------------------------------
// run_inner_bnb_wavefront
//
// Executes one complete inner BnB wavefront loop for all K outer nodes.
// The termination check is done entirely on the host by reading
// d_active_count_nxt after each synchronized eval_trans_children launch.
// This eliminates the check_any_active kernel and the extra D2H copy,
// reducing synchronization cost to one cudaDeviceSynchronize + one
// cudaMemcpy of K ints per wavefront level (K is at most 256 ints = 1 KB).
//
// The same D2H copy of counts also drives the grid dimension for the next
// level, so it is unavoidable — we simply remove the redundant flag copy.
// ---------------------------------------------------------------------------
static void run_inner_bnb_wavefront(
    GoICPGpu*    gpu,
    int          actual_k,
    float        optError_snap,
    int          inlierNum,
    float        init_tx, float init_ty, float init_tz, float init_tw,
    bool         is_ub_pass)
{
    // ---- Initialise batch state ----
    {
        int blocks = (actual_k + 255) / 256;
        init_batch_state<<<blocks, 256>>>(
            gpu->d_ub_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_active_count,
            gpu->d_ub_global_best,
            optError_snap,
            init_tx, init_ty, init_tz, init_tw,
            actual_k,
            is_ub_pass);

        init_trans_pool<<<blocks, 256>>>(
            gpu->d_trans_pool_cur,
            init_tx, init_ty, init_tz, init_tw,
            GPU_MAX_TRANS, actual_k);
        // No sync needed here: zero_next_counts below touches different memory
    }

    // ---- Wavefront loop ----
    // h_counts[k] = number of active trans nodes for outer node k at this level
    int h_counts[GPU_BATCH_K];

    // Prime h_counts with the initial state (1 root per outer node)
    for (int k = 0; k < actual_k; k++) h_counts[k] = 1;

    int max_levels = 60;  // safety cap
    for (int level = 0; level < max_levels; level++)
    {
        // Build compacted task list
        int M = 0;
        for (int k = 0; k < actual_k; k++) {
            int count = h_counts[k];
            if (count > GPU_MAX_TRANS) count = GPU_MAX_TRANS;
            for (int s = 0; s < count; s++) {
                gpu->h_tasks[M].k = k;
                gpu->h_tasks[M].s = s;
                M++;
            }
        }

        if (M == 0) break;

        // Copy compacted tasks to device
        CUDA_CHECK(cudaMemcpyAsync(gpu->d_tasks, gpu->h_tasks, sizeof(GpuTask) * M, cudaMemcpyHostToDevice, 0));

        // Zero next-level counts before expansion
        {
            int blocks = (actual_k + 255) / 256;
            zero_next_counts<<<blocks, 256>>>(gpu->d_active_count_nxt, actual_k);
        }

        // Each block handles one specific task (outer_node k, parent_slot s)
        // evaluating all 8 children in parallel (one warp per child)
        // Grid: (M)
        // Block: (EVAL_BLOCK_N = 256)
        dim3 grid(M);
        dim3 block(EVAL_BLOCK_N);
        int smem_bytes = 0;

        eval_trans_children<<<grid, block, smem_bytes>>>(
            gpu->d_pDataTemp,
            is_ub_pass ? nullptr : gpu->d_maxRotDis,
            gpu->d_rot_levels,         // per-node rotation level for lb pass
            gpu->d_trans_pool_cur,
            gpu->d_tasks,
            M,
            gpu->d_trans_pool_nxt,
            gpu->d_active_count_nxt,
            gpu->d_ub_best,
            gpu->d_ub_global_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            optError_snap,
            gpu->Nd,
            inlierNum,
            is_ub_pass,
            GPU_MAX_TRANS);

        // Sync is mandatory: we must read d_active_count_nxt to know the
        // next level's grid size, and the swap must complete before iter+1.
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read next-level counts — drives convergence check AND next grid dim.
        // This single D2H of K ints replaces two separate D2H copies
        // (the old flag copy + the count copy).
        CUDA_CHECK(cudaMemcpy(h_counts, gpu->d_active_count_nxt,
                              sizeof(int) * actual_k,
                              cudaMemcpyDeviceToHost));

        // Swap cur/nxt pools (pointer swap — no data movement)
        {
            GpuTransNode* tmp_pool   = gpu->d_trans_pool_cur;
            int*          tmp_counts = gpu->d_active_count;
            gpu->d_trans_pool_cur   = gpu->d_trans_pool_nxt;
            gpu->d_active_count     = gpu->d_active_count_nxt;
            gpu->d_trans_pool_nxt   = tmp_pool;
            gpu->d_active_count_nxt = tmp_counts;
        }
    }
}

extern "C" void GpuRunBatch(
    GoICPGpu*          gpu,
    const GpuRotBatch* rot_batch_cpu,
    int                actual_k,
    float              optError_snap,
    int                inlierNum,
    float              init_tx, float init_ty, float init_tz, float init_tw,
    GpuRotResult*      results_cpu)
{
    // ---- H2D: rotation batch metadata (for host result processing) ----
    memcpy(gpu->h_rot_batch, rot_batch_cpu,
           sizeof(GpuRotBatch) * actual_k);

    // ---- Build per-node rotation level array and upload ----
    // Also flatten rotation matrices into d_R (pre-allocated, no malloc/free).
    {
        float h_R[GPU_BATCH_K * 9];
        int   h_levels[GPU_BATCH_K];
        for (int k = 0; k < actual_k; k++) {
            memcpy(h_R + k * 9, rot_batch_cpu[k].R, sizeof(float) * 9);
            h_levels[k] = rot_batch_cpu[k].l;
        }
        CUDA_CHECK(cudaMemcpy(gpu->d_R, h_R,
                              sizeof(float) * actual_k * 9,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu->d_rot_levels, h_levels,
                              sizeof(int) * actual_k,
                              cudaMemcpyHostToDevice));
    }

    // ---- Kernel 1: Rotate all N points for all K nodes ----
    {
        dim3 grid(actual_k, (gpu->Nd + 255) / 256);
        dim3 block(256);
        rotate_points_kernel<<<grid, block>>>(
            gpu->d_pData, gpu->d_R, gpu->d_pDataTemp, gpu->Nd);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Pass 1: UB inner BnB (mirrors InnerBnB(NULL, &nodeTrans)) ----
    run_inner_bnb_wavefront(gpu, actual_k, optError_snap, inlierNum,
                            init_tx, init_ty, init_tz, init_tw,
                            /*is_ub_pass=*/true);

    // Collect UB results into d_rot_result (device-side copy)
    {
        int blocks = (actual_k + 255) / 256;
        collect_results_kernel<<<blocks, 256>>>(
            gpu->d_ub_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_rot_result, actual_k);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Pass 2: LB inner BnB (mirrors InnerBnB(maxRotDis[l], NULL)) ----
    // Re-initialises d_ub_best / d_active_count / pools before the LB pass.
    // The UB results are safely in d_rot_result (written above).
    run_inner_bnb_wavefront(gpu, actual_k, optError_snap, inlierNum,
                            init_tx, init_ty, init_tz, init_tw,
                            /*is_ub_pass=*/false);

    // After LB pass: d_ub_best[k] holds the minimum over all translation
    // children with maxRotDis subtracted — this IS the rotation lower bound
    // (exactly mirrors what InnerBnB(maxRotDis[l], NULL) returns on CPU).
    {
        // Read UB results from device (d_rot_result was written after UB pass)
        CUDA_CHECK(cudaMemcpy(gpu->h_rot_result, gpu->d_rot_result,
                              sizeof(GpuRotResult) * actual_k,
                              cudaMemcpyDeviceToHost));

        // Read LB values — d_ub_best now holds lb from Pass 2
        float h_lb[GPU_BATCH_K];
        CUDA_CHECK(cudaMemcpy(h_lb, gpu->d_ub_best,
                              sizeof(float) * actual_k,
                              cudaMemcpyDeviceToHost));

        for (int k = 0; k < actual_k; k++) {
            results_cpu[k]    = gpu->h_rot_result[k];
            results_cpu[k].lb = h_lb[k];
        }
    }
}

extern "C" void GpuFree(GoICPGpu* gpu)
{
    if (gpu->d_pData)           cudaFree(gpu->d_pData);
    if (gpu->d_maxRotDis)       cudaFree(gpu->d_maxRotDis);
    if (gpu->d_pDataTemp)       cudaFree(gpu->d_pDataTemp);
    if (gpu->d_R)               cudaFree(gpu->d_R);
    if (gpu->tex_dist)          cudaDestroyTextureObject(gpu->tex_dist);
    if (gpu->d_distArray)       cudaFreeArray(gpu->d_distArray);
    if (gpu->d_rot_batch)       cudaFree(gpu->d_rot_batch);
    if (gpu->d_rot_result)      cudaFree(gpu->d_rot_result);
    if (gpu->d_rot_levels)      cudaFree(gpu->d_rot_levels);
    if (gpu->d_trans_pool_cur)  cudaFree(gpu->d_trans_pool_cur);
    if (gpu->d_trans_pool_nxt)  cudaFree(gpu->d_trans_pool_nxt);
    if (gpu->d_active_count)    cudaFree(gpu->d_active_count);
    if (gpu->d_active_count_nxt)cudaFree(gpu->d_active_count_nxt);
    if (gpu->d_ub_best) cudaFree(gpu->d_ub_best);
    if (gpu->d_ub_global_best) cudaFree(gpu->d_ub_global_best);
    if (gpu->d_lb_best) cudaFree(gpu->d_lb_best);
    if (gpu->d_best_tx) cudaFree(gpu->d_best_tx);
    if (gpu->d_best_ty) cudaFree(gpu->d_best_ty);
    if (gpu->d_best_tz) cudaFree(gpu->d_best_tz);
    if (gpu->d_best_tw) cudaFree(gpu->d_best_tw);
    if (gpu->d_flag) cudaFree(gpu->d_flag);
    if (gpu->h_rot_batch) cudaFreeHost(gpu->h_rot_batch);
    if (gpu->h_rot_result) cudaFreeHost(gpu->h_rot_result);
    if (gpu->d_tasks) cudaFree(gpu->d_tasks);
    if (gpu->h_tasks) cudaFreeHost(gpu->h_tasks);
    memset(gpu, 0, sizeof(*gpu));
}

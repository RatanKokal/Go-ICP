// =============================================================================
// goicp_gpu.cu  —  Method A CUDA implementation (rev 3)
//
// Rev 3 changes vs rev 2.1 — the inner translation BnB is now CORRECT under
// truncation, so a full pool is no longer a fatal condition.  Three coupled
// changes:
//
//  (A) VALID ROTATION LB UNDER TRUNCATION
//      local_lb_rot now subtracts BOTH rotDis AND the child's maxTransDis:
//          d_lb = max(d_raw_centre - rotDis - maxTransDis, 0)
//      Each evaluated child therefore lower-bounds the registration error over
//      its OWN translation sub-cube (for every rotation in the rotation cube).
//      min over any covering set of evaluated children is then a valid LB for
//      the whole (rotation cube x translation cube), independent of how deep
//      the wavefront went.  Previously local_lb_rot omitted maxTransDis, so the
//      reported rotation LB was only valid once the BFS had run to convergence
//      — truncating it produced a too-high (invalid) LB and silently broke
//      global optimality.
//
//  (B) NON-FATAL TRUNCATION
//      When the per-node pool is full (slot >= GPU_MAX_TRANS) the child is no
//      longer pushed, but its bounds were already recorded in (A) BEFORE the
//      push decision, so the LB stays valid — just looser.  No abort.
//      d_overflow_flag is repurposed as a truncation COUNTER (telemetry only);
//      the host prints a one-line non-fatal notice if it fires.
//
//  (C) WIDTH LEAF TEST
//      A translation child is a leaf (recorded but not subdivided) once its
//      uncertainty is below the per-point MSE tolerance:
//          maxTransDis^2 <= mse_thresh
//      This stops pointless deep subdivision near convergence (rev 2 ran to a
//      fixed 60 levels) and is dimensionally consistent: maxTransDis^2 and
//      MSEThresh are both per-point squared distances.
//
//  GPU_MAX_TRANS is now a SOFT cap on frontier width, not a crash condition.
//  Raising it (linear memory cost) tightens the LB and reduces outer-BnB
//  iterations; it no longer risks a fatal abort.
//
//  NOTE: this keeps the breadth-first wavefront.  It is correct and bounded,
//  but at coarse translation widths nothing prunes (maxTransDis >> distances),
//  so the frontier still saturates the cap and the LB is looser than a
//  best-first inner search would give.  A best-first-per-node inner BnB is the
//  real performance fix; this rev makes the wavefront correct and crash-free.
//
// Synchronization per wavefront level (unchanged):
//   1x cudaDeviceSynchronize  (next level grid size from h_counts)
//   1x cudaMemcpy D2H of K ints
// =============================================================================

#include "goicp_gpu.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(1);                                                     \
    }                                                                \
} while(0)

// ---------------------------------------------------------------------------
// Constant memory for DT descriptor (file scope — required by CUDA)
// ---------------------------------------------------------------------------
__constant__ GpuDT c_dt;

// ---------------------------------------------------------------------------
// DT lookup — nearest-cell, with OOB extrapolation matching CPU DT3D::Distance
// ---------------------------------------------------------------------------
__device__ __forceinline__
float dt_lookup(float qx, float qy, float qz)
{
    float fx = (qx - c_dt.xMin) * c_dt.scale;
    float fy = (qy - c_dt.yMin) * c_dt.scale;
    float fz = (qz - c_dt.zMin) * c_dt.scale;

    int ix = __float2int_rn(fx);
    int iy = __float2int_rn(fy);
    int iz = __float2int_rn(fz);

    float ex = 0.f, ey = 0.f, ez = 0.f;

    if      (ix < 0)            { ex = (float)ix;               ix = 0; }
    else if (ix >= c_dt.size)   { ex = (float)(ix-c_dt.size+1); ix = c_dt.size-1; }

    if      (iy < 0)            { ey = (float)iy;               iy = 0; }
    else if (iy >= c_dt.size)   { ey = (float)(iy-c_dt.size+1); iy = c_dt.size-1; }

    if      (iz < 0)            { ez = (float)iz;               iz = 0; }
    else if (iz >= c_dt.size)   { ez = (float)(iz-c_dt.size+1); iz = c_dt.size-1; }

    float dist = tex3D<float>(c_dt.tex_dist, ix+0.5f, iy+0.5f, iz+0.5f);

    float oob = ex*ex + ey*ey + ez*ez;
    if (oob > 0.f) dist += sqrtf(oob) / c_dt.scale;

    return dist;
}

// ---------------------------------------------------------------------------
// Warp reduction — horizontal sum of 32 floats
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
// Kernel 1: rotate_points_kernel  (unchanged)
// Grid: (K, ceil(Nd/256))   Block: (256)
// ---------------------------------------------------------------------------
__global__ void rotate_points_kernel(
    const float* __restrict__ d_pData,
    const float* __restrict__ d_R,
    float*                    d_pDataTemp,
    int Nd)
{
    int k = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= Nd) return;

    const float* R = d_R + k * 9;
    float px = d_pData[i*3+0];
    float py = d_pData[i*3+1];
    float pz = d_pData[i*3+2];

    float* outX = d_pDataTemp + k * Nd * 3 + i;
    float* outY = outX + Nd;
    float* outZ = outY + Nd;
    *outX = R[0]*px + R[1]*py + R[2]*pz;
    *outY = R[3]*px + R[4]*py + R[5]*pz;
    *outZ = R[6]*px + R[7]*py + R[8]*pz;
}

// ---------------------------------------------------------------------------
// Kernel 2: eval_trans_children  (rev 3 — valid LB, leaf test, truncation)
//
// Grid:  dim3(actual_k, max_active_count_this_level)
// Block: dim3(EVAL_BLOCK_N = 256)   — 8 warps, one warp per child
//
// blockIdx.x = k  (outer rotation node index)
// blockIdx.y = s  (parent translation slot index)
//
// Each warp (w = threadIdx.x >> 5) evaluates one of the 8 children of the
// parent at slot s.  The point loop computes, at the child cube CENTRE:
//
//   d_raw = dt_lookup(rotated_point + translation_centre)
//   d_ub  = max(d_raw, 0)                              → UB numerator
//   d_lb  = max(d_raw - rotDis[i] - maxTransDis, 0)    → VALID rot+trans LB
//
//   local_ub       += d_ub^2   → rotation UB (min over children = achievable)
//   local_lb_rot   += d_lb^2   → rotation LB (min over a valid cover)
//   local_lb_trans += max(d_ub - maxTransDis, 0)^2
//                              → expansion heuristic only (not a reported bound)
//
// local_lb_rot subtracts maxTransDis so EVERY evaluated child lower-bounds the
// error over its own translation sub-cube.  Because the push decision happens
// AFTER this is recorded, dropping a child (full pool or leaf) keeps the
// reported LB valid — just looser.
//
// Trans-pool expansion uses d_ub (UB-pass criterion).  For coarse rotations
// (large rotDis) a d_lb-based criterion would clamp to ~0 and prune nothing;
// d_ub gives effective expansion ordering even there.  Note: this is now only
// a heuristic for WHICH nodes to spend pool slots on — correctness of the
// reported LB no longer depends on it.
//
// Atomic bit-cast trick: atomicMin on int is valid for non-negative floats
// because positive IEEE-754 floats order identically to their bit patterns as
// unsigned ints, and the sign bit is always 0.
// ---------------------------------------------------------------------------

#define EVAL_BLOCK_N 256

__global__ void eval_trans_children(
    const float* __restrict__ d_pDataTemp,      // [K*Nd*3]  SoA per k
    const float* __restrict__ d_maxRotDis,      // [MAXROTLEVEL*Nd]
    const int*   __restrict__ d_rot_levels,     // [K]
    const GpuTransNode* __restrict__ d_cur,     // [K*GPU_MAX_TRANS] current pool
    const int*   __restrict__ d_active_count_cur, // [K] active counts this level
    GpuTransNode* d_nxt,                        // [K*GPU_MAX_TRANS] next pool
    int*          d_active_count_nxt,           // [K]
    float*        d_ub_best,                    // [K]
    float*        d_rot_lb_best,                // [K]
    float*        d_ub_global_best,             // [1]
    float*        d_best_tx,                    // [K]
    float*        d_best_ty,
    float*        d_best_tz,
    float*        d_best_tw,
    int*          d_trunc_count,                // [1]  truncation telemetry
    float         optError_snap,
    float         mse_thresh,                   // per-point MSE tolerance (leaf)
    int           Nd,
    int           inlierNum,
    int           max_trans_slots)
{
    int k = blockIdx.x;
    int s = blockIdx.y;

    // Early exit for inactive slots.  The stored count may exceed
    // max_trans_slots after a truncated level; clamp so reads stay in bounds.
    int cnt = d_active_count_cur[k];
    if (cnt > max_trans_slots) cnt = max_trans_slots;
    if (s >= cnt) return;

    GpuTransNode parent = d_cur[k * max_trans_slots + s];

    float child_w     = parent.w * 0.5f;
    float half_w      = child_w  * 0.5f;
    float maxTransDis  = GPU_SQRT3 * 0.5f * child_w;
    float maxTransDis2 = maxTransDis * maxTransDis;

    // Leaf test: once the child's translation uncertainty is within the
    // per-point MSE tolerance, further subdivision cannot tighten the bound
    // by more than the tolerance — record it but do not expand.
    bool is_leaf = (maxTransDis2 <= mse_thresh);

    int w    = threadIdx.x >> 5;   // warp index = child index (0..7)
    int lane = threadIdx.x & 31;

    float cx = parent.x + (float)(w & 1)       * child_w;
    float cy = parent.y + (float)((w >> 1) & 1) * child_w;
    float cz = parent.z + (float)((w >> 2) & 1) * child_w;
    float tx = cx + half_w;
    float ty = cy + half_w;
    float tz = cz + half_w;

    // Rotated points for outer node k (SoA layout)
    const float* pTempX = d_pDataTemp + k * Nd * 3;
    const float* pTempY = pTempX + Nd;
    const float* pTempZ = pTempY + Nd;

    // Rotation uncertainty for this outer node (for LB computation)
    int my_level = d_rot_levels[k];
    const float* rotDis = d_maxRotDis + my_level * Nd;

    float local_ub       = 0.f;   // rotation UB accumulator
    float local_lb_rot   = 0.f;   // rotation LB accumulator (VALID: rot+trans)
    float local_lb_trans = 0.f;   // expansion heuristic only

    for (int i = lane; i < inlierNum; i += 32)
    {
        float px = pTempX[i] + tx;
        float py = pTempY[i] + ty;
        float pz = pTempZ[i] + tz;

        float d_raw = dt_lookup(px, py, pz);

        // UB path: clamp at 0
        float d_ub = d_raw < 0.f ? 0.f : d_raw;
        local_ub += d_ub * d_ub;

        // VALID LB path: subtract rotation AND translation uncertainty.
        // This is the change that makes truncation safe — each child now
        // lower-bounds error over its own translation sub-cube.
        float d_lb = d_raw - rotDis[i] - maxTransDis;
        if (d_lb < 0.f) d_lb = 0.f;
        local_lb_rot += d_lb * d_lb;

        // Expansion heuristic (UB-pass criterion). Not a reported bound.
        float dlt = d_ub - maxTransDis;
        if (dlt > 0.f) local_lb_trans += dlt * dlt;
    }

    // Warp-level horizontal reduction
    local_ub       = warp_reduce_sum(local_ub);
    local_lb_rot   = warp_reduce_sum(local_lb_rot);
    local_lb_trans = warp_reduce_sum(local_lb_trans);

    // Only lane 0 of each warp writes results
    if (lane == 0)
    {
        // ---- Rotation UB: update d_ub_best[k] and global best ----
        int new_ub_int = __float_as_int(local_ub);
        int* ub_int = (int*)&d_ub_best[k];
        int old_ub_int = atomicMin(ub_int, new_ub_int);
        if (old_ub_int > new_ub_int) {
            d_best_tx[k] = cx;
            d_best_ty[k] = cy;
            d_best_tz[k] = cz;
            d_best_tw[k] = child_w;
        }
        atomicMin((int*)d_ub_global_best, new_ub_int);

        // ---- Rotation LB: update d_rot_lb_best[k] (valid for this sub-cube) ----
        // Recorded BEFORE the push decision below, so a dropped child (leaf or
        // full pool) still contributes its valid bound.
        atomicMin((int*)&d_rot_lb_best[k], __float_as_int(local_lb_rot));

        // ---- Expand child to next level, unless it is a leaf or pruned ----
        float cur_global_ub = *d_ub_global_best;
        if (!is_leaf && local_lb_trans < cur_global_ub) {
            int slot = atomicAdd(&d_active_count_nxt[k], 1);
            if (slot < max_trans_slots) {
                GpuTransNode child;
                child.x  = cx;
                child.y  = cy;
                child.z  = cz;
                child.w  = child_w;
                child.lb_parent = local_lb_trans;
                d_nxt[k * max_trans_slots + slot] = child;
            } else {
                // Pool full: do not write. The child's valid LB was already
                // recorded above, so global optimality is preserved (looser
                // LB only). Count for telemetry; non-fatal.
                atomicAdd(d_trunc_count, 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: init_batch_state  (unchanged)
// ---------------------------------------------------------------------------
__global__ void init_batch_state(
    float* d_ub_best,
    float* d_rot_lb_best,
    float* d_best_tx, float* d_best_ty, float* d_best_tz, float* d_best_tw,
    int*   d_active_count,
    float* d_ub_global_best,
    float  optError_snap,
    float  init_tx, float init_ty, float init_tz, float init_tw,
    int    K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k == 0)
        *d_ub_global_best = optError_snap;
    if (k >= K) return;
    d_ub_best[k]      = optError_snap;
    d_rot_lb_best[k]  = optError_snap; // lb can't exceed current best
    d_best_tx[k]      = init_tx;
    d_best_ty[k]      = init_ty;
    d_best_tz[k]      = init_tz;
    d_best_tw[k]      = init_tw;
    d_active_count[k] = 1;
}

// ---------------------------------------------------------------------------
// Kernel 4: init_trans_pool  (unchanged)
// ---------------------------------------------------------------------------
__global__ void init_trans_pool(
    GpuTransNode* d_pool,
    float init_tx, float init_ty, float init_tz, float init_tw,
    int max_trans_slots, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    GpuTransNode root;
    root.x  = init_tx; root.y = init_ty; root.z = init_tz;
    root.w  = init_tw; root.lb_parent = 0.f;
    d_pool[k * max_trans_slots + 0] = root;
}

// ---------------------------------------------------------------------------
// Kernel 5: zero_next_counts  (unchanged)
// ---------------------------------------------------------------------------
__global__ void zero_next_counts(int* d_active_count_nxt, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    d_active_count_nxt[k] = 0;
}

// ---------------------------------------------------------------------------
// Kernel 6: collect_results_kernel  (unchanged)
// ---------------------------------------------------------------------------
__global__ void collect_results_kernel(
    const float* d_ub_best,
    const float* d_rot_lb_best,
    const float* d_best_tx, const float* d_best_ty,
    const float* d_best_tz, const float* d_best_tw,
    GpuRotResult* d_results, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    d_results[k].ub      = d_ub_best[k];
    d_results[k].lb      = d_rot_lb_best[k];
    d_results[k].best_tx = d_best_tx[k];
    d_results[k].best_ty = d_best_ty[k];
    d_results[k].best_tz = d_best_tz[k];
    d_results[k].best_tw = d_best_tw[k];
}

// =============================================================================
// Host-side API
// =============================================================================

// ---------------------------------------------------------------------------
// run_inner_bnb_wavefront  (rev 3)
//
// Single merged pass: computes rotation UB (d_ub_best) and rotation LB
// (d_rot_lb_best) in one wavefront over the translation BnB tree.
//
// The wavefront now terminates a branch via the width leaf test (in-kernel)
// rather than a fixed level cap; max_levels remains as a safety bound only.
// Active counts are clamped to GPU_MAX_TRANS on the host so the next grid Y
// dimension and all device reads stay in bounds after a truncated level.
// ---------------------------------------------------------------------------
static void run_inner_bnb_wavefront(
    GoICPGpu*    gpu,
    int          actual_k,
    float        optError_snap,
    float        mse_thresh,
    int          inlierNum,
    float        init_tx, float init_ty, float init_tz, float init_tw)
{
    // Reset per-outer-node state and trans pool
    {
        int blocks = (actual_k + 255) / 256;
        init_batch_state<<<blocks, 256>>>(
            gpu->d_ub_best,
            gpu->d_rot_lb_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_active_count,
            gpu->d_ub_global_best,
            optError_snap,
            init_tx, init_ty, init_tz, init_tw,
            actual_k);

        init_trans_pool<<<blocks, 256>>>(
            gpu->d_trans_pool_cur,
            init_tx, init_ty, init_tz, init_tw,
            GPU_MAX_TRANS, actual_k);
    }

    // h_counts[k] = number of active trans nodes for outer node k at this level
    int h_counts[GPU_BATCH_K];
    for (int k = 0; k < actual_k; k++) h_counts[k] = 1;  // root level

    const int max_levels = 60;  // safety bound; leaf test normally terminates first
    for (int level = 0; level < max_levels; level++)
    {
        // Compute max active count to size the Y dimension of the 2D grid.
        // h_counts are already clamped to GPU_MAX_TRANS (see end of loop).
        int max_count = 0;
        for (int k = 0; k < actual_k; k++)
            if (h_counts[k] > max_count) max_count = h_counts[k];

        if (max_count == 0) break;

        // Zero next-level counts before expansion
        {
            int blk = (actual_k + 255) / 256;
            zero_next_counts<<<blk, 256>>>(gpu->d_active_count_nxt, actual_k);
        }

        // 2D grid: (actual_k, max_count <= GPU_MAX_TRANS)
        dim3 grid(actual_k, max_count);
        dim3 block(EVAL_BLOCK_N);

        eval_trans_children<<<grid, block>>>(
            gpu->d_pDataTemp,
            gpu->d_maxRotDis,
            gpu->d_rot_levels,
            gpu->d_trans_pool_cur,
            gpu->d_active_count,          // current level counts (for early exit)
            gpu->d_trans_pool_nxt,
            gpu->d_active_count_nxt,
            gpu->d_ub_best,
            gpu->d_rot_lb_best,
            gpu->d_ub_global_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_overflow_flag,         // now a truncation counter
            optError_snap,
            mse_thresh,
            gpu->Nd,
            inlierNum,
            GPU_MAX_TRANS);

        // Mandatory sync: next grid Y dim depends on new active counts
        CUDA_CHECK(cudaDeviceSynchronize());

        // D2H next-level counts (drives convergence check and next grid dim)
        CUDA_CHECK(cudaMemcpy(h_counts, gpu->d_active_count_nxt,
                              sizeof(int) * actual_k,
                              cudaMemcpyDeviceToHost));

        // Clamp to the pool size: a truncated level may report a count larger
        // than GPU_MAX_TRANS, but only GPU_MAX_TRANS slots hold valid data.
        // Clamping keeps the next grid Y dim and all device reads in bounds.
        for (int k = 0; k < actual_k; k++)
            if (h_counts[k] > GPU_MAX_TRANS) h_counts[k] = GPU_MAX_TRANS;

        // Swap cur/nxt pools (pointer swap, no data movement)
        {
            GpuTransNode* tmp_pool    = gpu->d_trans_pool_cur;
            int*          tmp_counts  = gpu->d_active_count;
            gpu->d_trans_pool_cur     = gpu->d_trans_pool_nxt;
            gpu->d_active_count       = gpu->d_active_count_nxt;
            gpu->d_trans_pool_nxt     = tmp_pool;
            gpu->d_active_count_nxt   = tmp_counts;
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
    float              mse_thresh,
    GpuRotResult*      results_cpu)
{
    // H2D: rotation matrices and levels (small, synchronous transfers are fine)
    {
        float h_R[GPU_BATCH_K * 9];
        int   h_levels[GPU_BATCH_K];
        for (int k = 0; k < actual_k; k++) {
            memcpy(h_R + k*9, rot_batch_cpu[k].R, sizeof(float)*9);
            h_levels[k] = rot_batch_cpu[k].l;
        }
        CUDA_CHECK(cudaMemcpy(gpu->d_R, h_R,
                              sizeof(float) * actual_k * 9,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu->d_rot_levels, h_levels,
                              sizeof(int) * actual_k,
                              cudaMemcpyHostToDevice));
    }

    // Reset truncation counter before wavefront
    CUDA_CHECK(cudaMemset(gpu->d_overflow_flag, 0, sizeof(int)));

    // Rotate all Nd points for all K outer nodes
    {
        dim3 grid(actual_k, (gpu->Nd + 255) / 256);
        rotate_points_kernel<<<grid, dim3(256)>>>(
            gpu->d_pData, gpu->d_R, gpu->d_pDataTemp, gpu->Nd);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Single merged UB+LB wavefront pass
    run_inner_bnb_wavefront(gpu, actual_k, optError_snap, mse_thresh, inlierNum,
                            init_tx, init_ty, init_tz, init_tw);

    // Truncation telemetry (non-fatal). A high count means the LB is loose and
    // the outer BnB will need more iterations — raise GPU_MAX_TRANS to tighten.
    {
        int trunc = 0;
        CUDA_CHECK(cudaMemcpy(&trunc, gpu->d_overflow_flag,
                              sizeof(int), cudaMemcpyDeviceToHost));
        if (trunc > 0) {
            static int warned = 0;
            if (warned < 8) {  // avoid log spam over the whole run
                fprintf(stderr,
                    "[GoICP GPU] note: inner trans-BnB truncated %d node(s) at "
                    "GPU_MAX_TRANS=%d. LB is valid but looser; raise "
                    "GPU_MAX_TRANS to tighten.\n", trunc, GPU_MAX_TRANS);
                warned++;
            }
        }
    }

    // Collect results: UB from d_ub_best, LB from d_rot_lb_best
    {
        int blk = (actual_k + 255) / 256;
        collect_results_kernel<<<blk, 256>>>(
            gpu->d_ub_best,
            gpu->d_rot_lb_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_rot_result, actual_k);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // D2H results
    CUDA_CHECK(cudaMemcpy(gpu->h_rot_result, gpu->d_rot_result,
                          sizeof(GpuRotResult) * actual_k,
                          cudaMemcpyDeviceToHost));
    for (int k = 0; k < actual_k; k++)
        results_cpu[k] = gpu->h_rot_result[k];
}

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

    // Upload point cloud
    CUDA_CHECK(cudaMalloc(&gpu->d_pData, sizeof(float) * Nd * 3));
    CUDA_CHECK(cudaMemcpy(gpu->d_pData, pData_xyz,
                          sizeof(float) * Nd * 3, cudaMemcpyHostToDevice));

    // Upload maxRotDis: flatten [MAXROTLEVEL][Nd] → [MAXROTLEVEL*Nd]
    {
        float* flat = (float*)malloc(sizeof(float) * GPU_MAXROTLEVEL * Nd);
        for (int l = 0; l < GPU_MAXROTLEVEL; l++)
            memcpy(flat + l*Nd, maxRotDis_cpu[l], sizeof(float)*Nd);
        CUDA_CHECK(cudaMalloc(&gpu->d_maxRotDis,
                              sizeof(float) * GPU_MAXROTLEVEL * Nd));
        CUDA_CHECK(cudaMemcpy(gpu->d_maxRotDis, flat,
                              sizeof(float) * GPU_MAXROTLEVEL * Nd,
                              cudaMemcpyHostToDevice));
        free(flat);
    }

    // Upload DT distance array to CUDA Array and bind texture object
    cudaExtent extent = make_cudaExtent(dt_size, dt_size, dt_size);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray(&gpu->d_distArray, &channelDesc, extent));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)dt_dist,
                                              dt_size*sizeof(float), dt_size, dt_size);
    copyParams.dstArray = gpu->d_distArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    cudaResourceDesc resDesc = {};
    resDesc.resType               = cudaResourceTypeArray;
    resDesc.res.array.array       = gpu->d_distArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]  = cudaAddressModeClamp;
    texDesc.addressMode[1]  = cudaAddressModeClamp;
    texDesc.addressMode[2]  = cudaAddressModeClamp;
    texDesc.filterMode      = cudaFilterModePoint;
    texDesc.readMode        = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_CHECK(cudaCreateTextureObject(&gpu->tex_dist, &resDesc, &texDesc, nullptr));

    GpuDT dt_host;
    dt_host.tex_dist = gpu->tex_dist;
    dt_host.size     = dt_size;
    dt_host.xMin     = dt_xMin;
    dt_host.yMin     = dt_yMin;
    dt_host.zMin     = dt_zMin;
    dt_host.scale    = dt_scale;
    dt_host.d_dist   = nullptr;
    gpu->dt = dt_host;
    CUDA_CHECK(cudaMemcpyToSymbol(c_dt, &dt_host, sizeof(GpuDT)));

    // pDataTemp [K * Nd * 3]
    CUDA_CHECK(cudaMalloc(&gpu->d_pDataTemp,
                          sizeof(float) * 3 * GPU_BATCH_K * Nd));

    // Rotation matrix buffer [K*9]
    CUDA_CHECK(cudaMalloc(&gpu->d_R, sizeof(float) * GPU_BATCH_K * 9));

    // Per-batch metadata
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_batch,  sizeof(GpuRotBatch)  * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_result, sizeof(GpuRotResult) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_levels, sizeof(int)          * GPU_BATCH_K));

    // Wavefront pools
    size_t pool_sz = sizeof(GpuTransNode) * GPU_BATCH_K * GPU_MAX_TRANS;
    CUDA_CHECK(cudaMalloc(&gpu->d_trans_pool_cur, pool_sz));
    CUDA_CHECK(cudaMalloc(&gpu->d_trans_pool_nxt, pool_sz));

    // Per-node counters and bounds
    CUDA_CHECK(cudaMalloc(&gpu->d_active_count,     sizeof(int)   * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_active_count_nxt, sizeof(int)   * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_ub_best,          sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_rot_lb_best,      sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_ub_global_best,   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_tx,          sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_ty,          sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_tz,          sizeof(float) * GPU_BATCH_K));
    CUDA_CHECK(cudaMalloc(&gpu->d_best_tw,          sizeof(float) * GPU_BATCH_K));

    // Truncation counter (was overflow flag)
    CUDA_CHECK(cudaMalloc(&gpu->d_overflow_flag, sizeof(int)));

    // Pinned host mirrors
    CUDA_CHECK(cudaMallocHost(&gpu->h_rot_batch,  sizeof(GpuRotBatch)  * GPU_BATCH_K));
    CUDA_CHECK(cudaMallocHost(&gpu->h_rot_result, sizeof(GpuRotResult) * GPU_BATCH_K));

    printf("[GPU] Init complete. Nd=%d, DT=%dx%dx%d, "
           "pDataTemp=%.1f MB, TransPool=%.1f MB\n",
           Nd, dt_size, dt_size, dt_size,
           3.f * GPU_BATCH_K * Nd * sizeof(float) / 1e6f,
           2.f * pool_sz / 1e6f);
}

extern "C" void GpuFree(GoICPGpu* gpu)
{
    if (gpu->d_pData)            cudaFree(gpu->d_pData);
    if (gpu->d_maxRotDis)        cudaFree(gpu->d_maxRotDis);
    if (gpu->d_pDataTemp)        cudaFree(gpu->d_pDataTemp);
    if (gpu->d_R)                cudaFree(gpu->d_R);
    if (gpu->tex_dist)           cudaDestroyTextureObject(gpu->tex_dist);
    if (gpu->d_distArray)        cudaFreeArray(gpu->d_distArray);
    if (gpu->d_rot_batch)        cudaFree(gpu->d_rot_batch);
    if (gpu->d_rot_result)       cudaFree(gpu->d_rot_result);
    if (gpu->d_rot_levels)       cudaFree(gpu->d_rot_levels);
    if (gpu->d_trans_pool_cur)   cudaFree(gpu->d_trans_pool_cur);
    if (gpu->d_trans_pool_nxt)   cudaFree(gpu->d_trans_pool_nxt);
    if (gpu->d_active_count)     cudaFree(gpu->d_active_count);
    if (gpu->d_active_count_nxt) cudaFree(gpu->d_active_count_nxt);
    if (gpu->d_ub_best)          cudaFree(gpu->d_ub_best);
    if (gpu->d_rot_lb_best)      cudaFree(gpu->d_rot_lb_best);
    if (gpu->d_ub_global_best)   cudaFree(gpu->d_ub_global_best);
    if (gpu->d_best_tx)          cudaFree(gpu->d_best_tx);
    if (gpu->d_best_ty)          cudaFree(gpu->d_best_ty);
    if (gpu->d_best_tz)          cudaFree(gpu->d_best_tz);
    if (gpu->d_best_tw)          cudaFree(gpu->d_best_tw);
    if (gpu->d_overflow_flag)    cudaFree(gpu->d_overflow_flag);
    if (gpu->h_rot_batch)        cudaFreeHost(gpu->h_rot_batch);
    if (gpu->h_rot_result)       cudaFreeHost(gpu->h_rot_result);
    memset(gpu, 0, sizeof(*gpu));
}

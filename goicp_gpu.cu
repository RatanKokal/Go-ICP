// =============================================================================
// goicp_gpu.cu  —  Method A CUDA implementation (rev 4)
//
// Rev 4 performance + correctness improvements over rev 3:
//
//  (P2+P3) STREAM PIPELINING + FUSED MAX KERNEL
//      A persistent cudaStream_t (gpu->stream) chains all kernels in a
//      wavefront batch without CPU round-trips.  The per-level D2H of K active
//      counts (which forced cudaDeviceSynchronize + K-int memcpy per level)
//      is replaced by a single fused compute_max_and_flag kernel that writes
//      {has_active, max_active} to a 2-int device buffer.  The CPU reads
//      exactly 8 bytes per wavefront level via one cudaMemcpyAsync +
//      cudaStreamSynchronize — replacing ~20M pipeline flushes per full run.
//
//  (P5) SoA INPUT LAYOUT
//      d_pData is now stored as SoA (x-block | y-block | z-block) instead of
//      AoS (x0 y0 z0 x1 y1 z1 ...).  rotate_points_kernel reads
//      d_pData[i], d_pData[Nd+i], d_pData[2*Nd+i] — stride-1 instead of
//      stride-3 — giving fully coalesced global loads (~3x bandwidth improvement
//      on the hottest read path in the implementation).
//      d_pDataTemp output and eval_trans_children input were already SoA
//      (unchanged from rev 3).
//
// Rev 3 correctness foundation (retained):
//  (A) Valid rotation LB under truncation — local_lb_rot subtracts both
//      rotDis and maxTransDis before recording, so even dropped children
//      contribute a valid (looser) LB.
//  (B) Non-fatal truncation — pool-full children are not pushed but their
//      LB contribution is recorded first.  d_overflow_flag is a telemetry
//      counter, not a fatal flag.
//  (C) Width leaf test — subdivision stops when maxTransDis^2 <= mse_thresh,
//      eliminating the fixed-60-level cap.
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
// Kernel 1: rotate_points_kernel  (P5: SoA input layout)
// Grid: (K, ceil(Nd/256))   Block: (256)
//
// d_pData layout (SoA): [x0..x_{Nd-1} | y0..y_{Nd-1} | z0..z_{Nd-1}]
// Consecutive threads read d_pData[i], d_pData[Nd+i], d_pData[2Nd+i]
// — stride 1 per coordinate block → fully coalesced global loads.
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
    // SoA read: x-block at [0..Nd), y-block at [Nd..2Nd), z-block at [2Nd..3Nd)
    float px = d_pData[i];
    float py = d_pData[Nd + i];
    float pz = d_pData[2*Nd + i];

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
// Kernel 7: compute_max_and_flag  (P2+P3 fix)
//
// Replaces the host-side loop + per-level D2H of K ints + DeviceSynchronize.
// One thread block, up to GPU_BATCH_K (256) threads.
//
// Reads d_counts[0..K-1] (the next-level active counts produced by
// eval_trans_children) and atomically computes:
//   d_ctrl[0] = has_active  (1 if any count > 0, else 0)
//   d_ctrl[1] = max_active  (max of all counts, clamped to GPU_MAX_TRANS)
//
// The CPU reads exactly 8 bytes (2 ints) per wavefront level via one
// cudaMemcpyAsync + cudaStreamSynchronize — replacing ~K ints + full pipeline
// drain per level.
// ---------------------------------------------------------------------------
__global__ void compute_max_and_flag(
    const int* __restrict__ d_counts,
    int* d_ctrl,
    int K)
{
    __shared__ int s_max;
    __shared__ int s_any;
    if (threadIdx.x == 0) { s_max = 0; s_any = 0; }
    __syncthreads();

    if (threadIdx.x < K) {
        int c = d_counts[threadIdx.x];
        if (c > 0) atomicOr(&s_any, 1);
        atomicMax(&s_max, c);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_ctrl[0] = s_any;
        d_ctrl[1] = (s_max > GPU_MAX_TRANS) ? GPU_MAX_TRANS : s_max;
    }
}

// ---------------------------------------------------------------------------
// run_inner_bnb_wavefront  (rev 4: stream + fused max kernel)
//
// Single merged pass: computes rotation UB (d_ub_best) and rotation LB
// (d_rot_lb_best) in one wavefront over the translation BnB tree.
//
// Per-level synchronization (P2+P3 fix):
//   OLD: cudaDeviceSynchronize() + cudaMemcpy D2H of K ints + CPU max-loop
//   NEW: compute_max_and_flag kernel (on stream) + cudaMemcpyAsync 8 bytes
//        + cudaStreamSynchronize — one sync, one tiny transfer, no CPU loop.
//
// All kernels run on gpu->stream so they are ordered without extra syncs.
// ---------------------------------------------------------------------------
static void run_inner_bnb_wavefront(
    GoICPGpu*    gpu,
    int          actual_k,
    float        optError_snap,
    float        mse_thresh,
    int          inlierNum,
    float        init_tx, float init_ty, float init_tz, float init_tw)
{
    cudaStream_t S = gpu->stream;

    // Reset per-outer-node state and trans pool (on stream, ordered)
    {
        int blocks = (actual_k + 255) / 256;
        init_batch_state<<<blocks, 256, 0, S>>>(
            gpu->d_ub_best,
            gpu->d_rot_lb_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_active_count,
            gpu->d_ub_global_best,
            optError_snap,
            init_tx, init_ty, init_tz, init_tw,
            actual_k);

        init_trans_pool<<<blocks, 256, 0, S>>>(
            gpu->d_trans_pool_cur,
            init_tx, init_ty, init_tz, init_tw,
            GPU_MAX_TRANS, actual_k);
    }

    // Seed the first iteration: root level has exactly 1 active node per k.
    // Use synchronous cudaMemcpy — seed lives on the stack (pageable memory);
    // cudaMemcpyAsync from pageable host memory is unsafe if source goes out
    // of scope before the stream executes the transfer.
    {
        int seed[2] = {1, 1};   // has_active=1, max_active=1
        CUDA_CHECK(cudaMemcpy(gpu->d_wavefront_ctrl, seed,
                              2*sizeof(int), cudaMemcpyHostToDevice));
    }

    const int max_levels = 60;  // safety bound; leaf test normally terminates first
    for (int level = 0; level < max_levels; level++)
    {
        // Read the control word written by compute_max_and_flag (or the seed above).
        // This is the ONLY synchronisation point per level — replaces
        // cudaDeviceSynchronize + K-element D2H + CPU max-loop.
        int h_ctrl[2];
        CUDA_CHECK(cudaMemcpyAsync(h_ctrl, gpu->d_wavefront_ctrl,
                                  2*sizeof(int), cudaMemcpyDeviceToHost, S));
        CUDA_CHECK(cudaStreamSynchronize(S));

        int has_active = h_ctrl[0];
        int max_count  = h_ctrl[1];   // already clamped to GPU_MAX_TRANS by kernel

        if (!has_active || max_count == 0) break;

        // Zero next-level counts and ctrl buffer before expansion
        {
            int blk = (actual_k + 255) / 256;
            zero_next_counts<<<blk, 256, 0, S>>>(gpu->d_active_count_nxt, actual_k);
        }
        // Reset ctrl so compute_max_and_flag starts from 0
        CUDA_CHECK(cudaMemsetAsync(gpu->d_wavefront_ctrl, 0, 2*sizeof(int), S));

        // 2D grid: (actual_k, max_count <= GPU_MAX_TRANS)
        dim3 grid(actual_k, max_count);
        dim3 block(EVAL_BLOCK_N);

        eval_trans_children<<<grid, block, 0, S>>>(
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
            gpu->d_overflow_flag,         // truncation counter
            optError_snap,
            mse_thresh,
            gpu->Nd,
            inlierNum,
            GPU_MAX_TRANS);

        // Fused kernel: compute has_active + max_active from next-level counts.
        // Runs on stream S immediately after eval; result ready for next iteration.
        compute_max_and_flag<<<1, GPU_BATCH_K, 0, S>>>(
            gpu->d_active_count_nxt, gpu->d_wavefront_ctrl, actual_k);

        // Swap cur/nxt pools and counts (pointer swap on host, no data movement)
        {
            GpuTransNode* tmp_pool   = gpu->d_trans_pool_cur;
            int*          tmp_counts = gpu->d_active_count;
            gpu->d_trans_pool_cur    = gpu->d_trans_pool_nxt;
            gpu->d_active_count      = gpu->d_active_count_nxt;
            gpu->d_trans_pool_nxt    = tmp_pool;
            gpu->d_active_count_nxt  = tmp_counts;
        }
        // Note: d_active_count now points to the nxt pool's counts (the current
        // level for the next iteration). No host read of counts needed.
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
    cudaStream_t S = gpu->stream;

    // H2D: rotation matrices and levels (synchronous — sources are stack-
    // allocated pageable memory; cudaMemcpyAsync requires pinned memory).
    // Total transfer: GPU_BATCH_K*9*4 + GPU_BATCH_K*4 = ~2.3 KB — negligible.
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

    // Reset truncation counter before wavefront (on stream, ordered after H2D)
    CUDA_CHECK(cudaMemsetAsync(gpu->d_overflow_flag, 0, sizeof(int), S));

    // Rotate all Nd points for all K outer nodes (on stream — no explicit sync;
    // run_inner_bnb_wavefront uses the same stream, so ordering is guaranteed)
    {
        dim3 grid(actual_k, (gpu->Nd + 255) / 256);
        rotate_points_kernel<<<grid, dim3(256), 0, S>>>(
            gpu->d_pData, gpu->d_R, gpu->d_pDataTemp, gpu->Nd);
        // No cudaDeviceSynchronize here — eval_trans_children in the wavefront
        // runs on the same stream and will see the rotated points.
    }

    // Single merged UB+LB wavefront pass (uses gpu->stream internally)
    run_inner_bnb_wavefront(gpu, actual_k, optError_snap, mse_thresh, inlierNum,
                            init_tx, init_ty, init_tz, init_tw);
    // After run_inner_bnb_wavefront returns, the stream has been synchronised
    // at least once (inside the wavefront loop's per-level sync), so
    // d_overflow_flag is readable without an extra sync.

    // Truncation telemetry (non-fatal). A high percentage means GPU_MAX_TRANS
    // is the binding constraint on LB tightness; raise it to reduce outer-BnB
    // iterations at the cost of linear memory growth.
    {
        int trunc = 0;
        CUDA_CHECK(cudaMemcpy(&trunc, gpu->d_overflow_flag,
                              sizeof(int), cudaMemcpyDeviceToHost));
        if (trunc > 0) {
            static int warned = 0;
            if (warned < 8) {  // avoid log spam over the whole run
                float pct = 100.f * (float)trunc /
                            (float)((long long)actual_k * GPU_MAX_TRANS);
                fprintf(stderr,
                    "[GoICP GPU] note: inner trans-BnB truncated %d node(s) "
                    "(%.2f%% of K=%d x GPU_MAX_TRANS=%d capacity). "
                    "LB is valid but looser — raise GPU_MAX_TRANS to tighten.\n",
                    trunc, pct, actual_k, GPU_MAX_TRANS);
                warned++;
            }
        }
    }

    // Collect results on stream, then async D2H, then one final stream sync
    {
        int blk = (actual_k + 255) / 256;
        collect_results_kernel<<<blk, 256, 0, S>>>(
            gpu->d_ub_best,
            gpu->d_rot_lb_best,
            gpu->d_best_tx, gpu->d_best_ty, gpu->d_best_tz, gpu->d_best_tw,
            gpu->d_rot_result, actual_k);
        CUDA_CHECK(cudaMemcpyAsync(gpu->h_rot_result, gpu->d_rot_result,
                                  sizeof(GpuRotResult) * actual_k,
                                  cudaMemcpyDeviceToHost, S));
        CUDA_CHECK(cudaStreamSynchronize(S));  // wait for results to land in host
    }

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

    // P2+P3: create the persistent compute stream used by all batch kernels
    CUDA_CHECK(cudaStreamCreate(&gpu->stream));

    // Upload point cloud in SoA layout (P5 fix: coalesced reads in rotate kernel)
    // Input pData_xyz is AoS: [x0 y0 z0 x1 y1 z1 ...]
    // Stored as SoA:  [x0..x_{Nd-1} | y0..y_{Nd-1} | z0..z_{Nd-1}]
    CUDA_CHECK(cudaMalloc(&gpu->d_pData, sizeof(float) * Nd * 3));
    {
        float* pData_soa = (float*)malloc(sizeof(float) * Nd * 3);
        if (!pData_soa) { fprintf(stderr, "GpuInit: malloc pData_soa failed\n"); exit(1); }
        for (int i = 0; i < Nd; i++) {
            pData_soa[i]          = pData_xyz[i*3+0];   // x-block
            pData_soa[Nd   + i]   = pData_xyz[i*3+1];   // y-block
            pData_soa[2*Nd + i]   = pData_xyz[i*3+2];   // z-block
        }
        CUDA_CHECK(cudaMemcpy(gpu->d_pData, pData_soa,
                              sizeof(float) * Nd * 3, cudaMemcpyHostToDevice));
        free(pData_soa);
    }

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

    // P2+P3: fused max/flag buffer — {has_active[0], max_active[1]}
    CUDA_CHECK(cudaMalloc(&gpu->d_wavefront_ctrl, sizeof(int) * 2));

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
    // Synchronise stream before freeing resources it may still reference
    if (gpu->stream)             cudaStreamSynchronize(gpu->stream);

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
    if (gpu->d_wavefront_ctrl)   cudaFree(gpu->d_wavefront_ctrl);
    if (gpu->h_rot_batch)        cudaFreeHost(gpu->h_rot_batch);
    if (gpu->h_rot_result)       cudaFreeHost(gpu->h_rot_result);
    if (gpu->stream)             cudaStreamDestroy(gpu->stream);
    memset(gpu, 0, sizeof(*gpu));
}

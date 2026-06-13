#pragma once
// =============================================================================
// goicp_gpu.cuh  —  Method A GPU declarations (rev 3)
//
// Revision 3 changes vs rev 2:
//   - GpuTask struct removed; 2D grid (k=blockIdx.x, s=blockIdx.y) eliminates
//     CPU-side task compaction and the H2D task upload on every wavefront level.
//   - UB and LB inner-BnB wavefront passes merged into one.  eval_trans_children
//     computes local_ub (rotation UB) and local_lb_rot (rotation LB) in the
//     same point loop.  GpuRunBatch calls run_inner_bnb_wavefront once.
//   - d_rot_lb_best[K] replaces the formerly-unused d_lb_best[K].
//   - d_overflow_flag repurposed as a truncation counter (telemetry only);
//     overflow is non-fatal — LB remains valid, just looser.
//   - GPU_MAX_TRANS raised to 4096 (pool memory ~42 MB, safe on T4/A100).
//   - P2+P3: persistent cudaStream_t; fused compute_max_and_flag kernel
//     replaces per-level cudaDeviceSynchronize + K-element D2H.
//   - P5: d_pData stored in SoA layout for coalesced reads in rotate kernel.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants (mirror jly_goicp.h exactly)
// ---------------------------------------------------------------------------
#define GPU_PI         3.1415926536f
#define GPU_SQRT3      1.732050808f
#define GPU_MAXROTLEVEL 20
#define GPU_BATCH_K    256        // outer nodes per batch
#define GPU_MAX_TRANS  4096       // max active trans nodes per outer node per level

// ---------------------------------------------------------------------------
// Flat SoA transfer structs for H2D per batch
// ---------------------------------------------------------------------------

struct GpuRotBatch {
    float R[9];     // row-major rotation matrix
    float a, b, c;  // cube corner (angle-axis)
    float w;        // cube half-width
    int   l;        // child level — indexes maxRotDis[l]
};

struct GpuRotResult {
    float ub;           // best upper bound (min local_ub over all trans)
    float lb;           // rotation lower bound (min local_lb_rot over all trans)
    float best_tx;      // translation cube CORNER x of best ub node
    float best_ty;
    float best_tz;
    float best_tw;
};

// ---------------------------------------------------------------------------
// Active translation node pool
// Layout: pool[k * GPU_MAX_TRANS + slot]
// ---------------------------------------------------------------------------
struct GpuTransNode {
    float x, y, z, w;
    float lb_parent;
};

// ---------------------------------------------------------------------------
// Device-side DT descriptor
// ---------------------------------------------------------------------------
struct GpuDT {
    float* d_dist;               // unused direct pointer (texture is preferred)
    cudaTextureObject_t tex_dist;
    int    size;
    float  xMin, yMin, zMin;
    float  scale;
};

// ---------------------------------------------------------------------------
// GPU context — owns all persistent device allocations
// ---------------------------------------------------------------------------
struct GoICPGpu {
    // Persistent
    float*         d_pData;           // [Nd*3]
    float*         d_maxRotDis;       // [MAXROTLEVEL * Nd]
    float*         d_pDataTemp;       // [K * Nd * 3]
    float*         d_R;               // [K * 9]
    GpuDT          dt;
    cudaArray_t    d_distArray;       // backing store for texture
    cudaTextureObject_t tex_dist;

    // Per-batch transfers
    GpuRotBatch*   d_rot_batch;       // [K]
    GpuRotResult*  d_rot_result;      // [K]
    int*           d_rot_levels;      // [K]

    // Wavefront working memory
    GpuTransNode*  d_trans_pool_cur;  // [K * GPU_MAX_TRANS]
    GpuTransNode*  d_trans_pool_nxt;  // [K * GPU_MAX_TRANS]
    int*           d_active_count;    // [K]  current level active counts
    int*           d_active_count_nxt;// [K]  next level active counts
    float*         d_ub_best;         // [K]  running best UB per outer node
    float*         d_rot_lb_best;     // [K]  running best rotation LB per outer node
    float*         d_ub_global_best;  // [1]  cross-node best UB (drives pruning)
    float*         d_best_tx;         // [K]  translation corner for best UB
    float*         d_best_ty;
    float*         d_best_tz;
    float*         d_best_tw;

    // Truncation counter (incremented by kernel when GPU_MAX_TRANS exceeded)
    int*           d_overflow_flag;   // [1]

    // P2+P3: persistent stream + fused max/flag buffer
    // d_wavefront_ctrl[0] = has_active (any count > 0)
    // d_wavefront_ctrl[1] = max_active (clamped to GPU_MAX_TRANS)
    cudaStream_t   stream;            // all wavefront kernels run on this stream
    int*           d_wavefront_ctrl;  // [2] written by compute_max_and_flag kernel

    // Pinned host mirrors
    GpuRotBatch*   h_rot_batch;       // pinned [K]
    GpuRotResult*  h_rot_result;      // pinned [K]

    int Nd;
    int K;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

void GpuInit(GoICPGpu* gpu, int Nd,
             const float* pData_xyz,
             float** maxRotDis_cpu,
             const float* dt_dist,
             int dt_size,
             float dt_xMin, float dt_yMin, float dt_zMin,
             float dt_scale);

void GpuRunBatch(GoICPGpu* gpu,
                 const GpuRotBatch* rot_batch_cpu,
                 int actual_k,
                 float optError_snap,
                 int inlierNum,
                 float init_tx, float init_ty, float init_tz, float init_tw,
                 float mse_thresh,            // per-point MSE tolerance (trans leaf test)
                 GpuRotResult* results_cpu);

void GpuFree(GoICPGpu* gpu);

#ifdef __cplusplus
}
#endif

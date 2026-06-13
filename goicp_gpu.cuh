#pragma once
// =============================================================================
// goicp_gpu.cuh  —  Method A GPU declarations
//
// Design decisions grounded in source analysis:
//
// ROTNODE.l is the CHILD level (parent.l + 1), so maxRotDis[nodeRot.l]
// is the correct index for the lb kernel.
//
// initNodeTrans: x,y,z = cube corner (NOT centre); centre = x+w/2.
// The hot loop uses pDataTemp[i] + transX where transX = nodeTrans.x + w/2.
//
// doTrim=false for Method A: inlierNum == Nd, intro_select path is dead.
// All reductions sum over ALL Nd points.
//
// SQRT3 constant matches source exactly (1.732050808) — we do NOT correct
// the rounding-down here; that is a Method C concern. Method A reproduces
// CPU behaviour identically.
//
// Memory layout: d_pDataTemp[k][i] → flat index k*Nd*3 + i*3 + coord
// Ensures warp-coalesced reads when threads iterate over i with fixed k.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants (mirror jly_goicp.h exactly)
// ---------------------------------------------------------------------------
#define GPU_PI      3.1415926536f
#define GPU_SQRT3   1.732050808f
#define GPU_MAXROTLEVEL 20
#define GPU_BATCH_K     256       // outer nodes per batch
#define GPU_MAX_TRANS   1024      // max active trans nodes per outer node
                                  // per wavefront level; overflow → CPU fallback

// ---------------------------------------------------------------------------
// Flat SoA transfer structs for H2D per batch
// ---------------------------------------------------------------------------

// One entry per outer rotation node in the batch
struct GpuRotBatch {
    float R[9];     // row-major rotation matrix [0..8] = R11,R12,R13,R21,...
    float a, b, c;  // cube corner (angle-axis)
    float w;        // cube half-width (child width, not parent)
    int   l;        // child level — indexes maxRotDis[l]
    // Filled by CPU before H2D, used by lb kernel
};

// Per outer node result returned D2H
struct GpuRotResult {
    float ub;           // best upper bound found by inner BnB
    float lb;           // lower bound found by inner BnB
    float best_tx;      // translation cube CORNER x of best ub node
    float best_ty;
    float best_tz;
    float best_tw;      // translation cube width of best ub node
    // Centre = best_tx + best_tw/2, etc. — mirrors what CPU stores in nodeTrans
};

// ---------------------------------------------------------------------------
// Active translation node pool (per outer node, per wavefront level)
// Stored in global memory; one pool per outer node k.
// Layout: pool[k * GPU_MAX_TRANS + slot] = {x, y, z, w, lb_parent}
// ---------------------------------------------------------------------------
struct GpuTransNode {
    float x, y, z, w;
    float lb_parent;    // lb of this node (used only for debugging/diagnostics)
};

// ---------------------------------------------------------------------------
// Compacted Task structure for 1D kernel launch
// ---------------------------------------------------------------------------
struct GpuTask {
    int k;  // outer node index
    int s;  // slot index in the outer node's trans pool
};

// ---------------------------------------------------------------------------
// Device-side DT descriptor (mirrors DT3D but device-accessible)
// Populated once from DT3D after BuildDT()
// Note: DT3D::scale is double; we store as float here (sufficient precision
// for index computation at the grid resolutions used in practice).
// ---------------------------------------------------------------------------
struct GpuDT {
    float* d_dist;      // flat [SIZE][SIZE][SIZE] distance array, row-major z,y,x
    cudaTextureObject_t tex_dist; // Texture object for spatial lookups
    int    size;        // SIZE (300 by default)
    float  xMin, yMin, zMin;
    float  scale;       // (SIZE-1) / (max_coord - min_coord)
};

// ---------------------------------------------------------------------------
// GPU context — owns all persistent device allocations
// ---------------------------------------------------------------------------
struct GoICPGpu {
    // Persistent (allocated in GpuInit, freed in GpuFree)
    float*         d_pData;          // [Nd*3]  original data points, never changes
    float*         d_maxRotDis;      // [MAXROTLEVEL * Nd]  flattened
    float*         d_pDataTemp;      // [K * Nd * 3]  rotated points per batch
    float*         d_R;              // [K * 9]  rotation matrices for current batch
    GpuDT          dt;               // DT on device

    // Per-batch transfers (allocated once, reused)
    GpuRotBatch*   d_rot_batch;      // [K]  H2D
    GpuRotResult*  d_rot_result;     // [K]  D2H
    int*           d_rot_levels;     // [K]  per-node rotation level for lb pass

    // Wavefront working memory
    GpuTransNode*  d_trans_pool_cur; // [K * GPU_MAX_TRANS]
    GpuTransNode*  d_trans_pool_nxt; // [K * GPU_MAX_TRANS]
    int*           d_active_count;   // [K]  active node count per outer node
    int*           d_active_count_nxt;
    float*         d_ub_best;        // [K]  running best ub per outer node
    float*         d_ub_global_best; // [1]  running global best ub across all outer nodes
    float*         d_lb_best;        // [K]  running best lb (unused in Method A)
    float*         d_best_tx;        // [K]  best trans corner
    float*         d_best_ty;
    float*         d_best_tz;
    float*         d_best_tw;

    // Termination flag (persistent — avoids malloc/free per wavefront level)
    int*           d_flag;           // [1]  used by check_any_active

    // Task compaction buffers
    GpuTask*       d_tasks;          // [K * GPU_MAX_TRANS] device
    GpuTask*       h_tasks;          // [K * GPU_MAX_TRANS] pinned host

    // Pinned host mirrors for async transfers
    GpuRotBatch*   h_rot_batch;      // pinned [K]
    GpuRotResult*  h_rot_result;     // pinned [K]

    int Nd;
    int K;
};

// ---------------------------------------------------------------------------
// Public API called from jly_goicp_gpu.cpp
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Call after BuildDT() and Initialize() complete on CPU.
// Uploads pData, maxRotDis, and DT to device.
void GpuInit(GoICPGpu* gpu, int Nd,
             const float* pData_xyz,       // [Nd*3] interleaved x,y,z
             float** maxRotDis_cpu,        // [MAXROTLEVEL][Nd]
             // DT fields:
             const float* dt_dist,         // [SIZE^3] row-major z,y,x
             int dt_size,
             float dt_xMin, float dt_yMin, float dt_zMin,
             float dt_scale);

// Process one batch of K rotation nodes.
// rot_batch_cpu: array of K GpuRotBatch structs (angle-axis corners + R matrices)
// actual_k:      actual batch size (<= K, handles last batch)
// optError_snap: current best error at batch start (used for pruning)
// inlierNum:     Nd for untrimmed
// initTrans:     {x,y,z,w} of initNodeTrans from GoICP
// results_cpu:   output array [actual_k] filled on return
void GpuRunBatch(GoICPGpu* gpu,
                 const GpuRotBatch* rot_batch_cpu,
                 int actual_k,
                 float optError_snap,
                 int inlierNum,
                 float init_tx, float init_ty, float init_tz, float init_tw,
                 GpuRotResult* results_cpu);

void GpuFree(GoICPGpu* gpu);

#ifdef __cplusplus
}
#endif

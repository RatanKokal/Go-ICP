// =============================================================================
// jly_goicp_gpu.cpp  —  CPU host: OuterBnB_GPU()
//
// This file provides OuterBnB_GPU() which replaces OuterBnB() in jly_goicp.cpp.
// Everything else (Initialize, ICP, BuildDT, Clear) is unchanged.
//
// The batch loop mirrors the CPU OuterBnB exactly:
//   - same queue structure (std::priority_queue<ROTNODE>)
//   - same convergence check
//   - same ICP trigger condition
//   - same queue prune logic
//   - same child expansion (angle-axis → R matrix)
//
// The only structural difference: instead of evaluating children one at a time,
// we accumulate up to GPU_BATCH_K children into a batch and dispatch to GPU.
//
// IMPORTANT MAPPING from source analysis:
//   nodeRot.l = parent.l + 1           (child level)
//   maxRotDis[nodeRot.l]               (lb pass uses child level)
//   optT = nodeTrans.x + nodeTrans.w/2 (translation CENTRE, not corner)
//   InnerBnB UB pass: maxRotDisL = NULL → GpuRunBatch pass 1
//   InnerBnB LB pass: maxRotDisL = maxRotDis[l] → GpuRunBatch pass 2
// =============================================================================

#include "jly_goicp.h"        // original GoICP class (has friend declaration)
#include "goicp_gpu.cuh"      // GpuRotBatch, GpuRotResult, GoICPGpu, GpuRunBatch

#include <queue>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <algorithm>

using namespace std;

// ---------------------------------------------------------------------------
// Helper: compute rotation matrix from angle-axis (v1,v2,v3 = cube centre)
// Mirrors OuterBnB verbatim. Returns false if the node is outside the PI-ball.
// ---------------------------------------------------------------------------
static bool angle_axis_to_R(
    float v1, float v2, float v3,  // cube centre
    float w,                        // half-width (for PI-ball check)
    float R[9])                     // row-major output
{
    float norm_v = sqrtf(v1*v1 + v2*v2 + v3*v3);
    if (norm_v - (float)SQRT3 * w * 0.5f > (float)PI)
        return false;

    float t = norm_v;
    if (t > 0.f) {
        v1 /= t; v2 /= t; v3 /= t;

        float ct  = cosf(t);
        float ct2 = 1.f - ct;
        float st  = sinf(t);

        float tmp121 = v1*v2*ct2, tmp122 = v3*st;
        float tmp131 = v1*v3*ct2, tmp132 = v2*st;
        float tmp231 = v2*v3*ct2, tmp232 = v1*st;

        R[0] = ct + v1*v1*ct2;       R[1] = tmp121 - tmp122;  R[2] = tmp131 + tmp132;
        R[3] = tmp121 + tmp122;       R[4] = ct + v2*v2*ct2;  R[5] = tmp231 - tmp232;
        R[6] = tmp131 - tmp132;       R[7] = tmp231 + tmp232;  R[8] = ct + v3*v3*ct2;
    } else {
        // identity rotation
        R[0]=1.f; R[1]=0.f; R[2]=0.f;
        R[3]=0.f; R[4]=1.f; R[5]=0.f;
        R[6]=0.f; R[7]=0.f; R[8]=1.f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// OuterBnB_GPU — drop-in replacement for GoICP::OuterBnB()
//
// Called from RegisterGPU() in jly_main_gpu.cpp.
//
// Access to private GoICP members is granted via:
//   friend float OuterBnB_GPU(GoICP&, GoICPGpu&);
// declared in jly_goicp.h.
// ---------------------------------------------------------------------------
float OuterBnB_GPU(
    GoICP&     goicp,   // full access via friend declaration
    GoICPGpu&  gpu_ctx)
{
    // ---- Mirror OuterBnB initial error calculation ----
    // CPU version uses minDis[] + doTrim; we use DT directly (doTrim=false for
    // GPU path as documented — inlierNum == Nd).
    goicp.optError = 0;
    for (int i = 0; i < goicp.Nd; i++) {
        float d = goicp.dt.Distance(
            goicp.pData[i].x, goicp.pData[i].y, goicp.pData[i].z);
        goicp.optError += d * d;
    }
    printf("Error*: %f (Init)\n", goicp.optError);

    // ---- Initial ICP from identity rotation/translation ----
    {
        clock_t t0 = clock();
        Matrix R_icp = goicp.optR;  // identity (set in Initialize)
        Matrix t_icp = goicp.optT;  // zero
        float error = goicp.ICP(R_icp, t_icp);  // direct access via friend
        if (error < goicp.optError) {
            goicp.optError = error;
            goicp.optR = R_icp;
            goicp.optT = t_icp;
            printf("Error*: %f (ICP %.2fs)\n", goicp.optError,
                   (double)(clock()-t0)/CLOCKS_PER_SEC);
        }
    }

    // ---- Priority queue (identical to OuterBnB) ----
    priority_queue<ROTNODE> queueRot;
    queueRot.push(goicp.initNodeRot);

    // Batch accumulation buffers (static to avoid stack pressure)
    static GpuRotBatch  batch_in[GPU_BATCH_K];
    static ROTNODE      batch_nodes[GPU_BATCH_K];
    static GpuRotResult batch_out[GPU_BATCH_K];

    long long count = 0;

    while (true) {
        if (queueRot.empty()) {
            printf("[GPU] Rotation queue empty. optError=%f\n", goicp.optError);
            break;
        }

        // ---- Convergence check (must happen BEFORE popping) ----
        {
            ROTNODE top = queueRot.top();
            if ((goicp.optError - top.lb) <= goicp.SSEThresh) {
                printf("Error*: %f, LB: %f, epsilon: %f\n",
                       goicp.optError, top.lb, goicp.SSEThresh);
                break;
            }
        }

        // ---- Pop parents and expand children until batch is full ----
        // We collect children (not parents) into the batch.
        // Each parent produces 0..8 valid children after PI-ball rejection.
        int batch_size = 0;

        while (batch_size < GPU_BATCH_K && !queueRot.empty()) {
            // Re-check convergence per parent (optError may have improved)
            ROTNODE parent = queueRot.top();
            if ((goicp.optError - parent.lb) <= goicp.SSEThresh)
                break;
            queueRot.pop();

            if (count > 0 && count % 300 == 0)
                printf("LB=%f  L=%d\n", parent.lb, parent.l);
            count++;

            ROTNODE child_proto;
            child_proto.w = parent.w * 0.5f;
            child_proto.l = parent.l + 1;

            for (int j = 0; j < 8 && batch_size < GPU_BATCH_K; j++) {
                child_proto.a = parent.a + (float)(j & 1)       * child_proto.w;
                child_proto.b = parent.b + (float)((j>>1) & 1)  * child_proto.w;
                child_proto.c = parent.c + (float)((j>>2) & 1)  * child_proto.w;

                // Cube centre
                float v1 = child_proto.a + child_proto.w * 0.5f;
                float v2 = child_proto.b + child_proto.w * 0.5f;
                float v3 = child_proto.c + child_proto.w * 0.5f;

                float R[9];
                if (!angle_axis_to_R(v1, v2, v3, child_proto.w, R))
                    continue;  // PI-ball rejection

                batch_nodes[batch_size] = child_proto;

                GpuRotBatch& entry = batch_in[batch_size];
                memcpy(entry.R, R, sizeof(float)*9);
                entry.a = child_proto.a;
                entry.b = child_proto.b;
                entry.c = child_proto.c;
                entry.w = child_proto.w;
                entry.l = child_proto.l;  // child level — used for maxRotDis indexing

                batch_size++;
            }
        }

        if (batch_size == 0) break;

        // ---- GPU batch evaluation ----
        GpuRunBatch(
            &gpu_ctx,
            batch_in,
            batch_size,
            goicp.optError,            // optError_snap (pruning bound)
            goicp.inlierNum,
            goicp.initNodeTrans.x,
            goicp.initNodeTrans.y,
            goicp.initNodeTrans.z,
            goicp.initNodeTrans.w,
            batch_out);

        // ---- Process results: mirrors OuterBnB's per-child logic ----

        // Step 1: find best UB in batch and update optError / optR / optT
        {
            float best_ub = goicp.optError;
            int   best_k  = -1;
            for (int k = 0; k < batch_size; k++) {
                if (batch_out[k].ub < best_ub) {
                    best_ub = batch_out[k].ub;
                    best_k  = k;
                }
            }

            if (best_k >= 0) {
                goicp.optError = best_ub;

                // Set optR from the winning batch node's rotation matrix
                const float* R = batch_in[best_k].R;
                goicp.optR.val[0][0]=R[0]; goicp.optR.val[0][1]=R[1]; goicp.optR.val[0][2]=R[2];
                goicp.optR.val[1][0]=R[3]; goicp.optR.val[1][1]=R[4]; goicp.optR.val[1][2]=R[5];
                goicp.optR.val[2][0]=R[6]; goicp.optR.val[2][1]=R[7]; goicp.optR.val[2][2]=R[8];

                // Set optT from best translation found (cube CENTRE, not corner)
                float btx = batch_out[best_k].best_tx + batch_out[best_k].best_tw * 0.5f;
                float bty = batch_out[best_k].best_ty + batch_out[best_k].best_tw * 0.5f;
                float btz = batch_out[best_k].best_tz + batch_out[best_k].best_tw * 0.5f;
                goicp.optT.val[0][0] = btx;
                goicp.optT.val[1][0] = bty;
                goicp.optT.val[2][0] = btz;

                printf("Error*: %f\n", goicp.optError);

                // Step 2: Run ICP from the new best (R, T) — CPU ICP
                {
                    clock_t t0 = clock();
                    Matrix R_icp = goicp.optR;
                    Matrix t_icp = goicp.optT;
                    float error = goicp.ICP(R_icp, t_icp);  // friend access
                    if (error < goicp.optError) {
                        goicp.optError = error;
                        goicp.optR     = R_icp;
                        goicp.optT     = t_icp;
                        printf("Error*: %f (ICP %.2fs)\n", goicp.optError,
                               (double)(clock()-t0)/CLOCKS_PER_SEC);
                    }
                }

                // Step 3: Prune rotation queue with updated optError.
                // CPU uses heap property for early break — reproduced here.
                {
                    priority_queue<ROTNODE> queueRotNew;
                    while (!queueRot.empty()) {
                        ROTNODE node = queueRot.top();
                        queueRot.pop();
                        if (node.lb < goicp.optError)
                            queueRotNew.push(node);
                        else
                            break;  // remaining have lb >= optError (heap property)
                    }
                    queueRot = queueRotNew;
                }
            }
        }

        // Step 4: Push children from this batch whose lb < optError into queue
        for (int k = 0; k < batch_size; k++) {
            if (batch_out[k].lb >= goicp.optError)
                continue;
            ROTNODE& n = batch_nodes[k];
            n.lb = batch_out[k].lb;
            n.ub = batch_out[k].ub;
            queueRot.push(n);
        }
    }

    return goicp.optError;
}

// =============================================================================
// jly_goicp_gpu.cpp  —  CPU host: OuterBnB_GPU()  (rev 2)
//
// Changes from rev 1:
//   - GpuRunBatch now returns both .ub and .lb in one call (merged pass);
//     no separate h_lb read needed here.
//   - ICP is deferred: called only when optError drops by more than
//     ICP_IMPROVEMENT_THRESH relative to the last ICP run.  This prevents
//     the GPU from sitting idle on every marginal improvement during the
//     fine-grained tail of BnB, where improvements shrink rapidly.
// =============================================================================

#include "jly_goicp.h"
#include "goicp_gpu.cuh"

#include <queue>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <algorithm>

using namespace std;

// Only call ICP when optError improved by at least this fraction since
// the last ICP run.  0.005 = 0.5% — avoids trivial repeated ICP calls
// during the fine tail of BnB while still running ICP whenever there is a
// meaningful improvement.
static const float ICP_IMPROVEMENT_THRESH = 0.005f;

// ---------------------------------------------------------------------------
// Helper: angle-axis cube centre → rotation matrix
// Returns false if the cube is entirely outside the PI-ball.
// ---------------------------------------------------------------------------
static bool angle_axis_to_R(
    float v1, float v2, float v3,
    float w,
    float R[9])
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
        R[0]=1.f; R[1]=0.f; R[2]=0.f;
        R[3]=0.f; R[4]=1.f; R[5]=0.f;
        R[6]=0.f; R[7]=0.f; R[8]=1.f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// OuterBnB_GPU
// ---------------------------------------------------------------------------
float OuterBnB_GPU(GoICP& goicp, GoICPGpu& gpu_ctx)
{
    // Initial error from identity transform
    goicp.optError = 0;
    for (int i = 0; i < goicp.Nd; i++) {
        float d = goicp.dt.Distance(
            goicp.pData[i].x, goicp.pData[i].y, goicp.pData[i].z);
        goicp.optError += d * d;
    }
    printf("Error*: %f (Init)\n", goicp.optError);

    // Initial ICP from identity
    float last_icp_error = goicp.optError;
    {
        clock_t t0 = clock();
        Matrix R_icp = goicp.optR;
        Matrix t_icp = goicp.optT;
        float error = goicp.ICP(R_icp, t_icp);
        if (error < goicp.optError) {
            goicp.optError = error;
            goicp.optR     = R_icp;
            goicp.optT     = t_icp;
            last_icp_error = error;
            printf("Error*: %f (ICP %.2fs)\n", goicp.optError,
                   (double)(clock()-t0)/CLOCKS_PER_SEC);
        }
    }

    priority_queue<ROTNODE> queueRot;
    queueRot.push(goicp.initNodeRot);

    static GpuRotBatch  batch_in[GPU_BATCH_K];
    static ROTNODE      batch_nodes[GPU_BATCH_K];
    static GpuRotResult batch_out[GPU_BATCH_K];

    long long count = 0;

    while (true) {
        if (queueRot.empty()) {
            printf("[GPU] Rotation queue empty. optError=%f\n", goicp.optError);
            break;
        }

        // Convergence check before popping
        if ((goicp.optError - queueRot.top().lb) <= goicp.SSEThresh) {
            ROTNODE top = queueRot.top();
            printf("Error*: %f, LB: %f, epsilon: %f\n",
                   goicp.optError, top.lb, goicp.SSEThresh);
            break;
        }

        // Collect children from the queue into a batch
        int batch_size = 0;
        while (batch_size < GPU_BATCH_K && !queueRot.empty()) {
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

                float v1 = child_proto.a + child_proto.w * 0.5f;
                float v2 = child_proto.b + child_proto.w * 0.5f;
                float v3 = child_proto.c + child_proto.w * 0.5f;

                float R[9];
                if (!angle_axis_to_R(v1, v2, v3, child_proto.w, R))
                    continue;

                batch_nodes[batch_size] = child_proto;

                GpuRotBatch& entry = batch_in[batch_size];
                memcpy(entry.R, R, sizeof(float)*9);
                entry.a = child_proto.a;
                entry.b = child_proto.b;
                entry.c = child_proto.c;
                entry.w = child_proto.w;
                entry.l = child_proto.l;

                batch_size++;
            }
        }

        if (batch_size == 0) break;

        // GPU batch evaluation — returns both .ub and .lb from merged pass
        GpuRunBatch(
            &gpu_ctx,
            batch_in,
            batch_size,
            goicp.optError,
            goicp.inlierNum,
            goicp.initNodeTrans.x,
            goicp.initNodeTrans.y,
            goicp.initNodeTrans.z,
            goicp.initNodeTrans.w,
            batch_out);

        // ---- Step 1: find best UB in batch ----
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

                const float* R = batch_in[best_k].R;
                goicp.optR.val[0][0]=R[0]; goicp.optR.val[0][1]=R[1]; goicp.optR.val[0][2]=R[2];
                goicp.optR.val[1][0]=R[3]; goicp.optR.val[1][1]=R[4]; goicp.optR.val[1][2]=R[5];
                goicp.optR.val[2][0]=R[6]; goicp.optR.val[2][1]=R[7]; goicp.optR.val[2][2]=R[8];

                float btx = batch_out[best_k].best_tx + batch_out[best_k].best_tw * 0.5f;
                float bty = batch_out[best_k].best_ty + batch_out[best_k].best_tw * 0.5f;
                float btz = batch_out[best_k].best_tz + batch_out[best_k].best_tw * 0.5f;
                goicp.optT.val[0][0] = btx;
                goicp.optT.val[1][0] = bty;
                goicp.optT.val[2][0] = btz;

                printf("Error*: %f\n", goicp.optError);

                // ---- Step 2: ICP — deferred until improvement is significant ----
                // We skip ICP when the improvement over the last ICP run is tiny
                // (< ICP_IMPROVEMENT_THRESH).  This prevents the GPU from stalling
                // on repeated ICP calls during the fine tail of BnB.
                if (goicp.optError < last_icp_error * (1.f - ICP_IMPROVEMENT_THRESH)) {
                    clock_t t0 = clock();
                    Matrix R_icp = goicp.optR;
                    Matrix t_icp = goicp.optT;
                    float error = goicp.ICP(R_icp, t_icp);
                    last_icp_error = goicp.optError; // record error at ICP time
                    if (error < goicp.optError) {
                        goicp.optError = error;
                        goicp.optR     = R_icp;
                        goicp.optT     = t_icp;
                        last_icp_error = error;
                        printf("Error*: %f (ICP %.2fs)\n", goicp.optError,
                               (double)(clock()-t0)/CLOCKS_PER_SEC);
                    }
                }

                // ---- Step 3: prune rotation queue with updated optError ----
                {
                    priority_queue<ROTNODE> queueRotNew;
                    while (!queueRot.empty()) {
                        ROTNODE node = queueRot.top();
                        queueRot.pop();
                        if (node.lb < goicp.optError)
                            queueRotNew.push(node);
                        else
                            break;
                    }
                    queueRot = queueRotNew;
                }
            }
        }

        // ---- Step 4: push surviving children into queue ----
        for (int k = 0; k < batch_size; k++) {
            if (batch_out[k].lb >= goicp.optError)
                continue;
            ROTNODE& n = batch_nodes[k];
            n.lb = batch_out[k].lb;
            n.ub = batch_out[k].ub;
            queueRot.push(n);
        }
    }

    // Run a final ICP pass if the last improvement was below the threshold
    // (ensures we always finish with the best locally-refined result).
    if (goicp.optError < last_icp_error) {
        Matrix R_icp = goicp.optR;
        Matrix t_icp = goicp.optT;
        float error = goicp.ICP(R_icp, t_icp);
        if (error < goicp.optError) {
            goicp.optError = error;
            goicp.optR     = R_icp;
            goicp.optT     = t_icp;
            printf("Error*: %f (final ICP)\n", goicp.optError);
        }
    }

    return goicp.optError;
}

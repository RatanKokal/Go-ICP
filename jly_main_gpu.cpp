/********************************************************************
GPU-accelerated main function for Go-ICP
Replaces jly_main.cpp for the GPU build target.

Key differences from CPU jly_main.cpp:
  - Calls GoICP::Initialize() and BuildDT() to set up internal state
  - Extracts DT distance array (float) from DT3D::A.data[z][y][x].distance
  - Uploads everything to GPU via GpuInit()
  - Calls OuterBnB_GPU() instead of GoICP::Register()
  - Calls GoICP::Clear() for cleanup

DT3D field layout (from jly_3ddt.h):
  DT3D::SIZE          — grid side length
  DT3D::scale         — double; cast to float for GpuInit
  DT3D::xMin/yMin/zMin — double; cast to float for GpuInit
  DT3D::A.data[z][y][x].distance — float distance value
  Array3d indices: data[iz][iy][ix] in z-major order
********************************************************************/

#include <time.h>
#include <iostream>
#include <fstream>
using namespace std;

#include "jly_goicp.h"
#include "goicp_gpu.cuh"
#include "ConfigMap.hpp"

// Forward declaration — defined in jly_goicp_gpu.cpp
float OuterBnB_GPU(GoICP& goicp, GoICPGpu& gpu_ctx);

#define DEFAULT_OUTPUT_FNAME "output_gpu.txt"
#define DEFAULT_CONFIG_FNAME "config.txt"
#define DEFAULT_MODEL_FNAME  "model.txt"
#define DEFAULT_DATA_FNAME   "data.txt"

void parseInput(int argc, char **argv,
                string& modelFName, string& dataFName,
                int& NdDownsampled,
                string& configFName, string& outputFName);
void readConfig(string FName, GoICP& goicp);
int  loadPointCloud(string FName, int& N, POINT3D** p);

int main(int argc, char** argv)
{
    int Nm, Nd, NdDownsampled;
    clock_t clockBegin, clockEnd;
    string modelFName, dataFName, configFName, outputFname;
    POINT3D* pModel;
    POINT3D* pData;
    GoICP goicp;

    parseInput(argc, argv, modelFName, dataFName, NdDownsampled, configFName, outputFname);
    readConfig(configFName, goicp);

    // Load point clouds
    loadPointCloud(modelFName, Nm, &pModel);
    loadPointCloud(dataFName,  Nd, &pData);

    goicp.pModel = pModel;
    goicp.Nm     = Nm;
    goicp.pData  = pData;
    goicp.Nd     = Nd;

    // Build Distance Transform
    // For the GPU build, we skip the CPU BuildDT here and use GpuBuildDT below
    // (which runs on the GPU in ~30ms vs ~2.5s CPU time).
    // The DT3D struct fields (SIZE, expandFactor) are still read from config.
    // For the CPU-mode binary (GoICP via jly_main.cpp), BuildDT is called normally.
    int sz = goicp.dt.SIZE;

    // Downsample if requested
    if (NdDownsampled > 0)
        goicp.Nd = NdDownsampled;

    cout << "Model: " << modelFName << " (" << goicp.Nm << "), "
         << "Data: "  << dataFName  << " (" << goicp.Nd << ")" << endl;

    // Initialise GoICP internal state (allocates maxRotDis, minDis, etc.)
    // We call the private Initialize() via the public wrapper added to jly_goicp.h.
    goicp.Initialize_public();

    // ---- C3 fix: GPU path does not implement partial-sort trimming ----
    // The GPU eval kernel sums distances over the first inlierNum points by
    // array index, not the smallest inlierNum values after intro_select().
    // Force doTrim=false so the GPU UB/LB computation is correct.
    // trimFraction is honoured by the CPU ICP path (icp3d.do_trim) but the
    // outer BnB UB/LB will use all Nd points.
    if (goicp.doTrim) {
        fprintf(stderr,
            "[GoICP GPU] WARNING: doTrim=true (trimFraction=%.4f) is not "
            "supported by the GPU BnB path — forcing doTrim=false.\n"
            "  inlierNum reset: %d -> %d, SSEThresh reset: %.6g -> %.6g\n"
            "  ICP sub-calls still use trimFraction via icp3d.do_trim.\n",
            goicp.trimFraction,
            goicp.inlierNum, goicp.Nd,
            (double)goicp.SSEThresh, (double)(goicp.MSEThresh * goicp.Nd));
        goicp.doTrim    = false;
        goicp.inlierNum = goicp.Nd;
        goicp.SSEThresh = goicp.MSEThresh * goicp.Nd;
        // Fix 5: Initialize_public() set icp3d.do_trim from the old doTrim value.
        // Reset it now so intermediate ICP runs on all Nd points (consistent with BnB).
        goicp.SetIcpTrim(false);
    }

    // ---- Upload to GPU and build DT on GPU ----
    GoICPGpu gpu_ctx;
    {
        // Build flat pData array [Nd*3]
        float* pData_flat = new float[goicp.Nd * 3];
        for (int i = 0; i < goicp.Nd; i++) {
            pData_flat[i*3+0] = goicp.pData[i].x;
            pData_flat[i*3+1] = goicp.pData[i].y;
            pData_flat[i*3+2] = goicp.pData[i].z;
        }

        // Fix 3: Pass dt_dist=nullptr — GpuBuildDT will fill d_distArray.
        // dt_xMin/yMin/zMin/scale are all 0 here; GpuBuildDT overwrites c_dt.
        GpuInit(&gpu_ctx, goicp.Nd, pData_flat,
                goicp.GetMaxRotDis(),
                nullptr, sz,
                0.f, 0.f, 0.f, 0.f);

        delete[] pData_flat;
    }

    // Fix 3: Build DT on GPU (~30ms vs ~2.5s CPU).  Must run after GpuInit.
    {
        float* pModel_flat = new float[goicp.Nm * 3];
        for (int i = 0; i < goicp.Nm; i++) {
            pModel_flat[i*3+0] = goicp.pModel[i].x;
            pModel_flat[i*3+1] = goicp.pModel[i].y;
            pModel_flat[i*3+2] = goicp.pModel[i].z;
        }
        cout << "Building Distance Transform (GPU)..." << flush;
        clockBegin = clock();
        GpuBuildDT(&gpu_ctx, pModel_flat, goicp.Nm, sz, goicp.dt.expandFactor);
        clockEnd = clock();
        cout << (double)(clockEnd - clockBegin)/CLOCKS_PER_SEC << "s" << endl;
        delete[] pModel_flat;
    }

    // ---- Run GPU-accelerated registration ----
    cout << "Registering (GPU)..." << endl;
    clockBegin = clock();
    OuterBnB_GPU(goicp, gpu_ctx);
    clockEnd = clock();
    double elapsed = (double)(clockEnd - clockBegin) / CLOCKS_PER_SEC;

    GpuFree(&gpu_ctx);

    // Clean up GoICP internal allocations
    goicp.Clear_public();

    // ---- Output ----
    cout << "Optimal Rotation Matrix:" << endl;
    cout << goicp.optR << endl;
    cout << "Optimal Translation Vector:" << endl;
    cout << goicp.optT << endl;
    cout << "Finished in " << elapsed << "s" << endl;

    ofstream ofile(outputFname.c_str(), ofstream::out);
    ofile << elapsed << endl;
    ofile << goicp.optR << endl;
    ofile << goicp.optT << endl;
    ofile.close();

    free(pModel);
    free(pData);

    return 0;
}

void parseInput(int argc, char **argv,
                string& modelFName, string& dataFName,
                int& NdDownsampled,
                string& configFName, string& outputFName)
{
    modelFName   = DEFAULT_MODEL_FNAME;
    dataFName    = DEFAULT_DATA_FNAME;
    configFName  = DEFAULT_CONFIG_FNAME;
    outputFName  = DEFAULT_OUTPUT_FNAME;
    NdDownsampled = 0;

    if (argc > 5) outputFName  = argv[5];
    if (argc > 4) configFName  = argv[4];
    if (argc > 3) NdDownsampled = atoi(argv[3]);
    if (argc > 2) dataFName    = argv[2];
    if (argc > 1) modelFName   = argv[1];

    cout << "INPUT:" << endl;
    cout << "  model:  " << modelFName   << endl;
    cout << "  data:   " << dataFName    << endl;
    cout << "  Nd_ds:  " << NdDownsampled << endl;
    cout << "  config: " << configFName  << endl;
    cout << "  output: " << outputFName  << endl << endl;
}

void readConfig(string FName, GoICP& goicp)
{
    ConfigMap config(FName.c_str());

    goicp.MSEThresh             = config.getF("MSEThresh");
    goicp.initNodeRot.a         = config.getF("rotMinX");
    goicp.initNodeRot.b         = config.getF("rotMinY");
    goicp.initNodeRot.c         = config.getF("rotMinZ");
    goicp.initNodeRot.w         = config.getF("rotWidth");
    goicp.initNodeTrans.x       = config.getF("transMinX");
    goicp.initNodeTrans.y       = config.getF("transMinY");
    goicp.initNodeTrans.z       = config.getF("transMinZ");
    goicp.initNodeTrans.w       = config.getF("transWidth");
    goicp.trimFraction          = config.getF("trimFraction");
    if (goicp.trimFraction < 0.001f)
        goicp.doTrim = false;
    goicp.dt.SIZE               = config.getI("distTransSize");
    goicp.dt.expandFactor       = config.getF("distTransExpandFactor");

    cout << "CONFIG:" << endl;
    config.print();
    cout << endl;
}

int loadPointCloud(string FName, int& N, POINT3D** p)
{
    ifstream ifile(FName.c_str(), ifstream::in);
    if (!ifile.is_open()) {
        cout << "Unable to open point file '" << FName << "'" << endl;
        exit(-1);
    }
    ifile >> N;
    *p = (POINT3D*)malloc(sizeof(POINT3D) * N);
    for (int i = 0; i < N; i++)
        ifile >> (*p)[i].x >> (*p)[i].y >> (*p)[i].z;
    ifile.close();
    return 0;
}

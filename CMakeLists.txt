cmake_minimum_required(VERSION 3.18)

# Probe for CUDA first without enabling it in the project() call.
# This lets cmake configure on machines without nvcc (CUDA stays disabled),
# while enabling full CUDA compilation when nvcc is present.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(GoICP LANGUAGES CXX CUDA)
    set(HAS_CUDA TRUE)
else()
    project(GoICP LANGUAGES CXX)
    set(HAS_CUDA FALSE)
    message(WARNING "CUDA compiler not found — GoICP_gpu target skipped")
endif()

# ---- CPU target (unchanged) ----
add_executable(GoICP
    jly_main.cpp
    jly_goicp.cpp
    jly_3ddt.cpp
    matrix.cpp
    ConfigMap.cpp
    StringTokenizer.cpp
)
target_compile_features(GoICP PRIVATE cxx_std_11)

# ---- GPU target (only when CUDA is available) ----
if(HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(GoICP_gpu
        jly_main_gpu.cpp
        jly_goicp_gpu.cpp
        jly_goicp.cpp
        jly_3ddt.cpp
        matrix.cpp
        ConfigMap.cpp
        StringTokenizer.cpp
        goicp_gpu.cu          # CUDA source — compiled by nvcc
    )
    target_compile_features(GoICP_gpu PRIVATE cxx_std_11)
    set_target_properties(GoICP_gpu PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;89"  # sm_75 (T4), sm_80 (A100), sm_86/89 (RTX30/40)
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(GoICP_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            -Xcompiler -O3
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    target_link_libraries(GoICP_gpu PRIVATE CUDA::cudart)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — GoICP_gpu target enabled")
endif()
#include "megbrain/version.h"
#include "megbrain/common.h"

using namespace mgb;

Version mgb::get_version() {
#ifdef MGB_MAJOR
    return {MGB_MAJOR, MGB_MINOR, MGB_PATCH, MGB_IS_DEV};
#else
    return {MGE_MAJOR, MGE_MINOR, MGE_PATCH, MGB_IS_DEV};
#endif
}

#if __has_include("NvInfer.h") && MGB_ENABLE_TENSOR_RT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "NvInfer.h"
int mgb::get_tensorrt_version() {
    return NV_TENSORRT_VERSION;
}
#pragma GCC diagnostic pop
#else
int mgb::get_tensorrt_version() {
    return -1;
}
#endif

#if __has_include("cuda.h") && MGB_CUDA
#include "cuda.h"
int mgb::get_cuda_version() {
    return CUDA_VERSION;
}
#else
int mgb::get_cuda_version() {
    return -1;
}
#endif

#if __has_include("cudnn.h") && MGB_CUDA
#include "cudnn.h"
int mgb::get_cudnn_version() {
    return CUDNN_VERSION;
}
#else
int mgb::get_cudnn_version() {
    return -1;
}
#endif

#if __has_include("cuda.h") && MGB_CUDA
#include "cuda.h"
int mgb::get_cuda_driver_version() {
    int driver_version = -1;
    auto error_code = cudaDriverGetVersion(&driver_version);
    if (error_code != cudaSuccess) {
        mgb_log_warn("cudaDriverGetVersion failed, error code: %d", error_code);
        return -1;
    }
    return driver_version;
}
#else
int mgb::get_cuda_driver_version() {
    return -1;
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

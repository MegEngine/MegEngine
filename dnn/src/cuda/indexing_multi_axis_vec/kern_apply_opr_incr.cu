#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

#if !MEGDNN_DISABLE_FLOAT16
__device__ void atomicAdd(megdnn::dt_float16* address, megdnn::dt_float16 val) {
    ::megdnn::cuda::atomic_add(address, val);
}

__device__ void atomicAdd(megdnn::dt_bfloat16*, megdnn::dt_bfloat16) {
    __trap();
    ((int*)0)[0] = 1;
}
#endif

__device__ void atomicAdd(megdnn::dt_int8*, megdnn::dt_int8) {
    __trap();
    ((int*)0)[0] = 1;
}

__device__ void atomicAdd(megdnn::dt_uint8*, megdnn::dt_uint8) {
    __trap();
    ((int*)0)[0] = 1;
}

__device__ void atomicAdd(megdnn::dt_int16*, megdnn::dt_int16) {
    __trap();
    ((int*)0)[0] = 1;
}

__device__ void atomicAdd(megdnn::dt_bool*, megdnn::dt_bool) {
    __trap();
    ((int*)0)[0] = 1;
}

#define KERN_APPLY_OPR_OPR ::megdnn::cuda::indexing_multi_axis_vec::OprAtomicIncr
#include "./kern_apply_opr_impl.cuinl"

// vim: ft=cuda syntax=cpp.doxygen

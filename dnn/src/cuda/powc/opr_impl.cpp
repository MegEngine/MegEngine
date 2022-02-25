#include "./opr_impl.h"
#include "./kern.cuh"

#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

void PowCImpl::do_exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
        const int* exp_i) {
    powc_kern(dst, src, exp_f, exp_i, cuda_stream(handle()));
}

// vim: syntax=cpp.doxygen

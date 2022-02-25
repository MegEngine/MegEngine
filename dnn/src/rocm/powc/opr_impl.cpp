#include "hcc_detail/hcc_defs_prologue.h"

#include "./opr_impl.h"
#include "src/rocm/powc/powc.h.hip"

#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

void PowCImpl::do_exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
        const int* exp_i) {
    powc_kern(dst, src, exp_f, exp_i, hip_stream(handle()));
}

// vim: syntax=cpp.doxygen

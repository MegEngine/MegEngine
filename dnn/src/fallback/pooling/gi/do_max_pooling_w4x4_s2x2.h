#pragma once
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace fallback {

void do_max_pooling_w4x4_s2x2_float_gi(
        const dt_float32* src, dt_float32* dst, DType src_dtype, const int IH,
        const int IW, const int OH, const int OW, const int PH, const int PW);
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen

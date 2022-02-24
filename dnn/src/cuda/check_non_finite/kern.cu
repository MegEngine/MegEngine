#include "src/common/reduce_helper_device.h"

#include "megdnn/dtype.h"
#include "src/cuda/reduce_helper.cuh"

namespace megdnn {
namespace cuda {

#define COMMA ,

#define cb(_dtype)                                                          \
    INST_REDUCE(                                                            \
            device_reduce::CheckNonFiniteOp<                                \
                    _dtype COMMA dt_float32 COMMA dt_int32 COMMA dt_int32>, \
            false);

cb(dt_float32);
cb(dt_float16);
#undef cb
#undef COMMA
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

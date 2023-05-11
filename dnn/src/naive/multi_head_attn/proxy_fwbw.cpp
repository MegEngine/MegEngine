#include "src/naive/multi_head_attn/proxy_fwbw.h"
#include "megdnn/oprs/linalg.h"
#include "src/common/utils.cuh"
#include "src/naive/handle.h"
#include "src/naive/multi_head_attn/opr_impl.h"

namespace megdnn {
namespace naive {

#define cb(DType)                                          \
    void MHAForwardProxyOpr::move_scaler_to_device(        \
            Handle* handle, DTypeTrait<DType>::ctype* dst, \
            DTypeTrait<DType>::ctype* src) {               \
        MEGDNN_MARK_USED_VAR(handle);                      \
        *dst = *src;                                       \
    };
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

#define cb(DType)                                          \
    void MHABackwardProxyOpr::move_scaler_to_device(       \
            Handle* handle, DTypeTrait<DType>::ctype* dst, \
            DTypeTrait<DType>::ctype* src) {               \
        MEGDNN_MARK_USED_VAR(handle);                      \
        *dst = *src;                                       \
    };
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

}  // namespace naive
}  // namespace megdnn
   // vim: syntax=cpp.doxygen

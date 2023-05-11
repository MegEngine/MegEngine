#include "src/cuda/multi_head_attn/proxy_bw.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace cuda {

#define cb(DType)                                                                   \
    void MHABackwardProxyOpr::move_scaler_to_device(                                \
            Handle* handle, DTypeTrait<DType>::ctype* dst,                          \
            DTypeTrait<DType>::ctype* src) {                                        \
        cudaMemcpyAsync(                                                            \
                dst, src, sizeof(DTypeTrait<DType>::ctype), cudaMemcpyHostToDevice, \
                cuda_stream(handle));                                               \
    };
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen

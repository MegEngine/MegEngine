#include "./kern.cuh"

namespace megdnn {
namespace cuda {
#define cb(_dtype)                                             \
    INST_RUN_ELEMWISE(                                         \
            MaskedFillScalarKernOp<DTypeTrait<_dtype>::ctype>, \
            DTypeTrait<_dtype>::ctype, 1);
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
cb(::megdnn::dtype::Bool)

#undef cb
}  // namespace cuda
}  // namespace megdnn

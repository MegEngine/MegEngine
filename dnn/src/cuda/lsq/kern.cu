#include "./kern.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dtype)                                                                    \
    INST_RUN_ELEMWISE(                                                                \
            LSQKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 3);      \
    INST_RUN_ELEMWISE(                                                                \
            LSQBwdKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 3);   \
    INST_RUN_ELEMWISE(                                                                \
            LSQKernOpNonContig<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, \
            5);                                                                       \
    INST_RUN_ELEMWISE(                                                                \
            LSQBwdKernOpNonContig<DTypeTrait<_dtype>::ctype>,                         \
            DTypeTrait<_dtype>::ctype, 7);
cb(megdnn::dtype::Float32)

}  // namespace cuda
}  // namespace megdnn

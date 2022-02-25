#include "./kern.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dtype)                                                                    \
    INST_RUN_ELEMWISE(                                                                \
            TQTKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 1);      \
    INST_RUN_ELEMWISE(                                                                \
            TQTBwdKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 1);   \
    INST_RUN_ELEMWISE(                                                                \
            TQTKernOpNonContig<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, \
            3);                                                                       \
    INST_RUN_ELEMWISE(                                                                \
            TQTBwdKernOpNonContig<DTypeTrait<_dtype>::ctype>,                         \
            DTypeTrait<_dtype>::ctype, 5);
cb(megdnn::dtype::Float32)

}  // namespace cuda
}  // namespace megdnn

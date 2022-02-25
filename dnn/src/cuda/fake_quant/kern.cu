#include "./kern.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dtype)                                                                     \
    INST_RUN_ELEMWISE(                                                                 \
            FakeQuantKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 2); \
    INST_RUN_ELEMWISE(                                                                 \
            FakeQuantBwdKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype,  \
            2);                                                                        \
    INST_RUN_ELEMWISE(                                                                 \
            FakeQuantKernOpNonContig<DTypeTrait<_dtype>::ctype>,                       \
            DTypeTrait<_dtype>::ctype, 4);                                             \
    INST_RUN_ELEMWISE(                                                                 \
            FakeQuantBwdKernOpNonContig<DTypeTrait<_dtype>::ctype>,                    \
            DTypeTrait<_dtype>::ctype, 5);
cb(megdnn::dtype::Float32)

}  // namespace cuda
}  // namespace megdnn
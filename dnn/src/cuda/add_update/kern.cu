#include "./kern.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dtype)                                                                     \
    INST_RUN_ELEMWISE(                                                                 \
            AddUpdateKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 1); \
    INST_RUN_ELEMWISE(                                                                 \
            AddUpdateKernOpNonContig<DTypeTrait<_dtype>::ctype>,                       \
            DTypeTrait<_dtype>::ctype, 2);

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

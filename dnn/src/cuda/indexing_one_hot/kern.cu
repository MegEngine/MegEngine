#include "./kern.cuh"
#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dt)                                                              \
    typedef indexing_one_hot::OpGet<DTypeTrait<dtype::_dt>::ctype, dt_int32> \
            OpGet##_dt;                                                      \
    typedef indexing_one_hot::OpSet<DTypeTrait<dtype::_dt>::ctype, dt_int32> \
            OpSet##_dt;                                                      \
    INST_RUN_ELEMWISE(OpGet##_dt, void, 0);                                  \
    INST_RUN_ELEMWISE(OpSet##_dt, void, 0);

MEGDNN_FOREACH_DTYPE_NAME(cb)
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)

#undef cb

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

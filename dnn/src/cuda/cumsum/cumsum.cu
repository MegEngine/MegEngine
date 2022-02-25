#include "./kern_impl.cuinl"

namespace megdnn {
namespace cuda {
namespace cumsum {

#define INST_(T, Op, exclusive, reverse)                                  \
    template void run_kern<T, Op, exclusive, reverse>(                    \
            T*, void*, uint32_t, uint32_t, uint32_t, uint32_t, const Op&, \
            cudaStream_t)
#define INST(T)                      \
    INST_(T, SumOp<T>, true, true);  \
    INST_(T, SumOp<T>, false, true); \
    INST_(T, SumOp<T>, true, false); \
    INST_(T, SumOp<T>, false, false);

#define cb(DType) INST(typename DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace cumsum
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen

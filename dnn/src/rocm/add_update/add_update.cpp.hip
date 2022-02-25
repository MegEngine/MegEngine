#include "hcc_detail/hcc_defs_prologue.h"
#include "./add_update.h.hip"

namespace megdnn {
namespace rocm {

#define cb(_dtype)                                                         \
    INST_RUN_ELEMWISE(AddUpdateKernOp<DTypeTrait<_dtype>::ctype>,          \
                      DTypeTrait<_dtype>::ctype, 1);                       \
    INST_RUN_ELEMWISE(AddUpdateKernOpNonContig<DTypeTrait<_dtype>::ctype>, \
                      DTypeTrait<_dtype>::ctype, 2);

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace rocm
}  // namespace megdnn


// vim: ft=cpp syntax=cpp.doxygen


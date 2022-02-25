#include "src/common/argmxx_helper.h"

#include "megdnn/dtype.h"
#include "src/cuda/reduce_helper.cuh"

namespace megdnn {
namespace cuda {

#define INST(_dt)                                                                    \
    INST_REDUCE(argmxx::ArgmxxOp<DTypeTrait<_dt>::ctype MEGDNN_COMMA false>, false); \
    INST_REDUCE(argmxx::ArgmxxOp<DTypeTrait<_dt>::ctype MEGDNN_COMMA true>, false);

MEGDNN_FOREACH_COMPUTING_DTYPE(INST)

}  // namespace cuda
}  // namespace megdnn



#include "helper.h"
#include "megdnn/dtype.h"
#include "src/cuda/reduce_helper.cuh"

namespace megdnn {
namespace cuda {

using namespace device_reduce;
#define COMMA ,

INST_REDUCE(NormOp<dt_float32 COMMA dt_float32 COMMA dt_float32>, false);
INST_REDUCE(NormOp<dt_float16 COMMA dt_float16 COMMA dt_float16>, false);

INST_REDUCE(NormZeroOp<dt_float32 COMMA dt_float32 COMMA dt_float32>, false);
INST_REDUCE(NormZeroOp<dt_float16 COMMA dt_float16 COMMA dt_float16>, false);

INST_REDUCE(NormOneOp<dt_float32 COMMA dt_float32 COMMA dt_float32>, false);
INST_REDUCE(NormOneOp<dt_float16 COMMA dt_float16 COMMA dt_float16>, false);

INST_REDUCE(NormTwoOp<dt_float32 COMMA dt_float32 COMMA dt_float32>, false);
INST_REDUCE(NormTwoOp<dt_float16 COMMA dt_float16 COMMA dt_float16>, false);

#undef COMMA

}  // namespace cuda
}  // namespace megdnn
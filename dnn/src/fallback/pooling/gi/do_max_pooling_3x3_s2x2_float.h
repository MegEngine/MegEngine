#include "src/common/utils.h"

#include "megdnn/arch.h"

#include "src/fallback/general_intrinsic/gi_float.h"

namespace megdnn {
namespace fallback {

void do_max_pooling_3x3_s2x2_float_gi(
        const float* src, float* dst, size_t IH_, size_t IW_, size_t OH_, size_t OW_,
        size_t PH_, size_t PW_, const WorkspaceBundle& ws);

}  // namespace fallback
}  // namespace megdnn

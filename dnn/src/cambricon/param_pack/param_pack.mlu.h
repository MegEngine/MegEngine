#pragma once

#include "src/cambricon/utils.mlu.h"

namespace megdnn {
namespace cambricon {
namespace param_pack {

void concat_proxy(
        const BangHandle& handle, const void** srcs, void* dst, const int32_t* offsets,
        int64_t dtype_size, int64_t num_src);

}  // namespace param_pack
}  // namespace cambricon
}  // namespace megdnn
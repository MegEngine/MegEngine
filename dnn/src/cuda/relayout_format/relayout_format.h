#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/oprs.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {

struct RelayoutFormatFast {
    static bool usable(
            const TensorLayout& src_layout, const TensorLayout& dst_layout,
            const RelayoutFormat::Param::Mode& mode =
                    RelayoutFormat::Param::Mode::NCHW_NCHW4);
    static void exec(
            const TensorND& src, const TensorND& dst, cudaStream_t stream,
            RelayoutFormat::Param::Mode mode, int group);
};

}  // namespace relayout_format

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

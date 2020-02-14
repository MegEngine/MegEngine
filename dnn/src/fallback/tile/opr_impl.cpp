/**
 * \file dnn/src/fallback/tile/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/tile/opr_impl.h"

#include <numeric>
#include "src/naive/handle.h"
#include <cstring>
#include "src/common/tile_repeat_helper.h"
#include "src/common/utils.h"

namespace megdnn {
namespace fallback {

size_t TileImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &dst)
{
    auto workspace_size = get_workspace_in_bytes_fwd(src, dst);
    return workspace_size;
}

void TileImpl::exec(_megdnn_tensor_in src_,
        _megdnn_tensor_out dst_,
        _megdnn_workspace workspace)
{
    check_exec(src_.layout, dst_.layout, workspace.size);
    TensorShape src, dst, times;
    simplify_shape(src_.layout, dst_.layout, param().times,
            src, dst, times);
    auto nr_reduces = count_not_ones_in_shape(times);
    if (nr_reduces == 0) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(std::memcpy(dst_.raw_ptr, src_.raw_ptr,
                    sizeof(float) * dst.total_nr_elems()));
        return;
    }

    auto kern = [=]() {
        auto ndim = times.ndim;
        WorkspaceBundle workspaces(workspace.raw_ptr,
                {dst.total_nr_elems() * sizeof(float),
                dst.total_nr_elems() * sizeof(float)});
        auto workspace0 = static_cast<float *>(workspaces.get(0));
        auto workspace1 = static_cast<float *>(workspaces.get(1));


        float *current, *next;
        size_t state;

        init_tile_repeat_state(src_.ptr<dt_float32>(), dst_.ptr<dt_float32>(),
                workspace0, workspace1,
                current, next, state,
                nr_reduces);

        for (size_t i = ndim; i > 0; --i) {
            size_t j = i-1;
            if (times.shape[j] != 1) {
                // m = sshape[0]*...*sshape[i-2]
                auto m = std::accumulate(src.shape, src.shape+j, 1_z,
                        SafeMultiplies<size_t>());
                // n = sshape[i-1]*dshape[i]*...
                auto n = std::accumulate(dst.shape+i, dst.shape+ndim, 1_z,
                        SafeMultiplies<size_t>()) * src.shape[j];
                // forward is repeat (m, n) to (m*times, n)
                tile_or_repeat_single_axis(current, next,
                        m, n, times[j]);
                update_tile_repeat_state(src_.ptr<dt_float32>(),
                        dst_.ptr<dt_float32>(),
                        workspace0, workspace1,
                        current, next, state,
                        nr_reduces);
            }
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen


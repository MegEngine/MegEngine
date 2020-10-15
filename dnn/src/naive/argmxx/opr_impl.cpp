/**
 * \file dnn/src/naive/argmxx/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/argmxx/opr_impl.h"

#include "src/common/utils.h"
#include "src/common/reduce_helper.h"
#include "src/naive/handle.h"

#include <numeric>

namespace megdnn {

using namespace megdnn;

template <bool is_max> struct traits;

template <> struct traits<true> {
    static const float init;
    static bool better_than(float lhs, float rhs)
    { return lhs > rhs; }
};
const float traits<true>::init = std::numeric_limits<float>::lowest();

template <> struct traits<false> {
    static const float init;
    static float better_than(float lhs, float rhs)
    { return lhs < rhs; }
};
const float traits<false>::init = std::numeric_limits<float>::max();

template <typename T, bool is_max>
void exec_forward(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        const ArgmxxBase::Param &param)
{
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param.axis);
    for (size_t a = 0; a < A; ++a) for (size_t c = 0; c < C; ++c) {
        float best_val = traits<is_max>::init;
        size_t best_arg = 0;
        for (size_t b = 0; b < B; ++b) {
            float curr_val = float(src.ptr<T>()[(a*B+b)*C+c]);
            if (traits<is_max>::better_than(curr_val, best_val)) {
                best_val = curr_val;
                best_arg = b;
            }
        }
        dst.ptr<dt_int32>()[a*C+c] = best_arg;
    }
}

} // anonymous namespace

namespace megdnn {
namespace naive {

void ArgmaxForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN(static_cast<HandleImpl*>(handle()), \
                exec_forward<ctype MEGDNN_COMMA true>(src, dst, param())); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

void ArgminForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN(static_cast<HandleImpl*>(handle()), \
                exec_forward<ctype MEGDNN_COMMA false>(src, dst, param())); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen

/**
 * \file dnn/src/rocm/indexing_one_hot/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./opr_impl.h"
#include "src/rocm/indexing_one_hot/indexing_one_hot.h.hip"

#include "src/rocm/utils.h"
#include "src/rocm/elemwise_helper.h.hip"

using namespace megdnn;
using namespace rocm;
using namespace indexing_one_hot;

namespace {

    KernParam make_kern_param(const TensorLayout &layout, size_t axis) {
        KernParam ret;
        memset(&ret, 0, sizeof(ret));
        ret.shape_lo = layout.stride[axis];
        ret.stride_hi = axis > 0 ? layout.stride[axis - 1] : 1;
        ret.max_mid_index = layout[axis];
        return ret;
    }

} // anonymous namespace

void IndexingOneHotForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in index,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, index.layout, dst.layout, workspace.size);
    ElemwiseOpParamN<0> ele_param{dst.layout.total_nr_elems()};
    auto kern_param = make_kern_param(src.layout, m_param.axis);
    auto stream = hip_stream(handle());
    kern_param.error_tracker = m_error_tracker;
    kern_param.error_info = async_error_info(handle());

#define cb(_dt) \
    case DTypeTrait<_dt>::enumv: { \
        using ctype = DTypeTrait<_dt>::ctype; \
        using Op = OpGet<DTypeTrait<_dt>::ctype, dt_int32>; \
        Op op{src.ptr<ctype>(), index.ptr<dt_int32>(), dst.ptr<ctype>(), \
            kern_param}; \
        return run_elemwise<Op, void>(ele_param, stream, op); \
    }
    switch (src.layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        default:
            megdnn_throw(megdnn_mangle("bad dtype"));
    }
#undef cb
}

void IndexingSetOneHotForwardImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in index,
        _megdnn_tensor_in sub, _megdnn_workspace workspace) {
    check_exec(data.layout, index.layout, sub.layout, workspace.size);

    ElemwiseOpParamN<0> ele_param{sub.layout.total_nr_elems()};
    auto kern_param = make_kern_param(data.layout, m_param.axis);
    auto stream = hip_stream(handle());
    kern_param.error_tracker = m_error_tracker;
    kern_param.error_info = async_error_info(handle());

#define cb(_dt) \
    case DTypeTrait<_dt>::enumv: { \
        using ctype = DTypeTrait<_dt>::ctype; \
        using Op = OpSet<DTypeTrait<_dt>::ctype, dt_int32>; \
        Op op{data.ptr<ctype>(), index.ptr<dt_int32>(), sub.ptr<ctype>(), \
            kern_param}; \
        return run_elemwise<Op, void>(ele_param, stream, op); \
    }
    switch (data.layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        default:
            megdnn_throw(megdnn_mangle("bad dtype"));
    }
#undef cb
}

// vim: syntax=cpp.doxygen



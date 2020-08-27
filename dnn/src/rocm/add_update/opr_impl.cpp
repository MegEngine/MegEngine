/**
 * \file dnn/src/rocm/add_update/opr_impl.cpp
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
#include "src/rocm/add_update/add_update.h.hip"

#include "src/common/utils.h"

using namespace megdnn;
using namespace rocm;

void AddUpdateForwardImpl::exec(_megdnn_tensor_inout dest,
                                _megdnn_tensor_in delta) {
    check_exec(dest.layout, delta.layout);
    if (!dest.layout.is_contiguous()) {
        return exec_noncontig(dest, delta);
    }
    ElemwiseOpParamN<1> param;
    param[0] = delta;
    param[0].layout = param[0].layout.broadcast(dest.layout);
    param.init_from_given_tensor();
    auto stream = hip_stream(handle());
    switch (dest.layout.dtype.enumv()) {
#define cb(_dt)                                                \
    case DTypeTrait<_dt>::enumv: {                             \
        using ctype = DTypeTrait<_dt>::ctype;                  \
        return run_elemwise<AddUpdateKernOp<ctype>, ctype, 1>( \
                param, stream, {dest, m_param});               \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

        default:
            megdnn_throw(megdnn_mangle("unsupported dtype for AddUpdate"));
    }
}

void AddUpdateForwardImpl::exec_noncontig(_megdnn_tensor_inout dest,
                                          _megdnn_tensor_in delta) {
    ElemwiseOpParamN<2> param = make_param(dest, delta);
    auto stream = hip_stream(handle());
    switch (dest.layout.dtype.enumv()) {
#define cb(_dt)                                                         \
    case DTypeTrait<_dt>::enumv: {                                      \
        using ctype = DTypeTrait<_dt>::ctype;                           \
        return run_elemwise<AddUpdateKernOpNonContig<ctype>, ctype, 2>( \
                param, stream, {m_param});                              \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

        default:
            megdnn_throw(megdnn_mangle("unsupported dtype for AddUpdate"));
    }
}

// vim: syntax=cpp.doxygen


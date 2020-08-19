/**
 * \file dnn/src/cuda/cond_take/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./kern.cuh"
#include "src/common/utils.h"
#include "src/common/cond_take/predicate.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace cuda::cond_take;
using namespace megdnn::cond_take;

using Param = CondTake::Param;

WorkspaceBundle CondTakeImpl::make_bundle(size_t nr_item) {
    cuda_check(cudaSetDevice(concrete_handle(handle())->device_id()));
    auto gen_idx_wk_size = gen_idx_get_workspace_size(nr_item);
    return {nullptr,
            {(nr_item + 1) * sizeof(IdxType), gen_idx_wk_size},
            handle()->alignment_requirement()};
}

size_t CondTakeImpl::get_workspace_in_bytes(const TensorLayout& data) {
    return make_bundle(data.total_nr_elems()).total_size_in_bytes();
}

CondTakeImpl::Output CondTakeImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in mask,
        _megdnn_workspace workspace,
        DynOutMallocPolicyCall malloc_policy) {
    size_t size = check_exec_get_size(data.layout, mask.layout, workspace.size);
    auto wk_bundle = make_bundle(size);
    wk_bundle.set(workspace.raw_ptr);

    auto idx_tmp = static_cast<IdxType*>(wk_bundle.get(0));

    KParam kparam(param());
    auto stream = cuda_stream(handle());
    size_t out_size;
    switch (mask.layout.dtype.enumv()) {
#define cb(_dt) \
        case DTypeTrait<_dt>::enumv: { \
            using ctype = DTypeTrait<_dt>::ctype; \
            out_size = gen_idx(wk_bundle.get(1), wk_bundle.get_size(1), \
                    idx_tmp, mask.ptr<ctype>(), \
                    size, static_cast<uint32_t>(param().mode), kparam, \
                    stream); \
            break; \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
        default:
            megdnn_throw("bad mask dtype");
    }

    auto out_data = malloc_policy.alloc_output(0,
            data.layout.dtype, {out_size});
    auto out_idx = malloc_policy.alloc_output(1, dtype::Int32(), {out_size});
    auto out_idx_ptr = out_idx.ptr<dt_int32>();

    switch (data.layout.dtype.enumv()) {
#define cb(_dt) \
        case DTypeTrait<_dt>::enumv: { \
            using ctype = DTypeTrait<_dt>::ctype; \
            auto out_data_ptr = out_data.ptr<ctype>(); \
            auto data_ptr = data.ptr<ctype>(); \
            copy_output<ctype>( \
                    out_data_ptr, out_idx_ptr, data_ptr, idx_tmp, size, \
                    stream); \
            break; \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
        default:
            megdnn_throw("bad data dtype");
    }

    return {{out_data, out_idx}};
}

// vim: syntax=cpp.doxygen

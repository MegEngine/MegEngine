/**
 * \file dnn/src/naive/cond_take/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/common/utils.h"
#include "src/common/cond_take/predicate.cuh"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;
using namespace cond_take;

using Param = CondTake::Param;

namespace {

    template<uint32_t mode, typename ctype>
    void gen_index(
            size_t sz, dt_int32 *dest, const ctype *inp,
            cond_take::Pred<mode, ctype> pred) {
        int didx = 0;
        for (size_t i = 0; i < sz; ++ i) {
            if (pred(inp[i])) {
                dest[didx ++] = i;
            }
        }
        dest[sz] = didx;
    }

    template<typename ctype>
    void copy_data(size_t sz, dt_int32 *dest_idx, ctype *dest_data,
            const dt_int32 *src_idx, const ctype *src_data) {
        for (size_t i = 0; i < sz; ++ i) {
            auto idx = src_idx[i];
            dest_idx[i] = idx;
            dest_data[i] = src_data[idx];
        }
    }

} // anonymous namespace

size_t CondTakeImpl::get_workspace_in_bytes(const TensorLayout& data) {
    return (data.total_nr_elems() + 1) * sizeof(dt_int32);
}

CondTakeImpl::Output CondTakeImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in mask,
        _megdnn_workspace workspace,
        DynOutMallocPolicyCall malloc_policy) {
    auto size = check_exec_get_size(data.layout, mask.layout, workspace.size);
    auto idx_tmp = workspace.ptr<dt_int32>();

    switch (mask.layout.dtype.enumv()) {
#define cb(_dt) \
        case DTypeTrait<_dt>::enumv: { \
            using ctype = DTypeTrait<_dt>::ctype; \
            dispatch_genidx<ctype>(size, idx_tmp, mask.ptr<ctype>()); \
            break; \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
        default:
            megdnn_throw("bad mask dtype");
    }

    static_cast<HandleImpl*>(handle())->megcore_dispatcher()->sync();
    size_t out_size = idx_tmp[size];
    auto out_data = malloc_policy.alloc_output(
            0, data.layout.dtype, {out_size});
    auto out_idx = malloc_policy.alloc_output(1, dtype::Int32(), {out_size});
    auto out_idx_ptr = out_idx.ptr<dt_int32>();

    switch (data.layout.dtype.enumv()) {
#define cb(_dt) \
        case DTypeTrait<_dt>::enumv: { \
            using ctype = DTypeTrait<_dt>::ctype; \
            auto out_data_ptr = out_data.ptr<ctype>(); \
            auto data_ptr = data.ptr<ctype>(); \
            MEGDNN_DISPATCH_CPU_KERN_OPR( \
                copy_data<ctype>( \
                    out_size, out_idx_ptr, out_data_ptr, idx_tmp, data_ptr)); \
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

template<typename ctype>
void CondTakeImpl::dispatch_genidx(
        size_t size, dt_int32 *dest, const ctype *inp) {
    KParam kparam(m_param);
    switch (m_param.mode) {
#define cb(_m) \
        case Param::Mode::_m: { \
            Pred<PEnum::_m, ctype> pred(kparam); \
            MEGDNN_DISPATCH_CPU_KERN_OPR(gen_index( \
                        size, dest, inp, pred)); \
            return; \
        }
        MEGDNN_FOREACH_COND_TAKE_MODE(cb)
#undef cb
    }
    megdnn_assert_internal(0);
}

// vim: syntax=cpp.doxygen


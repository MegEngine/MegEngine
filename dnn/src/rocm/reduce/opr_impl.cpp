/**
 * \file dnn/src/rocm/reduce/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/reduce/opr_impl.h"
#include "src/rocm/reduce_helper.h.hip"

#include "src/rocm/handle.h"
#include "src/rocm/utils.h"

#include "src/common/reduce_helper.h"

namespace {

using namespace megdnn;
using namespace rocm;

template <template <typename, typename, typename> class Op>
size_t dispatch_dtype_workspace(const TensorLayout& src, const TensorLayout&,
                                size_t A, size_t B, size_t C,
                                Reduce::DataType data_type) {
#if !MEGDNN_DISABLE_FLOAT16
    using f16 = DTypeTrait<dtype::Float16>::ctype;
#endif
    using f32 = DTypeTrait<dtype::Float32>::ctype;
    using i32 = DTypeTrait<dtype::Int32>::ctype;
    if (data_type == Reduce::DataType::DEFAULT) {
#define cb(_dt)                                                             \
    case DTypeTrait<_dt>::enumv: {                                          \
        using ctype = DTypeTrait<_dt>::ctype;                               \
        return get_reduce_workspace_in_bytes<Op<ctype, ctype, ctype>>(A, B, \
                                                                      C);   \
    }
        switch (src.dtype.enumv()) {
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
            default:
                megdnn_assert_internal(false);
        }
#undef cb
    } else if (data_type == Reduce::DataType::FLOAT_O32xC32) {
        if (src.dtype == dtype::Float32())
            return get_reduce_workspace_in_bytes<Op<f32, f32, f32>>(A, B, C);
#if !MEGDNN_DISABLE_FLOAT16
        else if (src.dtype == dtype::Float16())
            return get_reduce_workspace_in_bytes<Op<f16, f32, f32>>(A, B, C);
#endif
        else if (src.dtype == dtype::Int32())
            return get_reduce_workspace_in_bytes<Op<i32, f32, f32>>(A, B, C);
    }
#if !MEGDNN_DISABLE_FLOAT16
    else if (data_type == Reduce::DataType::FLOAT_O16xC32) {
        if (src.dtype == dtype::Float16())
            return get_reduce_workspace_in_bytes<Op<f16, f16, f32>>(A, B, C);
        else if (src.dtype == dtype::Float32())
            return get_reduce_workspace_in_bytes<Op<f32, f16, f32>>(A, B, C);
    }
#endif
    megdnn_assert_internal(0);
}

template <template <typename, typename, typename> class Op>
void dispatch_dtype(hipStream_t stream, const TensorND& src,
                    const TensorND& dst, _megdnn_workspace workspace, size_t A,
                    size_t B, size_t C, Reduce::DataType data_type) {
#if !MEGDNN_DISABLE_FLOAT16
    using f16 = DTypeTrait<dtype::Float16>::ctype;
#endif
    using f32 = DTypeTrait<dtype::Float32>::ctype;
    using i32 = DTypeTrait<dtype::Int32>::ctype;
    if (data_type == Reduce::DataType::DEFAULT) {
        switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                             \
    case DTypeTrait<_dt>::enumv: {                                          \
        using ctype = DTypeTrait<_dt>::ctype;                               \
        return run_reduce<Op<ctype, ctype, ctype>, false>(                  \
                workspace.ptr<ctype>(), A, B, C, stream,                    \
                Op<ctype, ctype, ctype>(src.ptr<ctype>(), dst.ptr<ctype>(), \
                                        B));                                \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
            default:
                megdnn_assert_internal(false);
        }
    } else if (data_type == Reduce::DataType::FLOAT_O32xC32) {
        if (src.layout.dtype == dtype::Float32()) {
            return run_reduce<Op<f32, f32, f32>, false>(
                    workspace.ptr<f32>(), A, B, C, stream,
                    Op<f32, f32, f32>(src.ptr<f32>(), dst.ptr<f32>(), B));
        }
#if !MEGDNN_DISABLE_FLOAT16
        else if (src.layout.dtype == dtype::Float16()) {
            return run_reduce<Op<f16, f32, f32>, false>(
                    workspace.ptr<f32>(), A, B, C, stream,
                    Op<f16, f32, f32>(src.ptr<f16>(), dst.ptr<f32>(), B));
        }
#endif
        else if (src.layout.dtype == dtype::Float32()) {
            return run_reduce<Op<i32, f32, f32>, false>(
                    workspace.ptr<f32>(), A, B, C, stream,
                    Op<i32, f32, f32>(src.ptr<i32>(), dst.ptr<f32>(), B));
        }
    }
#if !MEGDNN_DISABLE_FLOAT16
    else if (data_type == Reduce::DataType::FLOAT_O16xC32) {
        if (src.layout.dtype == dtype::Float16()) {
            return run_reduce<Op<f16, f16, f32>, false>(
                    workspace.ptr<f32>(), A, B, C, stream,
                    Op<f16, f16, f32>(src.ptr<f16>(), dst.ptr<f16>(), B));
        } else {
            return run_reduce<Op<f32, f16, f32>, false>(
                    workspace.ptr<f32>(), A, B, C, stream,
                    Op<f32, f16, f32>(src.ptr<f32>(), dst.ptr<f16>(), B));
        }
    }
#endif
    megdnn_assert_internal(0);
#undef cb
}

}  // anonymous namespace

namespace megdnn {
namespace rocm {

void ReduceForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                             _megdnn_workspace workspace) {
    using namespace reduce;
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    get_ABC(src.layout, A, B, C, param().axis);
    auto stream = hip_stream(this->handle());
#define CASE(_mode, _op)                                                 \
    case _mode:                                                          \
        return dispatch_dtype<_op>(stream, src, dst, workspace, A, B, C, \
                                   param().data_type);
    switch (param().mode) {
        CASE(Mode::SUM, SumOp)
        CASE(Mode::SUM_SQR, SumSqrOp)
        CASE(Mode::PRODUCT, ProdOp)
        CASE(Mode::MIN, MinOp)
        CASE(Mode::MAX, MaxOp)
        CASE(Mode::MEAN, MeanOp)
        default:
            megdnn_assert_internal(false);
#undef CASE
    }
}

size_t ReduceForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                 const TensorLayout& dst) {
    megdnn_assert(param().data_type != Reduce::DataType::FLOAT_IO16xC32,
                  "FLOAT_IO16xC32 is deprecated");
    using namespace reduce;
    size_t A, B, C;
    get_ABC(src, A, B, C, param().axis);
#define CASE(_mode, _op)                                         \
    case _mode: {                                                \
        return dispatch_dtype_workspace<_op>(src, dst, A, B, C,  \
                                             param().data_type); \
        break;                                                   \
    }

    switch (param().mode) {
        CASE(Mode::SUM, SumOp)
        CASE(Mode::SUM_SQR, SumSqrOp)
        CASE(Mode::PRODUCT, ProdOp)
        CASE(Mode::MIN, MinOp)
        CASE(Mode::MAX, MaxOp)
        CASE(Mode::MEAN, MeanOp)
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}
}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen

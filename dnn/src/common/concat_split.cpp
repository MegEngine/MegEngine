/**
 * \file dnn/src/common/concat_split.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

#include <numeric>

namespace megdnn {

ConcatSplitBase::ConcatSplitBase(Handle *handle):
    OperatorBase(handle),
    m_get_layout([](const TensorND &tensor) { return tensor.layout; }),
    m_get_shape([](const TensorLayout &layout) { return TensorShape(layout); })
{
}

void ConcatSplitBase::check_layout_common(const TensorLayoutArray &srcs,
        const TensorLayout &dst)
{
    // ensure same data type
    for (auto &&src: srcs) {
        megdnn_assert(src.dtype == dst.dtype);
    }
    // ensure all layouts are contiguous
	for (auto &&src: srcs) {
        megdnn_assert_contiguous(src);
	}
    megdnn_assert_contiguous(dst);
    // ensure all layouts have the same ndim
    auto ndim = dst.ndim;
	for (auto &&src: srcs) {
        megdnn_assert_eq_size_t(src.ndim, ndim);
	}
	// ensure param().axis is correct
    auto errmsg = megdnn_mangle("param().axis=") +
        std::to_string(param().axis) + megdnn_mangle(", ndim=") +
        std::to_string(ndim);
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert(param().axis < static_cast<int32_t>(ndim), "%s",
                  errmsg.c_str());
    // ensure shape size for each axis is correct
    for (size_t i = 0; i < ndim; ++i) {
        if (i == static_cast<size_t>(param().axis)) {
            size_t sum = 0_z;
            for (auto &&src: srcs) sum += src.shape[i];
            megdnn_assert_eq_size_t(sum, dst.shape[i]);
        } else {
			for (auto &&src: srcs) {
				megdnn_assert(src.shape[i] == dst.shape[i]);
                megdnn_assert_eq_size_t(src.shape[i], dst.shape[i]);
			}
        }
    }
}

void ConcatSplitBase::get_ABC(const TensorShapeArray &srcs,
        size_t &A,
        size_t *B,
        size_t &C)
{
    auto axis = param().axis;
    auto shape_arr = srcs[0].shape;
    auto ndim = srcs[0].ndim;
    A = std::accumulate(shape_arr, shape_arr + axis,
            1_z, SafeMultiplies<size_t>());
    for (size_t i = 0u; i < srcs.size(); ++i) {
        B[i] = srcs[i].shape[axis];
    }
    C = std::accumulate(shape_arr + (axis+1), shape_arr + ndim,
            1_z, SafeMultiplies<size_t>());
}

void ConcatForward::deduce_layout(const TensorLayoutArray &srcs,
        TensorLayout &dst)
{
    dst = srcs[0];
    auto i = param().axis;
    dst.shape[i] = 0u;
    for (auto &&src: srcs) {
        dst.shape[i] += src.shape[i];
    }
    dst.init_contiguous_stride();
}

void ConcatForward::check_exec(const TensorLayoutArray &srcs,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_common(srcs, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(srcs, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void SplitForward::check_exec(const TensorLayout &src,
        const TensorLayoutArray &dsts,
        size_t workspace_in_bytes)
{
    check_layout_common(dsts, src);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dsts);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn
// vim: syntax=cpp.doxygen

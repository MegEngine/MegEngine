/**
 * \file dnn/src/common/reduce.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include <numeric>
#include "src/common/utils.h"

namespace {
using namespace megdnn;
using megdnn::Reduce;

DType get_out_dtype(const Reduce::DataType data_type, const DType inp_dtype) {
    if (data_type == Reduce::DataType::FLOAT_O16xC32) {
#if !MEGDNN_DISABLE_FLOAT16
        return dtype::Float16();
#else
        megdnn_assert_internal(0);
#endif
    }
    if (data_type == Reduce::DataType::FLOAT_O32xC32) {
        return dtype::Float32();
    }
    if (data_type == Reduce::DataType::QUINT_I8xO32) {
        megdnn_assert(inp_dtype.enumv() == DTypeEnum::Quantized8Asymm);
        return dtype::QuantizedS32(
                inp_dtype.param<dtype::Quantized8Asymm>().scale);
    }
    if (data_type == Reduce::DataType::QINT_I8xO32) {
        megdnn_assert(inp_dtype.enumv() == DTypeEnum::QuantizedS8);
        return dtype::QuantizedS32(
                inp_dtype.param<dtype::QuantizedS8>().scale);
    }
    megdnn_assert(data_type == Reduce::DataType::DEFAULT);
    return inp_dtype;
}
}  // namespace

namespace megdnn {

void ReduceForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    megdnn_assert(
            param().axis >= 0 && static_cast<uint32_t>(param().axis) < src.ndim,
            "axis: %d ndim: %zu", param().axis, src.ndim);
    dst = src;
    dst.shape[param().axis] = 1;

    dst.dtype = get_out_dtype(param().data_type, src.dtype);
    dst.format = src.format;
    dst.init_contiguous_stride();
}

void ReduceForward::check_exec(const TensorLayout& src, const TensorLayout& dst,
                               size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(dst);
    };
    megdnn_assert(param().data_type != Reduce::DataType::FLOAT_IO16xC32,
                  "FLOAT_IO16xC32 is deprecated");
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(dst);
    megdnn_assert(src.ndim == dst.ndim, "%s", errmsg().c_str());
    megdnn_assert(param().axis >= 0);
    uint32_t axis = param().axis;
    megdnn_assert(axis < src.ndim, "%s", errmsg().c_str());
    rep(i, src.ndim) {
        if (i != axis) {
            megdnn_assert(src.shape[i] == dst.shape[i], "%s", errmsg().c_str());
        } else {
            megdnn_assert(dst.shape[i] == 1_z, "%s", errmsg().c_str());
        }
    }
    megdnn_assert(src.dtype.category() == dst.dtype.category() ||
                  param().data_type == Reduce::DataType::FLOAT_O32xC32,
                  "the category of reduce output and input must be the same,"
                  " or the data_type is FLOAT_O32xC32");
    if (param().data_type == DataType::DEFAULT) {
        megdnn_assert(src.dtype == dst.dtype &&
                      (src.dtype.category() == DTypeCategory::FLOAT ||
                       src.dtype.category() == DTypeCategory::INT ||
                       src.dtype.category() == DTypeCategory::QUANTIZED));
    } else if (param().data_type == DataType::QUINT_I8xO32) {
        megdnn_assert(src.dtype.enumv() == DTypeEnum::Quantized8Asymm);
    } else if (param().data_type == DataType::QINT_I8xO32) {
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8);
    } else if (param().data_type == DataType::FLOAT_IO16xC32 ||
               param().data_type == DataType::FLOAT_O16xC32) {
        megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
    } else {
        megdnn_assert(param().data_type == DataType::FLOAT_O32xC32);
    }

    auto expected = get_out_dtype(param().data_type, src.dtype);
    megdnn_assert(expected == dst.dtype);

    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen

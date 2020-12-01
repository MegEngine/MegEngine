/**
 * \file dnn/src/cuda/relayout_format/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/handle.h"
#include "src/cuda/relayout_format/opr_impl.h"
#include "src/cuda/relayout_format/relayout_format.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

void RelayoutFormatImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                              _megdnn_workspace /* workspace */) {
    auto src_dtype = src.layout.dtype;
    megdnn_assert(
            param().mode == param::RelayoutFormat::Mode::NCHW4_CHWN4 ||
                    param().mode == param::RelayoutFormat::Mode::NCHW_NCHW4 ||
                    param().mode == param::RelayoutFormat::Mode::CHWN4_NCHW4 ||
                    param().mode == Param::Mode::NCHW_NCHW4_IC_SMALL ||
                    param().mode ==
                            Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT,
            "relayout format of cuda only support NCHW4->CHWN4 or "
            "CHWN4->NCHW4 or NCHW->NCHW4");
    if ((param().mode == param::RelayoutFormat::Mode::NCHW4_CHWN4 ||
         param().mode == param::RelayoutFormat::Mode::CHWN4_NCHW4) &&
        src_dtype.enumv() == DTypeEnum::QuantizedS8) {
        size_t row = 0, col = 0;
        if (param().mode == Param::RelayoutFormat::Mode::NCHW4_CHWN4) {
            row = src.layout[0],
            col = src.layout[1] * src.layout[2] * src.layout[3];
        } else {
            megdnn_assert(param().mode ==
                          param::RelayoutFormat::Mode::CHWN4_NCHW4);
            row = src.layout[0] * src.layout[1] * src.layout[2],
            col = src.layout[3];
        }
        TensorND trans_in, trans_out;
        trans_in.raw_ptr = src.raw_ptr;
        trans_in.layout = {{row, col}, dtype::Int32()};
        trans_in.layout.init_contiguous_stride();
        trans_out.raw_ptr = dst.raw_ptr;
        trans_out.layout = trans_in.layout;
        trans_out.layout.stride[0] = 1;
        trans_out.layout.stride[1] = row;
        return handle()->create_operator<RelayoutForward>()->exec(trans_in,
                                                                  trans_out);
    }
    if ((param().mode == Param::Mode::NCHW_NCHW4_IC_SMALL ||
         param().mode == Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT) &&
        src.layout[1] % 4 != 0) {
        megdnn_assert(src.raw_ptr != dst.raw_ptr && src.layout.ndim == 4,
                      "The mode of NCHW_NCHW4 and NCHW_NCHW4_CONV_DENSE_WEIGHT "
                      "of RelayoutFormat opr(cuda backend) does not support "
                      "src.ptr == dst.ptr");
        megdnn_assert(src.layout[1] <= 4);
        cuda_check(cudaMemsetAsync(dst.raw_ptr, 0,
                                   dst.layout.span().dist_byte(),
                                   cuda_stream(this->handle())));
        TensorLayout exec_dst_layout = dst.layout;
        exec_dst_layout[4] = src.layout[1];
        TensorLayout exec_src_layout =
                src.layout
                        .reshape({src.layout[0], src.layout[1], 1,
                                  src.layout[2], src.layout[3]})
                        .dimshuffle({0, 2, 3, 4, 1});
        return handle()->create_operator<RelayoutForward>()->exec(
                {src.raw_ptr, exec_src_layout}, {dst.raw_ptr, exec_dst_layout});
    }

    if (param().mode == Param::Mode::NCHW_NCHW4) {
        bool is_usable = relayout_format::RelayoutFormatFast::usable(
                src.layout, dst.layout);
        megdnn_assert(is_usable,
                      "RelayoutFormatNCHW_NCHW4 kernel not usable for %s(%s) "
                      "to %s(%s)",
                      src.layout.to_string().c_str(), src.layout.dtype.name(),
                      dst.layout.to_string().c_str(), dst.layout.dtype.name());
        relayout_format::RelayoutFormatFast::exec(src, dst,
                                                  cuda_stream(this->handle()));
    } else {
        TensorLayout exec_src, exec_dst;
        deduce_exec_layout(src.layout, dst.layout, exec_src, exec_dst);
        TensorND exec_src_nd{src.raw_ptr, exec_src};
        TensorND exec_dst_nd{dst.raw_ptr, exec_dst};
        handle()->create_operator<RelayoutForward>()->exec(exec_src_nd,
                                                           exec_dst_nd);
    }
}

size_t RelayoutFormatImpl::get_workspace_in_bytes(
        const TensorLayout& /* src */, const TensorLayout& /* dst */) {
    return 0;
}

// vim: syntax=cpp.doxygen

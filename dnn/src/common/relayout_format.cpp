/**
 * \file dnn/src/common/relayout_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/oprs.h"
#include "megdnn/tensor_format.h"
#include "src/common/utils.h"

using namespace megdnn;

void RelayoutFormat::deduce_layout_fwd(const TensorLayout& src,
                                       TensorLayout& dst) {
    using Param = param::RelayoutFormat;
    switch (param().mode) {
        case Param::Mode::NCHW_NHWCD4:
        case Param::Mode::NCHW_NHWCD4I:
            dst.ndim = 5;
            dst[0] = src[0];
            dst[1] = src[2];
            dst[2] = (src[1] + 3) / 4;
            dst[3] = src[3];
            dst[4] = 4;
            break;
        case Param::Mode::NCHW_NCHW4_IC_SMALL:
            dst.ndim = 5;
            megdnn_assert(src[1] <= 4_z, "ic should be less equal 4");
            dst[0] = src[0];
            dst[1] = div_ceil(src[1], 4_z);
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = 4;
            break;
        case Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT:
            megdnn_assert(src.ndim == 4, "src must be oihw, ndim == 4");
            megdnn_assert(src[1] <= 4_z, "ic should be less equal 4");
            dst.ndim = 5;
            dst[0] = src[0];
            dst[1] = div_ceil(src[1], 4_z);
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = 4;
            break;

        case Param::Mode::NCHW_NCHW88:
            dst.ndim = 5;
            dst[0] = src[0];
            dst[1] = div_ceil(src[1], 8_z);
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = 8;
            break;
        case Param::Mode::NCHW88_NCHW:
            dst.ndim = 4;
            dst[0] = src[0];
            dst[1] = src[1] * 8;
            dst[2] = src[2];
            dst[3] = src[3];
            break;
        case Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT:
            megdnn_assert(src.ndim == 4, "src must be oihw, ndim == 4");
            dst.ndim = 6;
            megdnn_assert(src[0] % 8 == 0,
                          "NCHW_NCHW88_CONV_DENSE_WEIGHT out channel must "
                          "align to 8");
            dst[0] = src[0] / 8;
            dst[1] = div_ceil(src[1], 8_z);
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = 8;
            dst[5] = 8;
            break;
        case Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT:
            megdnn_assert(src.ndim == 5, "src must be goihw, ndim == 5");
            dst.ndim = 6;
            dst[0] = div_ceil(src[0], 8_z);
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = src[4];
            dst[5] = 8;
            break;
        case Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT:
            megdnn_assert(src.ndim == 5, "src must be goihw, ndim == 5");
            dst.ndim = 7;
            dst[0] = src[0];
            megdnn_assert(src[1] % 8 == 0,
                          "NCHW_NCHW88_CONV_GROUP_WEIGHT out channel must "
                          "align to 8");
            dst[1] = src[1] / 8;
            dst[2] = div_ceil(src[2], 8_z);
            dst[3] = src[3];
            dst[4] = src[4];
            dst[5] = 8;
            dst[6] = 8;
            break;
        case Param::Mode::NHWC_NHWCD4:
        case Param::Mode::NHWC_NHWCD4I:
            megdnn_assert(src.ndim == 4);
            //! channel mod 4 should == 4
            megdnn_assert(src[3] % 4 == 0);
            dst.ndim = 5;
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[3] / 4;
            dst[3] = src[2];
            dst[4] = 4;
            break;
        case Param::Mode::NHWCD4_NHWC:
            megdnn_assert(src.ndim == 5);
            dst.ndim = 4;
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[3];
            dst[3] = src[2] * 4;
            break;
        case Param::Mode::NHWCD4_NCHW:
        case Param::Mode::NHWCD4I_NCHW:
            megdnn_assert(src.ndim == 5);
            dst.ndim = 4;
            dst[0] = src[0];
            dst[1] = src[2] * 4;
            dst[2] = src[1];
            dst[3] = src[3];
            break;
        case Param::Mode::INTER_WEIGHT_DENSE:
        case Param::Mode::INTER_WEIGHT_DENSEI:
            megdnn_assert(src.ndim == 4);
            megdnn_assert(src[0] % 4 == 0);
            dst.ndim = 5;
            dst[0] = src[0] / 4;
            dst[1] = src[2];
            dst[2] = src[3];
            dst[3] = round_up<size_t>(src[1], 4);
            dst[4] = 4;
            break;
        case Param::Mode::INTER_WEIGHT_GROUP:
        case Param::Mode::INTER_WEIGHT_GROUPI:
            // group conv filter
            megdnn_assert(src.ndim == 5);
            megdnn_assert(src[1] % 4 == 0 && src[2] % 4 == 0);
            dst.ndim = 6;
            dst[0] = src[0];
            dst[1] = src[1] / 4;
            dst[2] = src[3];
            dst[3] = src[4];
            dst[4] = src[2];
            dst[5] = 4;
            break;
        case Param::Mode::INTER_WEIGHT_CHAN:
        case Param::Mode::INTER_WEIGHT_CHANI:
            megdnn_assert(src.ndim == 5 && src[1] == 1 && src[2] == 1);
            // chanwise conv filter
            dst.ndim = 5;
            dst[0] = src[0] / 4;
            dst[1] = 1;
            dst[2] = src[3];
            dst[3] = src[4];
            dst[4] = 4;
            break;
        case Param::Mode::INTER_WEIGHT_DENSEI_DOT:
            megdnn_assert(src.ndim == 4);
            megdnn_assert(src[0] % 4 == 0);
            dst.ndim = 6;
            dst[0] = src[0] / 4;
            dst[1] = src[2];
            dst[2] = src[3];
            dst[3] = div_ceil<size_t>(src[1], 4);
            dst[4] = 4;
            dst[5] = 4;
            break;
        case Param::Mode::INTER_WEIGHT_GROUPI_DOT:
            megdnn_assert(src.ndim == 5);
            megdnn_assert(src[1] % 4 == 0 && src[2] % 4 == 0);
            dst.ndim = 7;
            dst[0] = src[0];
            dst[1] = src[1] / 4;
            dst[2] = src[3];
            dst[3] = src[4];
            dst[4] = src[2] / 4;
            dst[5] = 4;
            dst[6] = 4;
            break;
        case Param::Mode::NCHW4_CHWN4:
            megdnn_assert(src.ndim == 5);
            megdnn_assert(src[4] == 4);
            dst.ndim = 5;
            dst[0] = src[1];
            dst[1] = src[2];
            dst[2] = src[3];
            dst[3] = src[0];
            dst[4] = src[4];
            break;
        case Param::Mode::CHWN4_NCHW4:
            megdnn_assert(src.ndim == 5);
            megdnn_assert(src[4] == 4);
            dst.ndim = 5;
            dst[0] = src[3];
            dst[1] = src[0];
            dst[2] = src[1];
            dst[3] = src[2];
            dst[4] = src[4];
            break;
        case Param::Mode::NCHW_NCHW4:
            megdnn_assert(src.ndim == 4);
            dst.ndim = 5;
            dst[0] = src[0];
            dst[1] = div_ceil<size_t>(src[1], 4);
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = 4;
            break;
        default:
            megdnn_assert(0, "Invalid RelayoutFormat Mode");
            break;
    }
    TensorFormat dst_fmt;
    deduce_format(src.format, dst_fmt);
    dst.format = dst_fmt;
    if (!dst.dtype.valid()) {
        dst.dtype = src.dtype;
    }
    dst.init_contiguous_stride();
}

void RelayoutFormat::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    deduce_layout_fwd(src, dst);
}

void RelayoutFormat::deduce_format(TensorFormat src, TensorFormat& dst) {
    size_t align = handle()->image2d_pitch_alignment();
    using Param = param::RelayoutFormat;
#define CHECK_SRC(_expect)                                                \
    megdnn_assert(src == _expect, "invalid src format: expect=%s got=%s", \
                  _expect.to_string().c_str(), src.to_string().c_str())
    switch (param().mode) {
        case Param::Mode::NHWC_NHWCD4:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::NHWCD4_NHWC:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::NHWC_NHWCD4I:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = Image2DPack4TensorFormat::make_raw(2, align);
            break;
        case Param::Mode::NCHW_NHWCD4:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::NCHW_NCHW4:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::NCHW_NHWCD4I:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = Image2DPack4TensorFormat::make_raw(2, align);
            break;
        case Param::Mode::NHWCD4I_NCHW:
            CHECK_SRC(Image2DPack4TensorFormat::make_raw(2, align));
            dst = DefaultTensorFormat::make();
            break;
        case Param::Mode::NHWCD4_NCHW:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::INTER_WEIGHT_DENSE:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::INTER_WEIGHT_DENSEI:
        case Param::Mode::INTER_WEIGHT_DENSEI_DOT:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = Image2DPack4TensorFormat::make_raw(3, align);
            break;
        case Param::Mode::INTER_WEIGHT_GROUP:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::INTER_WEIGHT_GROUPI:
        case Param::Mode::INTER_WEIGHT_GROUPI_DOT:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = Image2DPack4TensorFormat::make_raw(4, align);
            break;
        case Param::Mode::INTER_WEIGHT_CHAN:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::INTER_WEIGHT_CHANI:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = Image2DPack4TensorFormat::make_raw(1, align);
            break;
        case Param::Mode::NCHW4_CHWN4:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::CHWN4_NCHW4:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;
        case Param::Mode::NCHW_NCHW88:
        case Param::Mode::NCHW88_NCHW:
        case Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT:
        case Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT:
        case Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT:
        case Param::Mode::NCHW_NCHW4_IC_SMALL:
        case Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT:
            CHECK_SRC(DefaultTensorFormat::make());
            dst = src;
            break;

        default:
            megdnn_throw("Invalid relayout format mode");
            break;
    }

    if (!dst.is_default() &&
        (
                handle()->type() != Handle::HandleType::NAIVE)) {
        megdnn_throw(
                "Only naive and opencl handle support "
                "Image2DPack4TensorFormat, try to export MGB_USE_MEGDNN_DBG=2 "
                "and also export CUDA_VISIBLE_DEVICES=\'\' at CUDA env"
                "to enable naive handle");
    }
#undef CHECK_SRC
}

void RelayoutFormat::check_layout_fwd(const TensorLayout& src,
                                      const TensorLayout& dst) {
    TensorLayout dst_expected;
    dst_expected.dtype = dst.dtype;
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
}

void RelayoutFormat::check_exec(const TensorLayout& src,
                                const TensorLayout& dst,
                                size_t workspace_in_bytes) {
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void RelayoutFormat::deduce_exec_layout(const TensorLayout& src,
                                        const TensorLayout& dst,
                                        TensorLayout& exec_src,
                                        TensorLayout& exec_dst) {
    check_layout_fwd(src, dst);
    using Param = param::RelayoutFormat;
    switch (param().mode) {
        case Param::Mode::NCHW_NCHW88:
            // nchw to nchw8c
            {
                TensorLayout work_space_layout(
                        {src[0], round_up(src[1], 8_z), src[2], src[3]},
                        src.dtype, src.format);
                exec_src = work_space_layout
                                   .reshape({src[0], div_ceil(src[1], 8_z), 8,
                                             src[2], src[3]})
                                   .dimshuffle({0, 1, 3, 4, 2});
                exec_dst = dst;
            }
            break;
        case Param::Mode::NCHW_NCHW4:
            // nchw to nchw4
            {
                TensorLayout work_space_layout(
                        {src[0], round_up(src[1], 4_z), src[2], src[3]},
                        src.dtype, src.format);
                exec_src = work_space_layout
                                   .reshape({src[0], div_ceil(src[1], 4_z), 4,
                                             src[2], src[3]})
                                   .dimshuffle({0, 1, 3, 4, 2});
                exec_dst = dst;
            }
            break;
        case Param::Mode::NCHW88_NCHW:
            // nchw8c to nchw
            exec_src = src;
            exec_dst = dst.reshape({dst[0], dst[1] / 8, 8, dst[2], dst[3]})
                               .dimshuffle({0, 1, 3, 4, 2});
            break;
        case Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT:
            // oihw to oihw8i8o
            {
                megdnn_assert(src.ndim == 4);
                megdnn_assert(src[0] % 8 == 0);
                TensorLayout work_space_layout(
                        {src[0], round_up(src[1], 8_z), src[2], src[3]},
                        src.dtype, src.format);
                exec_src =
                        work_space_layout
                                .reshape({src[0] / 8, 8, div_ceil(src[1], 8_z),
                                          8, src[2], src[3]})
                                .dimshuffle({0, 2, 4, 5, 3, 1});
                exec_dst = dst;
            }
            break;
        case Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT:
            // goihw to goihw8g
            {
                megdnn_assert(src.ndim == 5);
                TensorLayout work_space_layout(
                        {round_up(src[0], 8_z), src[1], src[2], src[3], src[4]},
                        src.dtype, src.format);
                exec_src = work_space_layout
                                   .reshape({div_ceil(src[0], 8_z), 8, src[1],
                                             src[2], src[3], src[4]})
                                   .dimshuffle({0, 2, 3, 4, 5, 1});
                exec_dst = dst;
            }
            break;
        case Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT:
            // goihw to goihw8i8o
            {
                megdnn_assert(src.ndim == 5);
                megdnn_assert(src[1] % 8 == 0);
                TensorLayout work_space_layout(
                        {src[0], src[1], round_up(src[2], 8_z), src[3], src[4]},
                        src.dtype, src.format);
                exec_src = work_space_layout
                                   .reshape({src[0], src[1] / 8, 8,
                                             div_ceil(src[2], 8_z), 8, src[3],
                                             src[4]})
                                   .dimshuffle({0, 1, 3, 5, 6, 4, 2});
                exec_dst = dst;
            }
            break;

        case Param::Mode::NCHW_NCHW4_IC_SMALL:
        case Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT:
            // nchw to nchw4c or oihw to oihw4i
            {
                TensorLayout work_space_layout(
                        {src[0], round_up(src[1], 4_z), src[2], src[3]},
                        src.dtype, src.format);
                exec_src = work_space_layout
                                   .reshape({src[0], div_ceil(src[1], 4_z), 4,
                                             src[2], src[3]})
                                   .dimshuffle({0, 1, 3, 4, 2});
                exec_dst = dst;
            }
            break;

        case Param::Mode::NCHW_NHWCD4:
        case Param::Mode::NCHW_NHWCD4I:
            // src is {N, C, H, W}
            // dst is {N, H, CB, W, 4}
            exec_src = src;
            exec_src[1] = (exec_src[1] + 3) / 4 * 4;
            exec_src.stride[0] = exec_src[1] * exec_src.stride[1];
            exec_src = exec_src.dimshuffle({0, 2, 3, 1});
            exec_src = exec_src.reshape({exec_src[0], exec_src[1], exec_src[2],
                                         exec_src[3] / 4, 4})
                               .dimshuffle({0, 1, 3, 2, 4});
            exec_dst = dst;
            break;
        case Param::Mode::NHWC_NHWCD4:
        case Param::Mode::NHWC_NHWCD4I:
            // src is {N, H, W, C},
            // dst is {N, H, CB, W, 4}
            exec_src = src.reshape({src[0], src[1], src[2], src[3] / 4, 4})
                               .dimshuffle({0, 1, 3, 2, 4});
            exec_dst = dst;
            break;
        case Param::Mode::NHWCD4_NHWC:
            // src is {N, H, CB, W, 4}
            // dst is {N, H, W, C},
            exec_src = src;
            exec_dst = dst.reshape({dst[0], dst[1], dst[2], dst[3] / 4, 4})
                               .dimshuffle({0, 1, 3, 2, 4});
            break;
        case Param::Mode::NHWCD4_NCHW:
        case Param::Mode::NHWCD4I_NCHW:
            exec_src = src;
            exec_dst = dst.reshape({dst[0], dst[1] / 4, 4, dst[2], dst[3]})
                               .dimshuffle({0, 3, 1, 4, 2});
            break;
        case Param::Mode::INTER_WEIGHT_DENSE:
        case Param::Mode::INTER_WEIGHT_DENSEI:
            // src is {OC, IC, FH, FW}
            // dst is {OCB, FH, FW, IC, 4}
            exec_src = src.reshape({src[0] / 4, 4, src[1], src[2], src[3]})
                               .dimshuffle({0, 3, 4, 2, 1});
            exec_dst = dst;
            // dst[3] may be round_uped, set to the real ic
            exec_dst.shape[3] = src[1];
            break;
        case Param::Mode::INTER_WEIGHT_GROUP:
        case Param::Mode::INTER_WEIGHT_GROUPI:
            // group conv filter
            // src is {G, ocpg, icpg, fh, fw}
            // dst is {G, ocpgb, fh, fw, icpg, 4}
            exec_src =
                    src.reshape({src[0], src[1] / 4, 4, src[2], src[3], src[4]})
                            .dimshuffle({0, 1, 4, 5, 3, 2});
            exec_dst = dst;
            break;
        case Param::Mode::INTER_WEIGHT_CHAN:
        case Param::Mode::INTER_WEIGHT_CHANI:
            megdnn_assert(src.ndim == 5);
            megdnn_assert(src[1] == 1 && src[2] == 1);
            // chanwise conv filter
            megdnn_assert(src[0] % 4 == 0);
            exec_src = src.reshape({src[0] / 4, 4, 1, src[3], src[4]})
                               .dimshuffle({0, 2, 3, 4, 1});
            exec_dst = dst;
            break;
        case Param::Mode::INTER_WEIGHT_DENSEI_DOT:
            // src is {oc, ic, fh , fw}
            // dst is {oc/4, fh, fw, ic/4, 4, 4}
            exec_src = src;
            exec_src[1] = round_up<size_t>(src[1], 4);
            exec_src.stride[0] = exec_src.stride[1] * exec_src[1];
            exec_src = exec_src.reshape({exec_src[0] / 4, 4, exec_src[1] / 4, 4,
                                         exec_src[2], exec_src[3]})
                               .dimshuffle({0, 4, 5, 2, 1, 3});
            exec_dst = dst;
            break;
        case Param::Mode::INTER_WEIGHT_GROUPI_DOT:
            // src is {G, ocpg, icpg, fh, fw}
            // dst is {G, ocpg/4, fh, fw, icpg/4, 4, 4}
            exec_src = src.reshape({src[0], src[1] / 4, 4, src[2] / 4, 4,
                                    src[3], src[4]})
                               .dimshuffle({0, 1, 5, 6, 3, 2, 4});
            exec_dst = dst;
            break;
        case Param::Mode::NCHW4_CHWN4:
            // src is {N, C/4, H, W, 4}
            // dst is {C/4, H, W, N, 4}
            exec_src = src.dimshuffle({1, 2, 3, 0, 4});
            exec_dst = dst;
            break;
        case Param::Mode::CHWN4_NCHW4:
            // src is {C/4, H, W, N, 4}
            // dst is {N, C/4, H, W, 4}
            exec_src = src.dimshuffle({3, 0, 1, 2, 4});
            exec_dst = dst;
            break;
        default:
            megdnn_assert(0, "Invalid RelayoutFormat Mode");
    }
}

// vim: syntax=cpp.doxygen

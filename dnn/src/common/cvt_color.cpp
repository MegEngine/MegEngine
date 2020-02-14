/**
 * \file dnn/src/common/cvt_color.cpp
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

namespace megdnn {

void CvtColorBase::deduce_layout_fwd(const TensorLayout& src,
                                     TensorLayout& dst) {
    auto errmsg = [&]() { return megdnn_layout_msg(src); };
    MEGDNN_MARK_USED_VAR(errmsg);

    auto mode = param().mode;
    if (mode == Param::Mode::YUV2RGB_NV21 ||
        mode == Param::Mode::YUV2BGR_NV21 ||
        mode == Param::Mode::YUV2RGB_NV12 ||
        mode == Param::Mode::YUV2BGR_NV12 ||
        mode == Param::Mode::YUV2RGB_YV12 ||
        mode == Param::Mode::YUV2BGR_YV12 ||
        mode == Param::Mode::YUV2RGB_YU12 ||
        mode == Param::Mode::YUV2BGR_YU12) {
        megdnn_log_warn(
                "Deprecated mode for cvtcolor, you should refer to the wiki "
                "for detail usage");
    }
    //! The origin YUV is YCrCb in opencv as histrical reasons, it will remove
    //! later
    if (mode == Param::Mode::YUV2RGB_NV21) {
        mode = Param::Mode::YCrCb2RGB;
    }
    if (mode == Param::Mode::YUV2BGR_NV21) {
        mode = Param::Mode::YCrCb2BGR;
    }

    megdnn_assert(
            src.ndim == 4_z && (src.shape[3] == 1_z || src.shape[3] == 3_z ||
                                src.shape[3] == 4_z),
            "%s", errmsg().c_str());

    size_t in = src.shape[0];
    size_t ih = src.shape[1];
    size_t iw = src.shape[2];
    size_t ic = src.shape[3];

    size_t oc = 1;
    size_t oh = ih;
    size_t ow = iw;

    switch (mode) {
        case Param::Mode::RGB2GRAY:
            megdnn_assert(ic == 3);
            oc = 1;
            break;
        case Param::Mode::RGB2YUV:
            megdnn_assert(ic == 3);
            oc = 3;
            break;
        case Param::Mode::YUV2RGB:
            megdnn_assert(ic == 3);
            oc = 3;
            break;
        case Param::Mode::GRAY2RGB:
            megdnn_assert(ic == 1);
            oc = 3;
            break;
        case Param::Mode::RGBA2RGB:
            megdnn_assert(ic == 4);
            oc = 3;
            break;
        case Param::Mode::RGBA2BGR:
            megdnn_assert(ic == 4);
            oc = 3;
            break;
        case Param::Mode::RGBA2GRAY:
            megdnn_assert(ic == 4);
            oc = 1;
            break;
        case Param::Mode::RGB2BGR:
            megdnn_assert(ic == 3);
            oc = 3;
            break;
        case Param::Mode::BGR2GRAY:
            megdnn_assert(ic == 3);
            oc = 1;
            break;
        case Param::Mode::BGR2RGB:
            megdnn_assert(ic == 3);
            oc = 3;
            break;
        case Param::Mode::YUV2GRAY_NV21:
        case Param::Mode::YUV2GRAY_NV12:
            megdnn_assert(ic == 1 && ih % 3 == 0 && iw % 2 == 0);
            oh = ih / 3 * 2;
            oc = 1;
            break;
        case Param::Mode::YUV2GRAY_YV12:
        case Param::Mode::YUV2GRAY_YU12:
            megdnn_assert(ic == 1 && ih % 6 == 0 && iw % 2 == 0);
            oh = ih / 3 * 2;
            oc = 1;
            break;
        case Param::Mode::YCrCb2BGR:
        case Param::Mode::YCrCb2RGB:
        case Param::Mode::YUV2RGB_NV21:
        case Param::Mode::YUV2RGB_NV12:
        case Param::Mode::YUV2BGR_NV21:
        case Param::Mode::YUV2BGR_NV12:
        case Param::Mode::BT601_YUV2RGB_NV21:
        case Param::Mode::BT601_YUV2RGB_NV12:
        case Param::Mode::BT601_YUV2BGR_NV21:
        case Param::Mode::BT601_YUV2BGR_NV12:
            megdnn_assert(ic == 1 && ih % 3 == 0 && iw % 2 == 0);
            oh = ih / 3 * 2;
            oc = 3;
            break;
        case Param::Mode::YUV2RGB_YV12:
        case Param::Mode::YUV2RGB_YU12:
        case Param::Mode::YUV2BGR_YV12:
        case Param::Mode::YUV2BGR_YU12:
        case Param::Mode::BT601_YUV2RGB_YV12:
        case Param::Mode::BT601_YUV2RGB_YU12:
        case Param::Mode::BT601_YUV2BGR_YV12:
        case Param::Mode::BT601_YUV2BGR_YU12:
            megdnn_assert(ic == 1 && ih % 6 == 0 && iw % 2 == 0);
            oh = ih / 3 * 2;
            oc = 3;
            break;
        default:
            megdnn_throw("Can not find property cvt_color operator.");
    }

    dst = TensorLayout(TensorShape({in, oh, ow, oc}), src.dtype);
}

void CvtColorBase::check_layout_fwd(const TensorLayout& src,
                                    const TensorLayout& dst) {
    megdnn_assert_eq_dtype(src, dst);
    TensorLayout dst_expected;
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_shape(dst_expected, dst);
}

void CvtColor::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    deduce_layout_fwd(src, dst);
}

void CvtColor::check_exec(const TensorLayout& src, const TensorLayout& dst,
                          size_t workspace_in_bytes) {
    check_layout_fwd(src, dst);
    megdnn_assert_contiguous(src);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen

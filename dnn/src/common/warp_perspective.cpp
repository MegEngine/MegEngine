/**
 * \file dnn/src/common/warp_perspective.cpp
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

void WarpPerspectiveBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &mat,
        const TensorLayout &mat_idx,
        const TensorLayout &dst)
{
    megdnn_assert_contiguous(mat);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(dst);
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " +
            megdnn_layout_msg(mat) + ", " +
            megdnn_layout_msg(mat_idx) + ", " +
            megdnn_layout_msg(dst) + ", " +
            param_msg();
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    if (param().format == param::WarpPerspective::Format::NHWCD4 ||
        param().format == param::WarpPerspective::Format::NCHW4) {
        megdnn_assert(src.ndim == 5_z, "%s", errmsg().c_str());
        megdnn_assert(dst.ndim == 5_z, "%s", errmsg().c_str());

    } else {
        megdnn_assert(param().format == param::WarpPerspective::Format::NHWC ||
                      param().format == param::WarpPerspective::Format::NCHW);
        megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
        megdnn_assert(dst.ndim == 4_z, "%s", errmsg().c_str());
    }
    megdnn_assert(mat.ndim == 3_z, "%s", errmsg().c_str());
    megdnn_assert(dst.shape[0] == mat.shape[0], "%s", errmsg().c_str());
    if (mat_idx.ndim) {
        megdnn_assert(mat_idx.dtype == dtype::Int32() && mat_idx.ndim == 1,
                "%s", errmsg().c_str());
        megdnn_assert(mat.shape[0] == mat_idx.shape[0], "%s", errmsg().c_str());
        megdnn_assert_contiguous(mat_idx);
    } else {
        megdnn_assert(src.shape[0] == dst.shape[0], "%s", errmsg().c_str());
    }
    megdnn_assert(mat.shape[1] == 3_z, "%s", errmsg().c_str());
    megdnn_assert(mat.shape[2] == 3_z, "%s", errmsg().c_str());

    if (param().format == param::WarpPerspective::Format::NCHW) {
        megdnn_assert(
                src.dtype.enumv() == DTypeEnum::Float32 ||
                        MEGDNN_FLOAT16_SELECT(
                                (src.dtype.enumv() == DTypeEnum::Float16 ||
                                 src.dtype.enumv() == DTypeEnum::BFloat16),
                                false) ||
                        src.dtype.enumv() == DTypeEnum::Int8 ||
                        src.dtype.enumv() == DTypeEnum::Uint8 ||
                        (src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                         src.dtype.enumv() == DTypeEnum::Quantized8Asymm),
                "WarpPerspective NCHW input dtype should be "
                "Float32/Int8/Uint8/QInt8/QUint8" MEGDNN_FLOAT16_SELECT(
                        "/Float16/BFloat16", "") ".");
        megdnn_assert(
                (src.dtype.category() == DTypeCategory::FLOAT &&
                 (src.dtype == mat.dtype ||
                  mat.dtype.enumv() == DTypeEnum::Float32)) ||
                        ((src.dtype.category() == DTypeCategory::INT ||
                          src.dtype.category() == DTypeCategory::QUANTIZED) &&
                         mat.dtype.enumv() == DTypeEnum::Float32),
                "The input to WarpPerspective is in NCHW format, in this "
                "case, if the input dtype is floating point, the "
                "transformation matrix should have same dtype as the "
                "input, otherwise, it should be in Float32, %s given.",
                mat.dtype.name());

        megdnn_assert(dst.dtype == src.dtype);
        megdnn_assert(src.shape[1] == dst.shape[1], "%s", errmsg().c_str());

        megdnn_assert(param().imode ==
                      param::WarpPerspective::InterpolationMode::LINEAR);
        megdnn_assert(param().bmode !=
                      param::WarpPerspective::BorderMode::TRANSPARENT);
        megdnn_assert(param().bmode !=
                      param::WarpPerspective::BorderMode::ISOLATED);
    } else if (param().format == param::WarpPerspective::Format::NHWC) {
        megdnn_assert(src.shape[3] == dst.shape[3], "%s", errmsg().c_str());
    } else if (param().format == param::WarpPerspective::Format::NCHW4) {
        megdnn_assert(dst.dtype == src.dtype);
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8,
                "src expected QuantizedS8, but got %s", src.dtype.name());
        megdnn_assert(mat.dtype == dtype::Float32(),
                      "matrix dtype expected float, got %s", mat.dtype.name());
        megdnn_assert(src.shape[4] == 4 && dst.shape[4] == 4);
        megdnn_assert(src.shape[1] == dst.shape[1], "%s", errmsg().c_str());

        megdnn_assert(param().imode ==
                      param::WarpPerspective::InterpolationMode::LINEAR);
        megdnn_assert(param().bmode !=
                      param::WarpPerspective::BorderMode::TRANSPARENT);
        megdnn_assert(param().bmode !=
                      param::WarpPerspective::BorderMode::ISOLATED);
    } else {
        megdnn_assert(param().format == param::WarpPerspective::Format::NHWCD4);
        megdnn_assert(
                src.dtype == dtype::Float32() ||
                        MEGDNN_FLOAT16_SELECT((src.dtype == dtype::Float16() ||
                                               src.dtype == dtype::BFloat16()),
                                              false) ||
                        src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                        src.dtype.enumv() == DTypeEnum::Quantized8Asymm,
                "WarpPerspective NHWCD4 input dtype should be "
                "Float32" MEGDNN_FLOAT16_SELECT(
                        "/Float16/BFloat16",
                        "") ",QunatizedS8, Quantized8Asymm.");
        megdnn_assert(
                (src.dtype == mat.dtype || mat.dtype == dtype::Float32()),
                "The input to WarpPerspective is in NHWCD4 format, in this "
                "case, if the input dtype is floating point, the "
                "transformation matrix should have same dtype as the "
                "input, %s given.",
                mat.dtype.name());
        megdnn_assert(dst.dtype == src.dtype);
        //! number of channels is same
        megdnn_assert(src.shape[2] == dst.shape[2], "%s", errmsg().c_str());
        megdnn_assert(param().imode ==
                      param::WarpPerspective::InterpolationMode::LINEAR);
        megdnn_assert(param().bmode !=
                      param::WarpPerspective::BorderMode::TRANSPARENT);
        megdnn_assert(param().bmode !=
                      param::WarpPerspective::BorderMode::ISOLATED);
    }
    megdnn_assert(src.format == dst.format);
}

std::string WarpPerspectiveBase::param_msg() const
{
    std::string res;
    res.append(megdnn_mangle("imode="));
    switch (param().imode) {
        case InterpolationMode::NEAREST:
            res.append(megdnn_mangle("NEAREST"));
            break;
        case InterpolationMode::LINEAR:
            res.append(megdnn_mangle("LINEAR"));
            break;
        case InterpolationMode::AREA:
            res.append(megdnn_mangle("AREA"));
            break;
        case InterpolationMode::CUBIC:
            res.append(megdnn_mangle("CUBIC"));
            break;
        case InterpolationMode::LANCZOS4:
            res.append(megdnn_mangle("LANCZOS4"));
            break;
    }
    res.append(megdnn_mangle("bmode="));
    switch (param().bmode) {
        case BorderMode::WRAP:
            res.append(megdnn_mangle("WRAP"));
            break;
        case BorderMode::CONSTANT:
            res.append(megdnn_mangle("CONSTANT"));
            break;
        case BorderMode::REFLECT:
            res.append(megdnn_mangle("REFLECT"));
            break;
        case BorderMode::REFLECT_101:
            res.append(megdnn_mangle("REFLECT_101"));
            break;
        case BorderMode::REPLICATE:
            res.append(megdnn_mangle("REPLICATE"));
            break;
        case BorderMode::TRANSPARENT:
            res.append(megdnn_mangle("TRANSPARENT"));
            break;
        case BorderMode::ISOLATED:
            res.append(megdnn_mangle("ISOLATED"));
            break;
    }
    if (param().bmode == BorderMode::CONSTANT) {
        res.append(", " + std::to_string(param().border_val));
    }
    return res;
}

int WarpPerspectiveBase::get_real_coord(int p, int len)
{
    auto bmode = param().bmode;
    if( (unsigned)p < (unsigned)len )
        ;
    else if( bmode == BorderMode::REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( bmode == BorderMode::REFLECT || bmode == BorderMode::REFLECT_101 )
    {
        int delta = (bmode == BorderMode::REFLECT_101);
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( bmode == BorderMode::WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        /*
        if( p >= len )
            p %= len;
        */
        while (p >= len) {
            p -= len;
        }
    }
    else if( bmode == BorderMode::CONSTANT )
        p = -1;
    return p;
}

void WarpPerspectiveForward::check_exec(const TensorLayout &src,
        const TensorLayout &mat,
        const TensorLayout &mat_idx,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_exec_allow_nhwc_mat_idx(src, mat, mat_idx, dst, workspace_in_bytes);
}

void WarpPerspectiveForward::check_exec_allow_nhwc_mat_idx(
        const TensorLayout& src, const TensorLayout& mat,
        const TensorLayout& mat_idx, const TensorLayout& dst,
        size_t workspace_in_bytes) {
    check_layout_fwd(src, mat, mat_idx, dst);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, mat, mat_idx, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    if (param().format != Param::Format::NHWC &&
        param().format != Param::Format::NCHW &&
        param().format != Param::Format::NCHW4) {
        megdnn_assert(!mat_idx.ndim,
                      "mat_idx not supported for current format");
    }
}

void WarpPerspectiveBackwardData::check_exec(const TensorLayout& mat,
                                             const TensorLayout& mat_idx,
                                             const TensorLayout& diff,
                                             const TensorLayout& grad,
                                             size_t workspace_in_bytes) {
    check_layout_fwd(grad, mat, mat_idx, diff);
    megdnn_assert(grad.dtype == dtype::Float32() MEGDNN_INC_FLOAT16(
                                        || grad.dtype == dtype::BFloat16()),
                  "Backward WarpPerspective only supports Float32/BFloat16.");
    auto required_workspace_in_bytes = get_workspace_in_bytes(mat, mat_idx, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void WarpPerspectiveBackwardMat::check_exec(const TensorLayout& src,
                                            const TensorLayout& mat,
                                            const TensorLayout& mat_idx,
                                            const TensorLayout& diff,
                                            const TensorLayout& grad,
                                            size_t workspace_in_bytes) {
    check_layout_fwd(src, mat, mat_idx, diff);
    megdnn_assert_eq_layout(mat, grad);
    megdnn_assert(grad.dtype == dtype::Float32() MEGDNN_INC_FLOAT16(
                                        || grad.dtype == dtype::BFloat16()),
                  "Backward WarpPerspective only supports Float32/BFloat16.");
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, mat, mat_idx, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen

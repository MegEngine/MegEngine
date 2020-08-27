/**
 * \file dnn/src/rocm/relayout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/relayout/relayout.h.hip"
#include "src/rocm/relayout/relayout_contiguous.h.hip"

#include "src/common/utils.h"
#include "src/rocm/utils.h"
#include "src/rocm/relayout/opr_impl.h"

using namespace megdnn;
using namespace rocm;

RelayoutForwardImpl::Param::Param(const TensorND &src, const TensorND &dst,
        RelayoutForwardImpl *opr):
    m_src{src}, m_dst{dst}, m_opr{opr}
{
    opr->check_layout_and_canonize(m_src.layout, m_dst.layout);
}

bool RelayoutForwardImpl::Param::try_copy_contig() {
    auto &&lsrc = m_src.layout, &&ldst = m_dst.layout;
    if (lsrc.ndim != 1 || ldst.ndim != 1)
        return false;
    if (lsrc.stride[0] != 1 || ldst.stride[0] != 1)
        return false;
    hip_check(hipMemcpyAsync(
                m_dst.raw_ptr, m_src.raw_ptr,
                ldst.total_nr_elems() * dtype_size(),
                hipMemcpyDeviceToDevice, m_opr->stream()));
    return true;
}

bool RelayoutForwardImpl::expand_dim2(
        TensorLayout &dst, const TensorLayout &src) {
    megdnn_assert(src.ndim == 2 && dst.ndim == 1);
    megdnn_assert(dst.shape[0] == src.shape[0] * src.shape[1]);
    if (src.stride[1] != 1 || dst.stride[0] != 1)
        return false;
    dst.ndim = 2;
    dst.stride[0] = src.shape[1];
    dst.stride[1] = 1;
    dst.shape[0] = src.shape[0];
    dst.shape[1] = src.shape[1];
    return true;
}

bool RelayoutForwardImpl::Param::try_copy_2d() {
    TensorLayout lsrc = m_src.layout, ldst = m_dst.layout;

    if (lsrc.ndim > 2 || ldst.ndim > 2)
        return false;

    if (ldst.ndim == 1 && lsrc.ndim == 1) {
        megdnn_assert(ldst.stride[0] != 1 || lsrc.stride[0] != 1);
        if (lsrc.stride[0] < 1 || ldst.stride[0] < 1)
            return false;
        // extend to ndim == 2
        megdnn_assert(ldst.shape[0] == lsrc.shape[0]);
        ldst.ndim = lsrc.ndim = 2;
        ldst.shape[1] = lsrc.shape[1] = 1;
        ldst.stride[1] = lsrc.stride[1] = 1;
    } else if (ldst.ndim < 2) {
        if (!expand_dim2(ldst, lsrc))
            return false;
    } else if (lsrc.ndim < 2) {
        if (!expand_dim2(lsrc, ldst))
            return false;
    }
    if (ldst.stride[1] != 1 || lsrc.stride[1] != 1 ||
            ldst.shape[0] != lsrc.shape[0] ||
            ldst.shape[1] != lsrc.shape[1] ||
            ldst.stride[0] < static_cast<ptrdiff_t>(ldst.shape[1]) ||
            lsrc.stride[0] < static_cast<ptrdiff_t>(ldst.shape[1]))
        return false;

    //! TODO: need refactor, hipMemcpy2DAsync has bug
    auto dsize = dtype_size();
    hip_check(hipMemcpy2DAsync(
            m_dst.raw_ptr, ldst.stride[0] * dsize,
            m_src.raw_ptr, lsrc.stride[0] * dsize,
            ldst.shape[1] * dsize, ldst.shape[0],
            hipMemcpyDeviceToDevice, m_opr->stream()));

    return true;
};

bool RelayoutForwardImpl::Param::try_copy_last_contig() {
    //! check if the last stride is contiguous
    auto gcd = [](size_t a, size_t b) {
        if (a > b) std::swap(a, b);
        size_t c;
        while (a != 0) {
            c = a;
            a = b % a;
            b = c;
        }
        return b;
    };
    auto has_negative_stride = [](const TensorLayout& layout) {
        rep(i, layout.ndim) {
            if (layout.stride[i] < 0) return true;
        }
        return false;
    };

    TensorLayout lsrc = m_src.layout, ldst = m_dst.layout;
    if (lsrc.stride[lsrc.ndim - 1] == 1 && ldst.stride[ldst.ndim - 1] == 1 &&
            !has_negative_stride(lsrc) && !has_negative_stride(ldst)) {
        size_t contiguous_size =
            gcd(lsrc.shape[lsrc.ndim - 1], ldst.shape[ldst.ndim - 1]);
        if (contiguous_size > 1) {
            copy_last_contiguous(m_dst, m_src, contiguous_size,
                                 m_opr->stream());
            return true;
        }
    }
    return false;
}

void RelayoutForwardImpl::Param::copy_general() {

    copy_noncontig_general(m_dst, m_src, m_opr->stream());
}

void RelayoutForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                               Handle *src_handle) {
    bool cross_dev = false;

    // check whether cross device copy
    if (src_handle && src_handle != handle()) {
        megcoreDeviceHandle_t dev;
        megcoreGetDeviceHandle(src_handle->megcore_computing_handle(), &dev);
        megcorePlatform_t plat;
        megcoreGetPlatform(dev, &plat);
        megdnn_assert(plat == megcorePlatformROCM,
                      "only relayout between rocm devices are supported");
        int dst_dev_id = -1, src_dev_id = -1;
        megcoreGetDeviceID(dev, &src_dev_id);

        megcoreGetDeviceHandle(this->handle()->megcore_computing_handle(),
                               &dev);
        megcoreGetDeviceID(dev, &dst_dev_id);

        megdnn_assert(src_dev_id >= 0 && dst_dev_id >= 0);
        cross_dev = src_dev_id != dst_dev_id;
    }
    Param param{src, dst, this};
    if (!param.try_copy_contig() && !param.try_copy_2d() &&
            !param.try_copy_last_contig()) {
        megdnn_assert(!cross_dev,
                      "cross-device general non-contig copy unsupported");
        param.copy_general();
    }
}

// vim: syntax=cpp.doxygen

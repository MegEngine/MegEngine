/**
 * \file dnn/src/cuda/relayout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/relayout/kern.cuh"
#include "src/cuda/relayout/kern_contiguous.cuh"
#include "src/cuda/relayout/kern_transpose.cuh"

#include "src/common/relayout_helper.h"
#include "src/common/utils.h"
#include "src/cuda/relayout/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

RelayoutForwardImpl::Param::Param(
        const TensorND& src, const TensorND& dst, RelayoutForwardImpl* opr)
        : m_src{src}, m_dst{dst}, m_opr{opr} {
    opr->check_layout_and_canonize(m_src.layout, m_dst.layout);
}

bool RelayoutForwardImpl::Param::try_transpose() {
    if (m_dst.layout.dtype.is_low_bit())
        return false;
    relayout::TransposeParam transp;
    bool trans = relayout::is_transpose(m_src.layout, m_dst.layout, transp);
    if (!trans)
        return false;
    size_t dsize = transp.c * m_src.layout.dtype.size();
    if (dsize != 1 && dsize != 2 && dsize != 4)
        return false;

    if (m_src.layout.dtype == dtype::Float32() && transp.batch == 1 && transp.c == 1) {
        auto handle = concrete_handle(m_opr->handle());
        cublas_check(cublasSgeam(
                handle->cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T, transp.m, transp.n,
                handle->one_device(), m_src.ptr<dt_float32>(), transp.n,
                handle->zero_device(), m_src.ptr<dt_float32>(), transp.n,
                m_dst.ptr<dt_float32>(), transp.m));
        return true;
    }
    float square_ratio = static_cast<float>(transp.m) / static_cast<float>(transp.n);
    if (transp.m < 32 || transp.n < 32 || square_ratio < 0.5f || square_ratio > 2.f)
        return false;
    size_t batch = transp.batch, m = transp.m, n = transp.n;
    size_t lda = n, ldb = m, stride_A = m * n, stride_B = m * n;
    auto&& stream = m_opr->stream();
#define RUN(_dt)                                                                \
    do {                                                                        \
        typedef DTypeTrait<dtype::_dt>::ctype ctype;                            \
        copy_by_transpose<ctype>(                                               \
                reinterpret_cast<const ctype*>(m_src.raw_ptr),                  \
                reinterpret_cast<ctype*>(m_dst.raw_ptr), batch, m, n, lda, ldb, \
                stride_A, stride_B, stream);                                    \
        return true;                                                            \
    } while (0)
    switch (dsize) {
        case 1:
            RUN(Int8);
        case 2:
            RUN(Float16);
        case 4:
            RUN(Int32);
    }
    megdnn_assert(0, "bad dtype size");
}

bool RelayoutForwardImpl::Param::try_copy_contig() {
    auto &&lsrc = m_src.layout, &&ldst = m_dst.layout;
    if (lsrc.ndim != 1 || ldst.ndim != 1)
        return false;
    if (lsrc.stride[0] != 1 || ldst.stride[0] != 1)
        return false;
    size_t copy_size = ldst.span().dist_byte();

    cuda_check(cudaMemcpyAsync(
            m_dst.raw_ptr, m_src.raw_ptr, copy_size, cudaMemcpyDeviceToDevice,
            m_opr->stream()));
    return true;
}

bool RelayoutForwardImpl::expand_dim2(TensorLayout& dst, const TensorLayout& src) {
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

bool RelayoutForwardImpl::Param::try_copy_2d(bool cross_dev) {
    TensorLayout lsrc = m_src.layout, ldst = m_dst.layout;

    if (lsrc.ndim > 2 || ldst.ndim > 2)
        return false;
    if (ldst.dtype.is_low_bit())
        return false;

    if (ldst.ndim == 1 && lsrc.ndim == 1) {
        megdnn_assert(ldst.stride[0] != 1 || lsrc.stride[0] != 1);
        if (lsrc.stride[0] < 1 || ldst.stride[0] < 1 || !cross_dev)
            // test case: src=16x128x128(49152, 384, 3), dst=16x128x128(16384, 128, 1)
            // for both src and dst are one-dimensional, and one of them are not
            // contiguous, the relayout opr will call cudaMemcpy2DAsync, and the
            // bandwidth=5GiB/s. it is better to call copy_general, the
            // bandwidth=100GiB/s. call cudaMemcpy2DAsync when cross_dev, OR return
            // false and call copy_general.
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
    if (ldst.stride[1] != 1 || lsrc.stride[1] != 1 || ldst.shape[0] != lsrc.shape[0] ||
        ldst.shape[1] != lsrc.shape[1] ||
        ldst.stride[0] < static_cast<ptrdiff_t>(ldst.shape[1]) ||
        lsrc.stride[0] < static_cast<ptrdiff_t>(ldst.shape[1]))
        return false;

    auto dsize = dtype_size();
    cuda_check(cudaMemcpy2DAsync(
            m_dst.raw_ptr, ldst.stride[0] * dsize, m_src.raw_ptr,
            lsrc.stride[0] * dsize, ldst.shape[1] * dsize, ldst.shape[0],
            cudaMemcpyDeviceToDevice, m_opr->stream()));

    return true;
};

bool RelayoutForwardImpl::Param::try_copy_last_contig() {
    if (m_dst.layout.dtype.is_low_bit())
        return false;
    //! check if the last stride is contiguous
    auto gcd = [](size_t a, size_t b) {
        if (a > b)
            std::swap(a, b);
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
            if (layout.stride[i] < 0)
                return true;
        }
        return false;
    };

    TensorLayout lsrc = m_src.layout, ldst = m_dst.layout;
    if (lsrc.stride[lsrc.ndim - 1] == 1 && ldst.stride[ldst.ndim - 1] == 1 &&
        !has_negative_stride(lsrc) && !has_negative_stride(ldst)) {
        size_t contiguous_size =
                gcd(lsrc.shape[lsrc.ndim - 1], ldst.shape[ldst.ndim - 1]);
        // FIXME: disable copy_last_contiguous when contiguous_size < 32 due to
        // performance issue
        if (contiguous_size >= 32) {
            copy_last_contiguous(m_dst, m_src, contiguous_size, m_opr->stream());
            return true;
        }
    }
    return false;
}

void RelayoutForwardImpl::Param::copy_general() {
    copy_noncontig_general(m_dst, m_src, m_opr->stream());
}

void RelayoutForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) {
    bool cross_dev = false;

    // check whether cross device copy
    if (src_handle && src_handle != handle()) {
        megcoreDeviceHandle_t dev;
        megcoreGetDeviceHandle(src_handle->megcore_computing_handle(), &dev);
        megcorePlatform_t plat;
        megcoreGetPlatform(dev, &plat);
        megdnn_throw_if(
                plat != megcorePlatformCUDA, megdnn_error,
                "only relayout between cuda devices are supported");
        int dst_dev_id = -1, src_dev_id = -1;
        megcoreGetDeviceID(dev, &src_dev_id);

        megcoreGetDeviceHandle(this->handle()->megcore_computing_handle(), &dev);
        megcoreGetDeviceID(dev, &dst_dev_id);

        megdnn_assert(src_dev_id >= 0 && dst_dev_id >= 0);
        cross_dev = src_dev_id != dst_dev_id;
    }
    Param param{src, dst, this};
    if (!param.try_transpose() && !param.try_copy_contig() &&
        !param.try_copy_2d(cross_dev) && !param.try_copy_last_contig()) {
        megdnn_assert(!cross_dev, "cross-device general non-contig copy unsupported");
        param.copy_general();
    }
}

// vim: syntax=cpp.doxygen

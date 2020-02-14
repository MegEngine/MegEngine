/**
 * \file dnn/src/naive/local/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class LocalForwardImpl: public LocalForward {
    public:
        using LocalForward::LocalForward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in filter,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }

        struct FloatNoncontigBatchKernParam {
            const void *src;
            const void *filter;
            void *dst;
            size_t n,
                   ic, ih, iw, oc, oh, ow,
                   fh, fw;
            uint32_t ph, pw, sh, sw;
            ptrdiff_t inp_bs, out_bs;   //!< stride for batch of input, output
            void *workspace;
        };
        typedef void (*float_noncontig_batch_kern)(
                const FloatNoncontigBatchKernParam &);

        FloatNoncontigBatchKernParam make_float_kern_param(
                _megdnn_tensor_in src,
                _megdnn_tensor_in filter,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) const;

        /*!
         * \brief get function address for computing kernel
         *
         * Used by GroupLocal impl, to compute on input tensors with
         * non-contiguous src/dst batch dimension
         *
         * No need to validate input \p src, \p filter and \p dst are used to
         * give actual input shapes for kerel dispatching.
         */
        virtual float_noncontig_batch_kern dispatch_float_noncontig_batch(
                const TensorLayout &src,
                const TensorLayout &filter,
                const TensorLayout &dst);

    protected:
        //! implement exec() using kernel returned by
        //! dispatch_f32_noncontig_batch()
        void exec_use_float_noncontig_batch(_megdnn_tensor_in src,
                _megdnn_tensor_in filter,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace);

    private:
        template<bool is_xcorr, typename dtype>
        static void naive_kern(const FloatNoncontigBatchKernParam &param);
};

class LocalBackwardDataImpl: public LocalBackwardData {
    public:
        using LocalBackwardData::LocalBackwardData;
        void exec(_megdnn_tensor_in filter,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

class LocalBackwardFilterImpl: public LocalBackwardFilter {
    public:
        using LocalBackwardFilter::LocalBackwardFilter;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

} // namespace naive
} // namespace megdnn

//! define local variables for fields in LocalImpl::FloatNoncontigBatchKernParam
#define UNPACK_LOCAL_FLOAT_NONCONTIG_BATCH_KERN_PARAM(_p, _dtype)       \
    const _dtype* src = static_cast<const _dtype*>(_p.src);        \
    const _dtype* filter = static_cast<const _dtype*>(_p.filter);  \
    _dtype* dst = static_cast<_dtype*>(_p.dst);                    \
    _dtype* workspace = static_cast<_dtype*>(_p.workspace);        \
    const int N = _p.n, IC = _p.ic, IH = _p.ih, IW = _p.iw, OC = _p.oc, \
              OH = _p.oh, OW = _p.ow, FH = _p.fh, FW = _p.fw;           \
    const uint32_t PH = _p.ph, PW = _p.pw, SH = _p.sh, SW = _p.sw;      \
    const ptrdiff_t INP_BS = _p.inp_bs, OUT_BS = _p.out_bs

// vim: syntax=cpp.doxygen

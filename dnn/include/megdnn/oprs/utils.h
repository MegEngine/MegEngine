/**
 * \file dnn/include/megdnn/oprs/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

//! base class for random number generators
class RNGBase: public OperatorBase {
    DEF_OPR_IMPL_CTOR(RNGBase, OperatorBase);
    public:
        virtual void exec(_megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &dst, size_t workspace_in_bytes);
};

//! sample from uniform distribution on the interval (0, 1]
class UniformRNG: public RNGBase {
    DEF_OPR_IMPL(UniformRNG, RNGBase, 0, 1);
    DEF_OPR_PARAM(UniformRNG);
};

//! sample from gaussian distribution
class GaussianRNG: public RNGBase {
    DEF_OPR_IMPL(GaussianRNG, RNGBase, 0, 1);
    DEF_OPR_PARAM(GaussianRNG);
};

/*!
 * \brief sleep for specific time on the computing device; useful for testing
 *      async problems
 */
class SleepForward: public OperatorBase {
    DEF_OPR_IMPL(SleepForward, OperatorBase, 0, 0);
    DEF_OPR_PARAM(Sleep);

    public:
        virtual void exec() = 0;
};
using Sleep = SleepForward;

/*!
 * \brief calculating checksum of a tensor
 *
 * data must be a one-dimensional contiguous tensor with dtype byte
 */
class ChecksumForward: public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(ChecksumForward, OperatorBase, 0, 1);

    public:
        using Result = opr_result::Checksum;

        virtual size_t get_workspace_in_bytes(const TensorLayout &data) = 0;

        virtual Result exec(_megdnn_tensor_in data,
                _megdnn_workspace workspace) = 0;

    protected:
        void check_exec(const TensorLayout &layout, size_t workspace_in_bytes);
};
using Checksum = ChecksumForward;

/*!
 * \brief calculating max absolute difference of the two input tensors
 *
 * src1 and src2 must be a one-dimensional contiguous tensor.
 */
class MaxTensorDiff : public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(MaxTensorDiff, OperatorBase, 0, 2);

    public:
        virtual size_t get_workspace_in_bytes(const TensorLayout& layout1,
                                              const TensorLayout& layout2) = 0;

        virtual float exec(_megdnn_tensor_in src1, _megdnn_tensor_in src2,
                           _megdnn_workspace workspace) = 0;

    protected:
        void check_exec(const TensorLayout& layout1,
                        const TensorLayout& layout2, size_t workspace_in_bytes);
};

/*!
 * \brief winograd preprocess opr.
 *
 * for the detail \see src/fallback/conv_bias/winograd/winograd.h
 *
 */
class WinogradFilterPreprocess : public OperatorBase {
    DEF_OPR_PARAM(Winograd);
    DEF_OPR_IMPL(WinogradFilterPreprocess, OperatorBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace) = 0;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&);

    void deduce_layout(const TensorLayout& src, TensorLayout& dst);

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
}  // namespace megdnn

#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen

/**
 * \file src/opr/include/megbrain/opr/search_policy/profile.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/utils/hash_ct.h"
#include "megbrain/utils/timer.h"
#include "megbrain/system.h"
#include "megbrain/comp_node.h"

#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {

#define MGB_FOREACH_FASTRUN_OPR(cb)   \
    cb(ConvolutionForward);           \
    cb(ConvBiasForward);              \
    cb(ConvolutionBackwardData);      \
    cb(ConvolutionBackwardFilter);    \
    cb(Convolution3DForward);         \
    cb(Convolution3DBackwardData);    \
    cb(Convolution3DBackwardFilter);  \
    cb(LocalShareForward);            \
    cb(LocalShareBackwardData);       \
    cb(LocalShareBackwardFilter);     \
    cb(DeformableConvForward);        \
    cb(DeformableConvBackwardFilter); \
    cb(DeformableConvBackwardData);   \
    cb(BatchConvBiasForward);

template <typename Opr>
struct OprArityTrait;

template <typename Opr, int _arity_in, int _arity_out>
struct OprArityTraitTmpl {
    static constexpr int arity_in = _arity_in;
    static constexpr int arity_out = _arity_out;
    static constexpr int arity = arity_in + arity_out;
};

#define INST_ARITY(_Opr, _in, _out) \
    template <>                     \
    struct OprArityTrait<_Opr> : public OprArityTraitTmpl<_Opr, _in, _out> {};

INST_ARITY(megdnn::ConvolutionBackwardData, 2, 1);
INST_ARITY(megdnn::ConvolutionBackwardFilter, 2, 1);
INST_ARITY(megdnn::Convolution3DForward, 2, 1);
INST_ARITY(megdnn::Convolution3DBackwardData, 2, 1);
INST_ARITY(megdnn::Convolution3DBackwardFilter, 2, 1);
INST_ARITY(megdnn::LocalShareForward, 2, 1);
INST_ARITY(megdnn::LocalShareBackwardData, 2, 1);
INST_ARITY(megdnn::LocalShareBackwardFilter, 2, 1);
INST_ARITY(megdnn::Convolution, 2, 1);
INST_ARITY(megdnn::DeformableConvForward, 4, 1);
INST_ARITY(megdnn::DeformableConvBackwardFilter, 4, 1);
INST_ARITY(megdnn::BatchConvBiasForward, 4, 1);
INST_ARITY(megdnn::ConvBias, 4, 1);
INST_ARITY(megdnn::DeformableConvBackwardData, 5, 3);

#undef INST_ARITY

template <typename Opr>
constexpr bool opr_supports_preprocess() {
    return std::is_same<Opr, megdnn::ConvolutionForward>::value ||
           std::is_same<Opr, megdnn::ConvBias>::value;
}

template <typename Opr>
constexpr bool opr_contain_bias() {
    return std::is_same<Opr, megdnn::ConvBias>::value;
}

template <typename Opr, bool has_prep>
struct PreprocessFilterImpl {
    using T = union {};
};

template <typename Opr>
struct PreprocessFilterImpl<Opr, true> {
    using T = typename Opr::PreprocessedFilter;
};

template <typename Opr>
using PreprocessFilter =
        typename PreprocessFilterImpl<Opr, opr_supports_preprocess<Opr>()>::T;

template <typename Opr>
struct AlgoChooserFuncId {};

#define DEF_FUNC_ID(func)                                                     \
    template <>                                                               \
    struct AlgoChooserFuncId<megdnn::func> {                                  \
        __attribute__(                                                        \
                (unused)) static constexpr sys::TimedFuncInvoker::FuncId ID = \
                static_cast<sys::TimedFuncInvoker::FuncId>(                   \
                        MGB_HASH_STR("megdnn::" #func));                      \
    };

MGB_FOREACH_FASTRUN_OPR(DEF_FUNC_ID)

#undef DEF_FUNC_ID

/* =================== TimedProfiler =================== */

/*!
 * \brief profile a megdnn opr conv with given param
 *
 * This class only provides static methods, and the entry point is
 * TimedProfiler::profile; it would run profiler in a timed environment by
 * sys::TimedFuncInvoker
 *
 * \tparam Opr megdnn opr impl
 */
template <typename Opr>
class TimedProfiler {
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    static constexpr int arity_out = OprArityTrait<Opr>::arity_out;
    static constexpr int arity = OprArityTrait<Opr>::arity;

    using TensorShapeArray = std::array<megdnn::TensorShape, arity>;

public:
    struct Param {
        char algo_name[128];
        size_t workspace;
        megdnn::DTypeEnum dtypes[arity];
        CompNode::Locator comp_node_loc;
        TensorShapeArray shapes;
        typename Opr::Param opr_param;
        bool allow_weight_preprocess;

        //! filled by profile()
        mutable double actual_timeout;
    };

    struct Result {
        double time;
    };

    static Maybe<Result> profile(const Param& param, double& timeout);

private:
    using TParam = sys::TimedFuncInvoker::Param;
    using TResult = sys::TimedFuncInvoker::Result;

    static const double timeout_setting;

    static double init_timeout_setting();
    static TResult prof_impl(const TParam& raw_param);
    static void prof_init_device(const TParam& raw_param);
};
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

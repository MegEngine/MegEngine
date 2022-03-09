/**
 * \file src/opr/include/megbrain/opr/search_policy/algo_chooser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <memory>
#include "megbrain/graph/cg.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/rdnn/algo_chooser.h"
#include "megdnn/oprs/base.h"

template <class MegDNNOpr>
struct MegDNNOpr2MGBOpr;

#define cb(_Opr)                            \
    template <>                             \
    struct MegDNNOpr2MGBOpr<megdnn::_Opr> { \
        using MGBOpr = mgb::opr::_Opr;      \
    };

DNN_FOREACH_FASTRUN_OPR(cb)

#undef cb

#define MGB_FOREACH_FASTRUN_OPR(cb) DNN_FOREACH_FASTRUN_OPR(cb)

namespace mgb {
namespace opr {

/* =================== AlgoChooser =================== */
/*!
 * \brief choose algorithm according to ExecutionPolicy
 *
 * This class only provides static methods, and the entry point is
 * AlgoChooser::setup_algo. When profiling is needed, it would first try to
 * retrive profiling stats from cache, and run TimedProfiler when necessary
 *
 * \tparam Opr megdnn operator impl
 */
template <typename Opr>
class AlgoChooser : public rdnn::AlgoChooser<Opr> {
    using Base = rdnn::AlgoChooser<Opr>;
    using MGBOpr = typename MegDNNOpr2MGBOpr<Opr>::MGBOpr;
    using ImplExecutionPolicy = typename Base::ImplExecutionPolicy;

public:
    using AlgoChooserHelper = typename Base::AlgoChooserHelper;
    using FixedTensorLayouts = typename Base::FixedTensorLayouts;
    /*!
     * \brief setup algorithm and return workspace size
     */
    static size_t setup_algo(
            const FixedTensorLayouts& layouts, Opr* megdnn_opr, const MGBOpr* mgb_opr,
            bool allow_weight_preprocess = false);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

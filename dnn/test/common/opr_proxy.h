/**
 * \file dnn/test/common/opr_proxy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "test/common/deduce_layout_proxy.h"
#include "test/common/exec_proxy.h"
#include "test/common/inspect_type.h"
#include "test/common/opr_trait.h"
#include "test/common/timer.h"
#include "test/common/workspace_wrapper.h"

#include <algorithm>

namespace megdnn {
namespace test {

template <typename Opr, size_t arity = OprTrait<Opr>::arity,
          bool has_workspace = OprTrait<Opr>::has_workspace,
          bool can_deduce_layout = OprTrait<Opr>::can_deduce_layout>
struct OprProxyDefaultImpl
        : public DeduceLayoutProxy<Opr, arity, can_deduce_layout>,
          public ExecProxy<Opr, arity, has_workspace> {};

template <typename Opr>
struct OprProxy : public OprProxyDefaultImpl<Opr> {};

template <typename Opr>
struct OprProxyVectorToSingle {};

template <>
struct OprProxy<ElemwiseForward> {
    static void deduce_layout(ElemwiseForward* opr,
                              TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() >= 2);
        auto inp = layouts;
        inp.pop_back();
        opr->deduce_layout(inp, layouts.back());
    }

    static void exec(ElemwiseForward* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto inp = tensors;
        inp.pop_back();
        opr->exec(inp, tensors.back());
    }
};

template <>
struct OprProxy<ElemwiseMultiType> {
    static void deduce_layout(ElemwiseMultiType* opr,
                              TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() >= 2);
        auto inp = layouts;
        inp.pop_back();
        opr->deduce_layout(inp, layouts.back());
    }

    static void exec(ElemwiseMultiType* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto inp = tensors;
        inp.pop_back();
        opr->exec(inp, tensors.back());
    }
};

template <>
struct OprProxy<ConcatForward> {
    static void deduce_layout(ConcatForward* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() >= 2);
        auto inp = layouts;
        inp.pop_back();
        opr->deduce_layout(inp, layouts.back());
    }

    static void exec(ConcatForward* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto inp = tensors;
        inp.pop_back();

        TensorLayoutArray layouts(tensors.size());
        std::transform(tensors.begin(), tensors.end(), layouts.begin(),
                       [](const TensorND& tensor) { return tensor.layout; });
        auto inp_layouts = layouts;
        inp_layouts.pop_back();

        WorkspaceWrapper W(opr->handle(), opr->get_workspace_in_bytes(
                                                  inp_layouts, layouts.back()));

        auto inp_tensors = tensors;
        inp_tensors.pop_back();
        opr->exec(inp_tensors, tensors.back(), W.workspace());
    }
};

template <>
struct OprProxy<SplitForward> : DeduceLayoutProxy<SplitForward, 0, false> {
    static void exec(SplitForward* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto out = tensors;
        out.erase(out.begin());

        TensorLayoutArray layouts(tensors.size());
        std::transform(tensors.begin(), tensors.end(), layouts.begin(),
                       [](const TensorND& tensor) { return tensor.layout; });
        auto out_layouts = layouts;
        out_layouts.erase(out_layouts.begin());

        WorkspaceWrapper W(
                opr->handle(),
                opr->get_workspace_in_bytes(layouts.front(), out_layouts));

        auto out_tensors = tensors;
        out_tensors.erase(out_tensors.begin());
        opr->exec(tensors.front(), out_tensors, W.workspace());
    }
};

//! OprProxy impl for tenary oprs with profiling support
template <class Opr, int arity>
struct OprProxyProfilingBase
        : public DeduceLayoutProxy<Opr, arity,
                                   OprTrait<Opr>::can_deduce_layout> {
    size_t warmup_times = 10, exec_times = 100;

    //! whether to enable profiling
    bool m_profiling;
    WorkspaceWrapper W;

    //! target algo setup by profiler; it can also be directly specified by the
    //! caller
    typename Opr::Algorithm* target_algo = nullptr;

    OprProxyProfilingBase(bool profile = false) { m_profiling = profile; }
};

template <class Opr>
struct OprProxyProfilingTernary : public OprProxyProfilingBase<Opr, 3> {
    using Base = OprProxyProfilingBase<Opr, 3>;
    using OprProxyProfilingBase<Opr, 3>::OprProxyProfilingBase;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == 3);
        if (!Base::W.valid()) {
            Base::W = WorkspaceWrapper(opr->handle(), 0);
        }
        if (Base::m_profiling && !Base::target_algo) {
            size_t min_time = std::numeric_limits<size_t>::max();
            for (auto algo :
                 opr->get_all_algorithms(tensors[0].layout, tensors[1].layout,
                                         tensors[2].layout)) {
                opr->execution_policy().algorithm = algo;
                auto workspace_size = opr->get_workspace_in_bytes(
                        tensors[0].layout, tensors[1].layout,
                        tensors[2].layout);
                Base::W.update(workspace_size);

                for (size_t times = 0; times < Base::warmup_times; ++times)
                    opr->exec(tensors[0], tensors[1], tensors[2],
                              Base::W.workspace());
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                Timer timer;
                timer.start();
                for (size_t times = 0; times < Base::exec_times; ++times) {
                    opr->exec(tensors[0], tensors[1], tensors[2],
                              Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                timer.stop();
                printf("%.3fms %s\n", timer.get_time_in_us() / 1e3,
                       algo->name());
                if (min_time > timer.get_time_in_us()) {
                    min_time = timer.get_time_in_us();
                    Base::target_algo = algo;
                }
            }
            opr->execution_policy().algorithm = Base::target_algo;
            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors[0].layout, tensors[1].layout, tensors[2].layout);
            Base::W.update(workspace_size);
        }
        if (!Base::target_algo) {
            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors[0].layout, tensors[1].layout, tensors[2].layout);
            Base::W.update(workspace_size);
        }
        opr->exec(tensors[0], tensors[1], tensors[2], Base::W.workspace());
    }
};

#define DEF_PROF3(c)                                                 \
    template <>                                                      \
    struct OprProxy<c> : public OprProxyProfilingTernary<c> {        \
        using OprProxyProfilingTernary<c>::OprProxyProfilingTernary; \
    }

DEF_PROF3(ConvolutionForward);
DEF_PROF3(ConvolutionBackwardData);
DEF_PROF3(ConvolutionBackwardFilter);
DEF_PROF3(LocalShareForward);
DEF_PROF3(LocalShareBackwardData);
DEF_PROF3(LocalShareBackwardFilter);

#undef DEF_PROF3

template <class Opr>
struct OprProxyProfiling5 : public OprProxyProfilingBase<Opr, 5> {
    using Base = OprProxyProfilingBase<Opr, 5>;
    using OprProxyProfilingBase<Opr, 5>::OprProxyProfilingBase;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == 5);
        if (!Base::W.valid()) {
            Base::W = WorkspaceWrapper(opr->handle(), 0);
        }
        if (Base::m_profiling && !Base::target_algo) {
            size_t min_time = std::numeric_limits<size_t>::max();
            for (auto algo :
                 opr->get_all_algorithms(tensors[0].layout, tensors[1].layout,
                                         tensors[2].layout, tensors[3].layout,
                                         tensors[4].layout)) {
                opr->execution_policy().algorithm = algo;
                auto workspace_size = opr->get_workspace_in_bytes(
                        tensors[0].layout, tensors[1].layout, tensors[2].layout,
                        tensors[3].layout, tensors[4].layout);
                Base::W.update(workspace_size);

                for (size_t times = 0; times < Base::warmup_times; ++times)
                    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3],
                              tensors[4], Base::W.workspace());
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                Timer timer;
                timer.start();
                for (size_t times = 0; times < Base::exec_times; ++times) {
                    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3],
                              tensors[4], Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                timer.stop();
                printf("%.3fms %s\n", timer.get_time_in_us() / 1e3,
                       algo->name());
                if (min_time > timer.get_time_in_us()) {
                    min_time = timer.get_time_in_us();
                    Base::target_algo = algo;
                }
            }
            opr->execution_policy().algorithm = Base::target_algo;
            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors[0].layout, tensors[1].layout, tensors[2].layout,
                    tensors[3].layout, tensors[4].layout);
            Base::W.update(workspace_size);
        }
        if (!Base::target_algo) {
            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors[0].layout, tensors[1].layout, tensors[2].layout,
                    tensors[3].layout, tensors[4].layout);
            Base::W.update(workspace_size);
        }
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                  Base::W.workspace());
    }
};

#define DEF_PROF5(c)                                     \
    template <>                                          \
    struct OprProxy<c> : public OprProxyProfiling5<c> {  \
        using OprProxyProfiling5<c>::OprProxyProfiling5; \
    }

DEF_PROF5(DeformableConvForward);
DEF_PROF5(DeformableConvBackwardFilter);
DEF_PROF5(ConvBiasForward);
DEF_PROF5(BatchConvBiasForward);
#undef DEF_PROF5

template <class Opr>
struct OprProxyProfiling8 : public OprProxyProfilingBase<Opr, 8> {
    using Base = OprProxyProfilingBase<Opr, 8>;
    using OprProxyProfilingBase<Opr, 8>::OprProxyProfilingBase;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == 8);
        if (!Base::W.valid()) {
            Base::W = WorkspaceWrapper(opr->handle(), 0);
        }
        if (Base::m_profiling && !Base::target_algo) {
            size_t min_time = std::numeric_limits<size_t>::max();
            for (auto algo : opr->get_all_algorithms(
                         tensors[0].layout, tensors[1].layout,
                         tensors[2].layout, tensors[3].layout,
                         tensors[4].layout, tensors[5].layout,
                         tensors[6].layout, tensors[7].layout)) {
                opr->execution_policy().algorithm = algo;
                auto workspace_size = opr->get_workspace_in_bytes(
                        tensors[0].layout, tensors[1].layout, tensors[2].layout,
                        tensors[3].layout, tensors[4].layout, tensors[5].layout,
                        tensors[6].layout, tensors[7].layout);
                Base::W.update(workspace_size);

                for (size_t times = 0; times < Base::warmup_times; ++times)
                    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3],
                              tensors[4], tensors[5], tensors[6], tensors[7],
                              Base::W.workspace());
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                Timer timer;
                timer.start();
                for (size_t times = 0; times < Base::exec_times; ++times) {
                    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3],
                              tensors[4], tensors[5], tensors[6], tensors[7],
                              Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                timer.stop();
                printf("%.3fms %s\n", timer.get_time_in_us() / 1e3,
                       algo->name());
                if (min_time > timer.get_time_in_us()) {
                    min_time = timer.get_time_in_us();
                    Base::target_algo = algo;
                }
            }
            opr->execution_policy().algorithm = Base::target_algo;
            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors[0].layout, tensors[1].layout, tensors[2].layout,
                    tensors[3].layout, tensors[4].layout, tensors[5].layout,
                    tensors[6].layout, tensors[7].layout);
            Base::W.update(workspace_size);
        }
        if (!Base::target_algo) {
            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors[0].layout, tensors[1].layout, tensors[2].layout,
                    tensors[3].layout, tensors[4].layout, tensors[5].layout,
                    tensors[6].layout, tensors[7].layout);
            Base::W.update(workspace_size);
        }
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                  tensors[5], tensors[6], tensors[7], Base::W.workspace());
    }
};

#define DEF_PROF8(c)                                     \
    template <>                                          \
    struct OprProxy<c> : public OprProxyProfiling8<c> {  \
        using OprProxyProfiling8<c>::OprProxyProfiling8; \
    }

DEF_PROF8(DeformableConvBackwardData);

#undef DEF_PROF8
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen

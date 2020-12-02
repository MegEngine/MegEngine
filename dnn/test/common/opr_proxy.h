/**
 * \file dnn/test/common/opr_proxy.h
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

#include "test/common/deduce_layout_proxy.h"
#include "test/common/exec_proxy.h"
#include "test/common/inspect_type.h"
#include "test/common/opr_algo_proxy.h"
#include "test/common/opr_trait.h"
#include "test/common/timer.h"
#include "test/common/workspace_wrapper.h"

#include <algorithm>
#include <memory>

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
struct OprWeightPreprocessProxy : public OprProxyDefaultImpl<Opr> {};

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
    typename Opr::AlgorithmInfo target_algo_info;

    OprProxyProfilingBase(bool profile = false) { m_profiling = profile; }

    //! used for alloc tensor for weight preprocess
    static std::shared_ptr<TensorNDArray> alloc_tensors(
            Handle* handle, const TensorLayoutArray& layouts) {
        auto deleter = [handle](TensorNDArray* ptr) {
            for (auto&& i : *ptr) {
                auto pdata = static_cast<dt_byte*>(i.raw_ptr) +
                             i.layout.span().low_byte;
                megdnn_free(handle, pdata);
            }
            delete ptr;
        };
        std::shared_ptr<TensorNDArray> ret{new TensorNDArray, deleter};
        for (size_t i = 0; i < layouts.size(); ++i) {
            auto span = layouts[i].span();
            ret->emplace_back(static_cast<dt_byte*>(
                                      megdnn_malloc(handle, span.dist_byte())) -
                                      span.low_byte,
                              layouts[i]);
        }
        return ret;
    }

    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == arity);
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        TensorLayoutArray layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        if (m_profiling && !target_algo_info.valid()) {
            size_t min_time = std::numeric_limits<size_t>::max();
            for (auto algo :
                 AlgoProxy<Opr, arity>::get_all_algorithms_info(opr, layouts)) {
                opr->execution_policy().algo = algo;
                auto workspace_size =
                        AlgoProxy<Opr, arity>::get_workspace_in_bytes(opr,
                                                                      layouts);
                W.update(workspace_size);

                for (size_t times = 0; times < warmup_times; ++times)
                    AlgoProxy<Opr, arity>::exec(opr, tensors, W.workspace());
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                Timer timer;
                timer.start();
                for (size_t times = 0; times < exec_times; ++times) {
                    AlgoProxy<Opr, arity>::exec(opr, tensors, W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                timer.stop();
                printf("%.3fms %s\n", timer.get_time_in_us() / 1e3,
                       algo.name.c_str());
                if (min_time > timer.get_time_in_us()) {
                    min_time = timer.get_time_in_us();
                    target_algo_info = algo;
                }
            }
            opr->execution_policy().algo = target_algo_info;
            auto workspace_size =
                    AlgoProxy<Opr, arity>::get_workspace_in_bytes(opr, layouts);
            W.update(workspace_size);
        }
        if (!target_algo_info.valid()) {
            auto workspace_size =
                    AlgoProxy<Opr, arity>::get_workspace_in_bytes(opr, layouts);
            W.update(workspace_size);
        }
        AlgoProxy<Opr, arity>::exec(opr, tensors, W.workspace());
    }
};

#define DEF_PROF(c, arity)                                            \
    template <>                                                       \
    struct OprProxy<c> : public OprProxyProfilingBase<c, arity> {     \
        using OprProxyProfilingBase<c, arity>::OprProxyProfilingBase; \
    }

DEF_PROF(ConvolutionForward, 3);
DEF_PROF(ConvolutionBackwardData, 3);
DEF_PROF(ConvolutionBackwardFilter, 3);
DEF_PROF(LocalShareForward, 3);
DEF_PROF(LocalShareBackwardData, 3);
DEF_PROF(LocalShareBackwardFilter, 3);

DEF_PROF(DeformableConvForward, 5);
DEF_PROF(DeformableConvBackwardFilter, 5);
DEF_PROF(BatchConvBiasForward, 5);
DEF_PROF(ConvBiasForward, 5);

DEF_PROF(DeformableConvBackwardData, 8);
#undef DEF_PROF

template <class Opr, int arity>
struct OprWeightPreprocessProxyImpl : public OprProxyProfilingBase<Opr, arity> {
    using Base = OprProxyProfilingBase<Opr, arity>;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == arity);
        if (!Base::W.valid()) {
            Base::W = WorkspaceWrapper(opr->handle(), 0);
        }

        TensorLayoutArray layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        if (Base::m_profiling && !Base::target_algo_info.desc.valid()) {
            size_t min_time = std::numeric_limits<size_t>::max();
            for (auto algo :
                 AlgoProxy<Opr, arity>::get_all_algorithms_info(opr, layouts)) {
                opr->execution_policy().algo = algo;

                auto preprocess_tensors =
                        weight_prerocess(opr, tensors, algo.desc);
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                typename Opr::PreprocessedFilter preprocessed_filter{
                        nullptr, *preprocess_tensors};

                auto workspace_size =
                        AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                                opr, layouts, &preprocessed_filter);
                Base::W.update(workspace_size);

                for (size_t times = 0; times < Base::warmup_times; ++times) {
                    AlgoProxy<Opr, arity>::exec(opr, tensors,
                                                &preprocessed_filter,
                                                Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                Timer timer;
                timer.start();
                for (size_t times = 0; times < Base::exec_times; ++times) {
                    AlgoProxy<Opr, arity>::exec(opr, tensors,
                                                &preprocessed_filter,
                                                Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                timer.stop();
                printf("%.3fms %s\n", timer.get_time_in_us() / 1e3,
                       algo.name.c_str());
                if (min_time > timer.get_time_in_us()) {
                    min_time = timer.get_time_in_us();
                    Base::target_algo_info = algo;
                }
            }
            opr->execution_policy().algo = Base::target_algo_info;
            auto preprocess_tensors =
                    weight_prerocess(opr, tensors, Base::target_algo_info.desc);
            megcoreSynchronize(opr->handle()->megcore_computing_handle());
            typename Opr::PreprocessedFilter preprocessed_filter{
                    nullptr, *preprocess_tensors};
            auto workspace_size = AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                    opr, layouts, &preprocessed_filter);
            Base::W.update(workspace_size);
        }
        auto preprocess_tensors =
                weight_prerocess(opr, tensors, Base::target_algo_info.desc);
        megcoreSynchronize(opr->handle()->megcore_computing_handle());
        typename Opr::PreprocessedFilter preprocessed_filter{
                nullptr, *preprocess_tensors};
        if (!Base::target_algo_info.valid()) {
            auto workspace_size = AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                    opr, layouts, &preprocessed_filter);
            Base::W.update(workspace_size);
        }
        AlgoProxy<Opr, arity>::exec(opr, tensors, &preprocessed_filter,
                                    Base::W.workspace());
    }

    //! handle weight preprocess
    std::shared_ptr<TensorNDArray> weight_prerocess(
            Opr* opr, const TensorNDArray& tensors,
            const typename Opr::AlgorithmDesc&) {
        TensorLayoutArray layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        auto weight_perprocess_layouts =
                AlgoProxy<Opr, arity>::deduce_preprocessed_filter_layout(
                        opr, layouts);
        auto preprocessed_filter_tensors_ptr =
                Base::alloc_tensors(opr->handle(), weight_perprocess_layouts);
        typename Opr::PreprocessedFilter preprocessed_filter{
                nullptr, *preprocessed_filter_tensors_ptr};
        size_t preprocess_workspace_size =
                AlgoProxy<Opr, arity>::get_preprocess_workspace_in_bytes(
                        opr, layouts);
        WorkspaceWrapper preprocess_workspace(opr->handle(),
                                              preprocess_workspace_size);
        AlgoProxy<Opr, arity>::exec_preprocess(
                opr, tensors, layouts, &preprocessed_filter,
                preprocess_workspace.workspace());
        return preprocessed_filter_tensors_ptr;
    }
};

#define DEF_PROF(c, arity)                                    \
    template <>                                               \
    struct OprWeightPreprocessProxy<c>                        \
            : public OprWeightPreprocessProxyImpl<c, arity> { \
        using OprWeightPreprocessProxyImpl<                   \
                c, arity>::OprWeightPreprocessProxyImpl;      \
    }

DEF_PROF(ConvolutionForward, 3);
DEF_PROF(ConvBias, 5);
#undef DEF_PROF

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
